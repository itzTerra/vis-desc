import argparse
import gc
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tiktoken
import json
import re
from tqdm import tqdm
import torch
import torch.distributed as dist

from models.llm.agents import (
    ModelAgent,
    VLLMAgent,
    APIAgent,
    ModelConfig,
    EINFRA_MODELS,
    LOCAL_MODELS,
)
from models.llm.prompts import PROMPTS, Prompt, schema_for_prompt
from utils import DATA_DIR, PersistentDict, get_device_name
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 16
METRICS_DIR = DATA_DIR / "metrics" / "llm"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_MODELS = EINFRA_MODELS + LOCAL_MODELS
MODEL_BY_ID = {model.id: model for model in AVAILABLE_MODELS}
MODEL_BY_NAME = {model.name: model for model in AVAILABLE_MODELS}


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        encoding_name: Encoding to use (default: cl100k_base for GPT-4)

    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


class LLMPersistentMetrics(PersistentDict):
    """Persistent metrics storage for LLM model evaluations."""

    @classmethod
    def create_for_prompt(
        cls,
        prompt: Prompt,
    ) -> "LLMPersistentMetrics":
        """Create metrics file for a prompt using the format specified in metric-docs.md.

        Filename format: promptId_yyyy-MM-dd-hh-mm-ss.json
        - promptId: result of Prompt.get_id()
        - Timestamp: ISO 8601 format (yyyy-MM-dd-hh-mm-ss)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        prompt_id = prompt.get_id()
        filename = METRICS_DIR / f"{prompt_id}_{timestamp}.json"

        metrics = cls(filename)
        prompt_text = prompt.build_user_prompt("{{TEXT_SEGMENT}}")
        metrics.setdefault("prompt", prompt_text)
        metrics.setdefault("prompt_token_count", count_tokens(prompt_text))
        metrics.setdefault("models", [])
        return metrics


def calculate_performance_metrics(
    latencies: list[float],
) -> dict:
    """
    Calculate performance metrics from latency measurements.

    Args:
        latencies: List of per-sample latencies in milliseconds

    Returns:
        dict with keys: throughput, latency_mean, latency_std
    """
    latencies_arr = np.array(latencies)
    throughput = 1000.0 / np.mean(latencies_arr)

    return {
        "throughput": float(throughput),
        "latency_mean": float(np.mean(latencies_arr)),
        "latency_std": float(np.std(latencies_arr)),
    }


def parse_output(response: str) -> tuple[str | None, bool]:
    """
    Parse the rating from model output.

    Returns:
        Tuple of (parsed_rating, had_error)
        - parsed_rating: The extracted rating as string, or None if error
        - had_error: True if parsing failed
    """

    def finalize(value: str | None, had_error: bool) -> tuple[str | None, bool]:
        if value is None:
            return None, True
        if value not in {"0", "1", "2", "3", "4", "5"}:
            return None, True
        return value, had_error

    def fallback_from_text(text: str) -> tuple[str | None, bool]:
        # Fallback: take the last integer and last boolean (true/false) in text
        numbers = re.findall(r"(?<!-)\d+(?!-)", text)
        bools = re.findall(r"\b(?:true|false)\b", text, flags=re.IGNORECASE)
        if numbers:
            val = int(numbers[-1])
            bonus_applied = bools and bools[-1].lower() == "true"
            if bonus_applied:
                val = max(0, min(5, val + 1))
            return finalize(str(val), False)
        return finalize(None, True)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)

            rating = parsed.get("rating")
            if rating is None:
                rating = parsed.get("rating_without_action_bonus")

            if rating is not None:
                rating = int(rating)
                if parsed.get("action_bonus_applied") is True:
                    rating = max(0, min(5, rating + 1))
                return finalize(str(rating), False)
            # JSON parsed but no rating fields; try fallback extraction
            return fallback_from_text(response)
        # No JSON substring found; try fallback extraction
        return fallback_from_text(response)
    except (json.JSONDecodeError, Exception):
        # JSON parse failed; try fallback extraction
        return fallback_from_text(response)


def evaluate_model_on_prompt(
    agent: ModelAgent,
    prompt: Prompt,
    texts: list[str],
    metrics: "LLMPersistentMetrics",
    dataset_name: str,
    debug_parse: bool = False,
    use_structured_outputs: bool = True,
    structured_schema: dict | None = None,
) -> None:
    """
    Evaluate a model on a prompt in batches and update metrics after each batch.

    Args:
        agent: ModelAgent instance to evaluate
        prompt: Prompt object to use
        texts: List of texts to evaluate
        metrics: LLMPersistentMetrics instance for storing results
        dataset_name: Name of the dataset (train/test)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        debug_parse: If True, pause on parse failures and show raw output
    """
    time_start = datetime.now().isoformat()
    model_name = agent.model_name if hasattr(agent, "model_name") else "unknown"
    device = get_device_name()

    all_outputs = []
    all_latencies = []
    output_errors = 0

    user_prompts = [prompt.build_user_prompt(text) for text in texts]

    model_result = {
        "model_name": model_name,
        "dataset": dataset_name,
        "outputs": all_outputs,
        "output_errors": output_errors,
        "device": device,
        "batch_size": BATCH_SIZE,
        "performance": None,
        "time_start": time_start,
        "time_end": None,
    }
    metrics.setdefault("models", []).append(model_result)

    for i in tqdm(
        range(0, len(user_prompts), BATCH_SIZE),
        desc="Processing batches",
        leave=False,
        file=sys.stdout,
    ):
        batch = user_prompts[i : i + BATCH_SIZE]

        start = time.perf_counter()
        responses = agent.generate_batch(
            prompts=batch,
            system_prompt=prompt.system,
            use_structured_outputs=use_structured_outputs,
            structured_schema=structured_schema,
        )
        end = time.perf_counter()

        batch_latency_ms = (end - start) * 1000 / len(batch)
        all_latencies.extend([batch_latency_ms] * len(batch))

        for response in responses:
            parsed_output, had_error = parse_output(response)
            all_outputs.append(parsed_output)
            if had_error:
                output_errors += 1
                if debug_parse:
                    print("\n[DEBUG] Failed to parse model output:", file=sys.stderr)
                    print(response, file=sys.stderr)

        model_result["output_errors"] = output_errors
        model_result["performance"] = calculate_performance_metrics(all_latencies)
        model_result["time_end"] = datetime.now().isoformat()
        metrics._dump()


def create_agent_for_model(model_config: ModelConfig) -> ModelAgent:
    """Create appropriate agent based on model config.

    Args:
        model_config: ModelConfig instance to create agent for

    Returns:
        ModelAgent instance configured for the model
    """
    is_einfra = any(m.id == model_config.id for m in EINFRA_MODELS)
    is_local = any(m.id == model_config.id for m in LOCAL_MODELS)

    if is_einfra:
        base_url = os.environ.get("EINFRA_BASE_URL")
        api_key = os.environ.get("EINFRA_API_KEY")
        if not base_url or not api_key:
            raise ValueError(
                "EINFRA_BASE_URL and EINFRA_API_KEY environment variables must be set for eInfra models"
            )
        return APIAgent(model_config=model_config, api_key=api_key, base_url=base_url)
    elif is_local:
        return VLLMAgent(model_config=model_config)
    else:
        raise ValueError(f"Unknown model: {model_config.id}")


def cleanup_distributed_group() -> None:
    """Clean up PyTorch distributed process group to prevent reinitialization errors."""
    if dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_agent(agent: ModelAgent) -> None:
    """Clean up agent resources, particularly for vLLM models.

    Args:
        agent: ModelAgent instance to cleanup
    """
    if agent is None:
        return

    if hasattr(agent, "llm") and agent.llm is not None:
        if hasattr(agent.llm, "llm_engine") and hasattr(
            agent.llm.llm_engine, "driver_worker"
        ):
            try:
                agent.llm.llm_engine.driver_worker.shutdown()
            except Exception:
                pass

    del agent
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models on prompts and save results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model by name on prompt 0 with train set
  python run.py --dataset train --models Ministral3-14b

  # Evaluate all models on all prompts with test set
  python run.py --dataset test --models all --prompts -1

  # Evaluate all local models on prompt 0
  python run.py --dataset train --models local --prompts 0

  # Evaluate all einfra models on prompt 0
  python run.py --dataset train --models einfra --prompts 0

  # Evaluate on both train and test datasets
  python run.py --dataset both --models Ministral3-14b --prompts 0

  # Evaluate specific models by name on specific prompts
  python run.py --dataset train --models Ministral3-14b Gemma3-12b --prompts 0 1

  # Evaluate models by ID
  python run.py --dataset train --models unsloth/Ministral-3-14B-Instruct-2512-FP8

  # Use custom data file
  python run.py --dataset train --data-file /path/to/custom.parquet --models Ministral3-14b
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        choices=["train", "test", "both"],
        help="Dataset to evaluate on (use 'both' to evaluate on both train and test)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=False,
        help="Models to evaluate. Use model names (e.g., Ministral3-14b) or IDs (e.g., unsloth/Ministral-3-14B-Instruct-2512-FP8), or use 'all', 'local', 'einfra'",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        nargs="+",
        default=[0],
        help="Prompt indices to evaluate (0-based). Use -1 for all prompts (default: 0)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to parquet file with 'text' column. "
        "Default: DATA_DIR/datasets/small/{dataset}.parquet",
    )
    parser.add_argument(
        "--debug-parse",
        action="store_true",
        help="Pause on parse_output failures and show raw model output",
    )
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="Disable structured outputs (JSON schema-guided generation)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompts and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("\neInfra models:")
        for i, model in enumerate(EINFRA_MODELS, 1):
            print(f"  {i}. {model.name} ({model.id})")
        print("\nLocal models:")
        for i, model in enumerate(LOCAL_MODELS, 1):
            print(f"  {i}. {model.name} ({model.id})")
        sys.exit(0)

    if args.list_prompts:
        print(f"Available prompts ({len(PROMPTS)} total):")
        for i, prompt in enumerate(PROMPTS):
            print(f"  {i}. {prompt.get_id()}")
        sys.exit(0)

    if not args.dataset:
        parser.error(
            "--dataset is required unless using --list-models or --list-prompts"
        )

    if not args.models:
        parser.error(
            "--models is required unless using --list-models or --list-prompts"
        )

    if "all" in args.models:
        models_to_eval = AVAILABLE_MODELS
    elif "local" in args.models:
        models_to_eval = LOCAL_MODELS
    elif "einfra" in args.models:
        models_to_eval = EINFRA_MODELS
    else:
        models_to_eval = []
        for model_ref in args.models:
            if model_ref in MODEL_BY_NAME:
                models_to_eval.append(MODEL_BY_NAME[model_ref])
            elif model_ref in MODEL_BY_ID:
                models_to_eval.append(MODEL_BY_ID[model_ref])
            else:
                print(f"❌ Error: Unknown model '{model_ref}'")
                available = ", ".join([f"{m.name} ({m.id})" for m in AVAILABLE_MODELS])
                print(f"Available models: {available}")
                sys.exit(1)

    if -1 in args.prompts:
        prompts_to_eval = list(range(len(PROMPTS)))
    else:
        prompts_to_eval = args.prompts
        for idx in prompts_to_eval:
            if idx < 0 or idx >= len(PROMPTS):
                print(
                    f"❌ Error: Invalid prompt index {idx} (valid range: 0-{len(PROMPTS) - 1})"
                )
                sys.exit(1)

    if args.dataset == "both":
        datasets_to_eval = ["train", "test"]
    else:
        datasets_to_eval = [args.dataset]

    datasets = {}
    for dataset_name in datasets_to_eval:
        if args.data_file:
            data_path = Path(args.data_file)
        else:
            data_path = DATA_DIR / "datasets" / "small" / f"{dataset_name}.parquet"

        if not data_path.exists():
            print(f"❌ Error: Dataset file not found at {data_path}")
            sys.exit(1)

        df = pd.read_parquet(data_path)
        texts = df["text"].tolist()

        datasets[dataset_name] = {"texts": texts}
        print(f"✓ Loaded {len(texts)} samples from {dataset_name} set")

    print("\n📊 Evaluation Summary:")
    print(
        f"  Datasets to evaluate: {len(datasets_to_eval)} - {', '.join(datasets_to_eval)}"
    )
    models_str = ", ".join([f"{m.name} ({m.id})" for m in models_to_eval])
    print(f"  Models to evaluate: {len(models_to_eval)} - {models_str}")
    print(f"  Prompts to evaluate: {len(prompts_to_eval)} - {prompts_to_eval}")
    print(
        f"  Total evaluations: {len(datasets_to_eval) * len(models_to_eval) * len(prompts_to_eval)}"
    )
    print(f"  Results will be saved to: {METRICS_DIR}")
    print(f"  Structured outputs: {'OFF' if args.no_structured_outputs else 'ON'}")

    total_evals = len(datasets_to_eval) * len(models_to_eval) * len(prompts_to_eval)
    current_eval = 0
    successful_evals = 0
    metrics_files = {}

    login(token=os.environ.get("HF_TOKEN"))

    for model_config in models_to_eval:
        print(f"\n{'=' * 60}")
        print(f"🤖 Loading model: {model_config.name} ({model_config.id})")
        print(f"{'=' * 60}")

        agent = None
        try:
            agent = create_agent_for_model(model_config)
            print("✓ Model loaded successfully")

            for prompt_idx in prompts_to_eval:
                prompt = PROMPTS[prompt_idx]

                if prompt_idx not in metrics_files:
                    metrics = LLMPersistentMetrics.create_for_prompt(prompt)
                    metrics_files[prompt_idx] = metrics.filename
                else:
                    metrics = LLMPersistentMetrics(metrics_files[prompt_idx])

                for dataset_name in datasets_to_eval:
                    current_eval += 1
                    dataset_data = datasets[dataset_name]

                    print(
                        f"\n  [{current_eval}/{total_evals}] Evaluating {model_config.name} on prompt {prompt_idx} with {dataset_name} set..."
                    )
                    print(f"      Prompt ID: {prompt.get_id()}")

                    try:
                        evaluate_model_on_prompt(
                            agent,
                            prompt,
                            dataset_data["texts"],
                            metrics,
                            dataset_name,
                            args.debug_parse,
                            use_structured_outputs=(not args.no_structured_outputs),
                            structured_schema=schema_for_prompt(prompt),
                        )
                        print("      ✓ Metrics updated and persisted")
                        successful_evals += 1

                    except Exception as e:
                        print(
                            f"      ❌ Error evaluating {model_config.name} on prompt {prompt_idx} with {dataset_name}: {e}"
                        )
                        import traceback

                        traceback.print_exc()

        except Exception as e:
            print(f"❌ Error loading model {model_config.name}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            cleanup_agent(agent)
            cleanup_distributed_group()

    print(f"\n{'=' * 60}")
    print(f"✓ Evaluation complete! ({successful_evals}/{total_evals} successful)")
    print("✓ Metrics files saved to:")
    for filepath in metrics_files.values():
        print(f"  - {filepath}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import os

    main()
