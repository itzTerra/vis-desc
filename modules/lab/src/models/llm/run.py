import argparse
import enum
import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm
import torch
import torch.distributed as dist

from models.llm.agents import (
    AVAILABLE_MODELS,
    MODEL_BY_ID,
    MODEL_BY_NAME,
    ModelAgent,
    VLLMAgent,
    APIAgent,
    ModelConfig,
    EINFRA_MODELS,
    LOCAL_MODELS,
)
from models.llm.prompts import PROMPTS, OPTIMIZED_PROMPTS, Prompt, schema_for_prompt
from utils import DATA_DIR, PersistentDict, get_device_name
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

BATCH_SIZE = 16
METRICS_DIR = DATA_DIR / "metrics" / "llm"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


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
        metrics.setdefault("system", prompt.system)
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


class OutputParseStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK_SUCCESS = "fallback"


def parse_output(response: str) -> tuple[str | None, OutputParseStatus]:
    """
    Parse the rating from model output.

    Returns:
        Tuple of (parsed_rating, parse_status)
        - parsed_rating: The extracted rating as string, or None if error
        - parse_status: OutputParseStatus indicating success, failure, or fallback success
    """

    def finalize(
        value: str | None, had_error: bool
    ) -> tuple[str | None, OutputParseStatus]:
        if value is None:
            return None, OutputParseStatus.FAILED
        if value not in {"0", "1", "2", "3", "4", "5"}:
            return None, OutputParseStatus.FAILED
        return value, OutputParseStatus.SUCCESS

    def fallback_from_text(text: str) -> tuple[str | None, OutputParseStatus]:
        # Fallback: take the last integer in text
        numbers = re.findall(r"(?<!-)\d+(?!-)", text)
        if numbers:
            val = int(numbers[-1])
            return finalize(str(val), OutputParseStatus.FALLBACK_SUCCESS)
        return finalize(None, OutputParseStatus.FAILED)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)

            rating = parsed.get("rating")

            if rating is not None:
                rating = int(rating)
                return finalize(str(rating), OutputParseStatus.SUCCESS)
            # JSON parsed but no rating field; try fallback extraction
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
    seed: int,
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
        seed: Random seed for model sampling
        debug_parse: If True, pause on parse failures and show raw output
        use_structured_outputs: Whether to use structured outputs
        structured_schema: Schema for structured outputs
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
        "seed": seed,
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
            seed=seed,
        )
        end = time.perf_counter()

        batch_latency_ms = (end - start) * 1000 / len(batch)
        all_latencies.extend([batch_latency_ms] * len(batch))

        for response in responses:
            parsed_output, parse_status = parse_output(response)
            all_outputs.append(parsed_output)
            if parse_status == OutputParseStatus.FAILED:
                output_errors += 1
                if debug_parse:
                    logger.debug("Failed to parse model output: %r", response)
            elif parse_status == OutputParseStatus.FALLBACK_SUCCESS:
                if debug_parse:
                    logger.debug("Fallback parsed model output: %r", response)

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

  # Use optimized prompt set
  python run.py --dataset train --models Ministral3-14b --prompt-set optimized

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
        "--prompt-set",
        type=str,
        choices=["standard", "optimized"],
        default="standard",
        help="Which prompt set to use: 'standard' for PROMPTS (14 variants) or 'optimized' for OPTIMIZED_PROMPTS (default: standard)",
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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[40, 41, 42],
        help="Random seeds for model sampling (default: 40 41 42)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to file and show raw model output on parse failures",
    )

    args = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if args.debug:
        debug_log_dir = DATA_DIR / "output" / "debug_logs"
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        debug_log_path = debug_log_dir / "run_debug.log"

        file_handler = logging.FileHandler(debug_log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        logger.info(
            "Debug logging enabled. Logs will be written to: %s", debug_log_path
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    if args.list_models:
        logger.info("Available models:")
        logger.info("\neInfra models:")
        for i, model in enumerate(EINFRA_MODELS, 1):
            logger.info("  %d. %s (%s)", i, model.name, model.id)
        logger.info("\nLocal models:")
        for i, model in enumerate(LOCAL_MODELS, 1):
            logger.info("  %d. %s (%s)", i, model.name, model.id)
        sys.exit(0)

    if args.list_prompts:
        prompts = OPTIMIZED_PROMPTS if args.prompt_set == "optimized" else PROMPTS
        logger.info("Prompt Set: %s", args.prompt_set)
        logger.info("Available prompts (%d total):", len(prompts))
        for i, prompt in enumerate(prompts):
            logger.info("  %d. %s", i, prompt.get_id())
        sys.exit(0)

    if not args.dataset:
        parser.error(
            "--dataset is required unless using --list-models or --list-prompts"
        )

    if not args.models:
        parser.error(
            "--models is required unless using --list-models or --list-prompts"
        )

    selected_prompts = OPTIMIZED_PROMPTS if args.prompt_set == "optimized" else PROMPTS

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
                logger.error("Unknown model '%s'", model_ref)
                available = ", ".join([f"{m.name} ({m.id})" for m in AVAILABLE_MODELS])
                logger.error("Available models: %s", available)
                sys.exit(1)

    if -1 in args.prompts:
        prompts_to_eval = list(range(len(selected_prompts)))
    else:
        prompts_to_eval = args.prompts
        for idx in prompts_to_eval:
            if idx < 0 or idx >= len(selected_prompts):
                logger.error(
                    "Invalid prompt index %d (valid range: 0-%d)",
                    idx,
                    len(selected_prompts) - 1,
                )
                sys.exit(1)

    if args.dataset == "both":
        datasets_to_eval = ["train", "test"]
    else:
        datasets_to_eval = [args.dataset]

    datasets = {}
    for dataset_name in datasets_to_eval:
        data_path = DATA_DIR / "datasets" / "small" / f"{dataset_name}.parquet"

        if not data_path.exists():
            logger.error("Dataset file not found at %s", data_path)
            sys.exit(1)

        df = pd.read_parquet(data_path)
        texts = df["text"].tolist()

        datasets[dataset_name] = {"texts": texts}
        logger.info("✓ Loaded %d samples from %s set", len(texts), dataset_name)

    models_str = ", ".join([f"{m.name} ({m.id})" for m in models_to_eval])
    logger.info("\n📊 Evaluation Summary:")
    logger.info(
        "  Datasets to evaluate: %d - %s",
        len(datasets_to_eval),
        ", ".join(datasets_to_eval),
    )
    logger.info("  Models to evaluate: %d - %s", len(models_to_eval), models_str)
    logger.info("  Prompt set: %s", args.prompt_set)
    logger.info("  Prompts to evaluate: %d - %s", len(prompts_to_eval), prompts_to_eval)
    logger.info("  Seeds to use: %s", args.seeds)
    logger.info(
        "  Total evaluations: %d",
        len(datasets_to_eval)
        * len(models_to_eval)
        * len(prompts_to_eval)
        * len(args.seeds),
    )
    logger.info("  Results will be saved to: %s", METRICS_DIR)
    logger.info(
        "  Structured outputs: %s", "OFF" if args.no_structured_outputs else "ON"
    )

    total_evals = (
        len(datasets_to_eval)
        * len(models_to_eval)
        * len(prompts_to_eval)
        * len(args.seeds)
    )
    current_eval = 0
    successful_evals = 0
    metrics_files = {}

    login(token=os.environ.get("HF_TOKEN"))

    for model_config in models_to_eval:
        logger.info("\n%s", "=" * 60)
        logger.info("🤖 Loading model: %s (%s)", model_config.name, model_config.id)
        logger.info("%s", "=" * 60)

        agent = None
        try:
            agent = create_agent_for_model(model_config)
            logger.info("✓ Model loaded successfully")

            for prompt_idx in prompts_to_eval:
                prompt = selected_prompts[prompt_idx]

                if prompt_idx not in metrics_files:
                    metrics = LLMPersistentMetrics.create_for_prompt(prompt)
                    metrics_files[prompt_idx] = metrics.filename
                else:
                    metrics = LLMPersistentMetrics(metrics_files[prompt_idx])

                for seed in args.seeds:
                    for dataset_name in datasets_to_eval:
                        current_eval += 1
                        dataset_data = datasets[dataset_name]

                        logger.info(
                            "\n  [%d/%d] Evaluating %s on prompt %d with %s set (seed=%d)...",
                            current_eval,
                            total_evals,
                            model_config.name,
                            prompt_idx,
                            dataset_name,
                            seed,
                        )
                        logger.info("      Prompt ID: %s", prompt.get_id())

                        try:
                            evaluate_model_on_prompt(
                                agent,
                                prompt,
                                dataset_data["texts"],
                                metrics,
                                dataset_name,
                                seed,
                                args.debug,
                                use_structured_outputs=(not args.no_structured_outputs),
                                structured_schema=schema_for_prompt(prompt),
                            )
                            logger.info("      ✓ Metrics updated and persisted")
                            successful_evals += 1

                        except Exception as e:
                            logger.error(
                                "Error evaluating %s on prompt %d with %s (seed=%d): %s",
                                model_config.name,
                                prompt_idx,
                                dataset_name,
                                seed,
                                e,
                            )
                            logger.debug("Traceback:", exc_info=True)

        except Exception as e:
            logger.error("Error loading model %s: %s", model_config.name, e)
            logger.debug("Traceback:", exc_info=True)
        finally:
            cleanup_agent(agent)
            cleanup_distributed_group()

    logger.info("\n%s", "=" * 60)
    logger.info(
        "✓ Evaluation complete! (%d/%d successful)", successful_evals, total_evals
    )
    logger.info("✓ Metrics files saved to:")
    for filepath in metrics_files.values():
        logger.info("  - %s", filepath)
    logger.info("%s", "=" * 60)

    for handler in logging.root.handlers:
        handler.flush()
    logging.shutdown()


if __name__ == "__main__":
    main()
