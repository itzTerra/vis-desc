import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sammo.base import LLMResult, Runner, EvaluationScore
from sammo.components import Output
from sammo.dataformatters import PlainFormatter
from sammo.instructions import MetaPrompt, Paragraph
from sammo.mutators import BagOfMutators, Paraphrase, Rewrite
from sammo.search import BeamSearch
from sammo.data import DataTable

from models.llm.agents import VLLMAgent, MODEL_BY_NAME
from models.llm.prompts import PROMPT_PARTS, GUIDELINE_CONFIGS, schema_for_suffix_key
from utils import DATA_DIR

load_dotenv()


class MistralRunner(Runner):
    """SAMMO Runner implementation using Mistral-Small3.2 via vLLM."""

    def __init__(self, vllm_agent: VLLMAgent):
        self.agent = vllm_agent
        super().__init__()

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        randomness: float = 0.0,
        seed: int = 0,
        **kwargs,
    ) -> LLMResult:
        """Generate text using the Mistral vLLM agent.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (not used, agent has default)
            randomness: Temperature for sampling (not used, agent config has default)
            seed: Random seed (not used in vLLM calls)
            **kwargs: Additional arguments

        Returns:
            LLMResult containing generated text response
        """
        system_prompt = kwargs.get("system_prompt")
        user_prompt = prompt

        if system_prompt is None and "\n\n" in prompt:
            system_prompt, user_prompt = prompt.split("\n\n", 1)

        response_text = self.agent.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            use_structured_outputs=True,
            structured_schema=schema_for_suffix_key("cot"),
        )

        return LLMResult(response_text)


def load_train_dataset(data_file: str | None = None) -> tuple[list[str], list[int]]:
    """Load train dataset with text segments and their ratings.

    Args:
        data_file: Optional path to custom parquet file

    Returns:
        Tuple of (texts, ratings)
    """
    if data_file:
        data_path = Path(data_file)
    else:
        data_path = DATA_DIR / "datasets" / "small" / "train.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_parquet(data_path)
    texts = df["text"].tolist()

    if "rating" in df.columns:
        ratings = df["rating"].tolist()
    elif "label" in df.columns:
        ratings = df["label"].tolist()
    else:
        raise ValueError("Dataset must have 'rating' or 'label' column")

    print(f"✓ Loaded {len(texts)} samples from train set")
    return texts, ratings


def accuracy_metric(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    """Calculate accuracy metric for rating predictions.

    Extracts integer ratings from predictions (looking for digits 0-5)
    and compares with ground truth.

    Args:
        y_true: DataTable with ground truth ratings in outputs
        y_pred: DataTable with predicted ratings in outputs

    Returns:
        EvaluationScore with accuracy value
    """
    y_true_values = y_true.outputs.values
    y_pred_values = y_pred.outputs.normalized_values()

    def _extract_rating(value) -> int | None:
        try:
            obj = json.loads(str(value))
            if isinstance(obj, dict):
                if "rating" in obj and isinstance(obj["rating"], int):
                    return obj["rating"]
        except Exception:
            pass
        for char in str(value):
            if char in "012345":
                return int(char)
        return None

    n_correct = 0
    for y_p, y_t in zip(y_pred_values, y_true_values):
        pred_rating = _extract_rating(y_p)
        try:
            if pred_rating is not None and pred_rating == int(y_t):
                n_correct += 1
        except (ValueError, TypeError):
            continue

    accuracy = n_correct / len(y_true_values) if y_true_values else 0.0
    return EvaluationScore(accuracy)


def mean_squared_error_metric(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    """Calculate mean squared error for rating predictions.

    Args:
        y_true: DataTable with ground truth ratings
        y_pred: DataTable with predicted ratings

    Returns:
        EvaluationScore with negative MSE (for maximization)
    """
    y_true_values = y_true.outputs.values
    y_pred_values = y_pred.outputs.normalized_values()

    def _extract_rating(value) -> int | None:
        try:
            obj = json.loads(str(value))
            if isinstance(obj, dict):
                if "rating" in obj and isinstance(obj["rating"], int):
                    return obj["rating"]
        except Exception:
            pass
        for char in str(value):
            if char in "012345":
                return int(char)
        return None

    errors = []
    for y_p, y_t in zip(y_pred_values, y_true_values):
        try:
            pred_rating = _extract_rating(y_p)
            if pred_rating is not None:
                error = pred_rating - int(y_t)
                errors.append(error * error)
            else:
                errors.append(25.0)
        except (ValueError, TypeError):
            errors.append(25.0)

    mse = sum(errors) / len(errors) if errors else 25.0
    return EvaluationScore(-mse)


class InitialPromptCandidates:
    """Callable to generate initial prompt candidates for SAMMO search."""

    def __init__(self, dtrain: DataTable, system_prompt: str):
        self.dtrain = dtrain
        self.system_prompt = system_prompt

    def __call__(self):
        """Build initial candidate prompts using guideline and suffix variations.

        Returns:
            Output component with MetaPrompt structure
        """
        example_formatter = PlainFormatter()

        examples = PROMPT_PARTS["examples"]["cot"]
        output_format = PROMPT_PARTS["output_format"]["cot"]
        input_part = PROMPT_PARTS["input"].replace("{{TEXT_SEGMENT}}", "{{input}}")

        instructions = MetaPrompt(
            [
                Paragraph(self.system_prompt, reference_id="system"),
                Paragraph("\n\n"),
                Paragraph(
                    GUIDELINE_CONFIGS["medium"]["text"], reference_id="guideline"
                ),
                Paragraph("\n\n"),
                Paragraph(examples, reference_id="examples"),
                Paragraph("\n\n"),
                Paragraph(output_format, reference_id="output_format"),
                Paragraph("\n\n"),
                Paragraph(input_part, reference_id="input"),
            ],
            render_as="raw",
            data_formatter=example_formatter,
        )

        return Output(
            instructions.with_extractor("raise"),
            minibatch_size=1,
            on_error="empty_result",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize prompts using SAMMO with Mistral-Small3.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to use from train set (default: all)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["accuracy", "mse"],
        default="mse",
        help="Metric to optimize (accuracy or mse) (default: mse)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )

    args = parser.parse_args()

    model_config = MODEL_BY_NAME.get("Mistral-Small3.2-24b")
    if not model_config:
        print("❌ Error: Mistral-Small3.2-24b not found in available models")
        sys.exit(1)

    vllm_agent = VLLMAgent(model_config=model_config)
    runner = MistralRunner(vllm_agent)

    texts, ratings = load_train_dataset()

    d_train = DataTable.from_records(
        [{"input": text, "output": str(rating)} for text, rating in zip(texts, ratings)]
    )

    if args.n_samples is not None and args.n_samples < len(d_train):
        d_train = d_train.sample(args.n_samples, seed=args.seed)
        print(f"✓ Using {len(d_train)} samples for optimization")

    system_prompt = PROMPT_PARTS["system"]

    mutation_operators = BagOfMutators(
        InitialPromptCandidates(d_train, system_prompt),
        Paraphrase("#system"),
        Paraphrase("#guideline"),
        Paraphrase("#output_format"),
        Rewrite(
            "#system",
            "Rewrite this system prompt to better steer the model toward accurate visual descriptiveness ratings while staying concise:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#guideline",
            "Rewrite this guideline to be clearer and more concise for rating visual descriptiveness:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#output_format",
            "Rewrite this prompt output format to be clearer and more concise:\n\n{{{{text}}}}",
        ),
        sample_for_init_candidates=False,
    )

    metric_fn = (
        accuracy_metric if args.metric == "accuracy" else mean_squared_error_metric
    )
    maximize = args.metric == "accuracy"

    prompt_optimizer = BeamSearch(
        runner,
        mutation_operators,
        metric_fn,
        maximize=maximize,
        mutations_per_beam=3,
        depth=3,
    )

    print("\n🚀 Running prompt optimization...")
    print("=" * 60)
    prompt_optimizer.fit(d_train)

    print("\n" + "=" * 60)
    print("📊 Optimization Results")
    print("=" * 60)
    prompt_optimizer.show_report()

    print("\n" + "=" * 60)
    print("🏆 Best Prompt")
    print("=" * 60)
    print(prompt_optimizer.best_prompt)

    output_path = DATA_DIR / "output" / "optimized_prompt.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(str(prompt_optimizer.best_prompt))
    print(f"\n✓ Best prompt saved to: {output_path}")


if __name__ == "__main__":
    main()
