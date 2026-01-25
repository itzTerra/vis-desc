import argparse
import asyncio
import json
import math
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sammo.base import LLMResult, Runner, EvaluationScore
from sammo.components import Output
from sammo.data import DataTable
from sammo.dataformatters import DataFormatter
from sammo.extractors import Extractor, StripWhitespace
from sammo.instructions import InputData, MetaPrompt, Paragraph
from sammo.mutators import BagOfMutators, Paraphrase, Rewrite
from sammo.search import BeamSearch

from models.llm.agents import VLLMAgent, MODEL_BY_NAME
from models.llm.prompts import PROMPT_PARTS, schema_for_suffix_key
from models.llm.run import parse_output
from utils import DATA_DIR

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

DEBUG_MODE = False
DEBUG_LOG_PATH: Path | None = None
DEBUG_LOG_HANDLER: logging.FileHandler | None = None

STRUCTURED_OUTPUTS_ENABLED = True
STRUCTURED_SCHEMA_SUFFIX = "base"
INITIAL_PROMPT_EXAMPLES_KEY = "base"
INITIAL_PROMPT_OUTPUT_FORMAT_KEY = "base"
INITIAL_PROMPT_TASK_DESCRIPTION_KEY = "full"


@dataclass
class _BatchRequest:
    prompt: str
    system_prompt: str | None
    use_structured_outputs: bool
    structured_schema: dict | None
    seed: int | None
    future: asyncio.Future
    schema_key: str


class RatingExtractor(Extractor):
    """Extract parsed rating from model output."""

    def _extract_from_single_value(self, text: str):
        rating, _ = parse_output(str(text))
        return [rating] if rating is not None else [None]


class TextSegmentFormatter(DataFormatter):
    """Format inputs and parse model outputs for prompt optimization."""

    def __init__(self):
        super().__init__(flatten_1d_dicts=True, include_ids=False, orient="item")

    def _dump(self, records):
        grouped = records.group_by(self._orient)
        segments = []
        for _, items in grouped:
            segments.extend(
                f"<text_segment>{item['value']}</text_segment>" for item in items
            )
        return "\n".join(segments)

    def get_extractor(self, child, on_error="raise"):
        return RatingExtractor(
            StripWhitespace(child),
            on_error=on_error,
        )


class VLLMBatchedRunner(Runner):
    """SAMMO Runner that micro-batches requests before hitting vLLM."""

    def __init__(
        self,
        vllm_agent: VLLMAgent,
        batch_size: int = 32,
        max_batch_delay: float = 0.05,
        use_structured_outputs: bool = STRUCTURED_OUTPUTS_ENABLED,
        structured_schema_suffix: str = STRUCTURED_SCHEMA_SUFFIX,
    ):
        super().__init__()
        self.agent = vllm_agent
        self.batch_size = batch_size
        self.max_batch_delay = max_batch_delay
        self.use_structured_outputs = use_structured_outputs
        self.structured_schema = schema_for_suffix_key(structured_schema_suffix)
        self._queue: list[_BatchRequest] = []
        self._queue_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        randomness: float = 0.0,
        seed: int = 0,
        **kwargs,
    ) -> LLMResult:
        system_prompt = kwargs.get("system_prompt")
        user_prompt = prompt

        if system_prompt is None and "\n\n" in prompt:
            system_prompt, user_prompt = prompt.split("\n\n", 1)

        # if DEBUG_MODE and DEBUG_LOG_PATH:
        #     logger.debug(
        #         f"[Model Input]\nSystem prompt: {system_prompt}\nUser prompt: {user_prompt}"
        #     )

        seed = kwargs.get("seed", seed)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        request = _BatchRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            use_structured_outputs=self.use_structured_outputs,
            structured_schema=self.structured_schema,
            seed=seed,
            future=future,
            schema_key=json.dumps(self.structured_schema, sort_keys=True)
            if self.structured_schema
            else "",
        )

        async with self._queue_lock:
            self._queue.append(request)
            if len(self._queue) >= self.batch_size:
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                await self._flush_locked()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._delayed_flush())

        return await future

    async def _delayed_flush(self) -> None:
        try:
            await asyncio.sleep(self.max_batch_delay)
            async with self._queue_lock:
                if self._queue:
                    await self._flush_locked()
        finally:
            self._flush_task = None

    async def _flush_locked(self) -> None:
        if not self._queue:
            return

        pending = self._queue
        self._queue = []

        grouped: dict[tuple, list[_BatchRequest]] = {}
        for req in pending:
            key = (
                req.system_prompt,
                req.use_structured_outputs,
                req.schema_key,
                req.seed,
            )
            grouped.setdefault(key, []).append(req)

        for (
            system_prompt,
            use_structured_outputs,
            _,
            seed,
        ), requests in grouped.items():
            prompts = [r.prompt for r in requests]
            schema = requests[0].structured_schema if requests else None

            # if DEBUG_MODE and DEBUG_LOG_PATH:
            #     for idx, (sys_p, user_p) in enumerate(
            #         zip([system_prompt] * len(prompts), prompts)
            #     ):
            #         logger.debug(
            #             f"[Batch Input {idx}]\nSystem prompt: {sys_p}\nUser prompt: {user_p}"
            #         )

            try:
                responses = await asyncio.to_thread(
                    self.agent.generate_batch,
                    prompts,
                    system_prompt=system_prompt,
                    use_structured_outputs=use_structured_outputs,
                    structured_schema=schema,
                    seed=seed,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                for req in requests:
                    if not req.future.done():
                        req.future.set_exception(exc)
                continue

            for response_text, req in zip(responses, requests):
                if not req.future.done():
                    req.future.set_result(LLMResult(response_text))


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

    Extracts integer ratings from predictions using parse_output
    and compares with ground truth.

    Args:
        y_true: DataTable with ground truth ratings in outputs
        y_pred: DataTable with predicted ratings in outputs

    Returns:
        EvaluationScore with accuracy value
    """
    y_true_values = y_true.outputs.values
    y_pred_values = y_pred.outputs.normalized_values()

    n_correct = 0
    for i, (y_p, y_t) in enumerate(zip(y_pred_values, y_true_values)):
        parsed_rating, _ = parse_output(str(y_p))
        if DEBUG_MODE and DEBUG_LOG_PATH:
            logger.debug(
                f"[Sample {i}] Raw output: {y_p!r} | Parsed rating: {parsed_rating} | Ground truth: {y_t}"
            )
        try:
            if parsed_rating is not None and int(parsed_rating) == int(y_t):
                n_correct += 1
        except (ValueError, TypeError):
            continue

    accuracy = n_correct / len(y_true_values) if y_true_values else 0.0
    return EvaluationScore(accuracy)


def mean_squared_error_metric(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    """Calculate mean squared error for rating predictions.

    Uses parse_output to extract ratings from predictions.

    Args:
        y_true: DataTable with ground truth ratings
        y_pred: DataTable with predicted ratings

    Returns:
        EvaluationScore with MSE
    """
    y_true_values = y_true.outputs.values
    y_pred_values = y_pred.outputs.normalized_values()

    errors = []
    for i, (y_p, y_t) in enumerate(zip(y_pred_values, y_true_values)):
        try:
            parsed_rating, _ = parse_output(str(y_p))
            if parsed_rating is not None:
                error = int(parsed_rating) - int(y_t)
                errors.append(error * error)
                if DEBUG_MODE and DEBUG_LOG_PATH:
                    logger.debug(
                        f"[Sample {i}] Raw output: {y_p!r} | Parsed rating: {parsed_rating} | Ground truth: {y_t} | Error: {error}"
                    )
            else:
                logger.debug(
                    f"Failed to parse rating from prediction {i}: {y_p!r} (true: {y_t})"
                )
                errors.append(25.0)
        except (ValueError, TypeError) as e:
            logger.debug(f"Error parsing prediction {i}: {y_p!r} (true: {y_t}) - {e}")
            errors.append(25.0)

    mse = sum(errors) / len(errors) if errors else 25.0
    return EvaluationScore(mse)


def accuracy_rmse_metric(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    """Blend accuracy and RMSE into a single score (higher is better)."""

    y_true_values = y_true.outputs.values
    y_pred_values = y_pred.outputs.normalized_values()

    n_correct = 0
    squared_errors: list[float] = []

    for i, (y_p, y_t) in enumerate(zip(y_pred_values, y_true_values)):
        try:
            true_rating = int(y_t)
        except (ValueError, TypeError) as exc:
            logger.debug(f"Error parsing ground truth for sample {i}: {y_t!r} - {exc}")
            squared_errors.append(25.0)
            continue

        parsed_rating, _ = parse_output(str(y_p))
        # if DEBUG_MODE and DEBUG_LOG_PATH:
        #     logger.debug(
        #         f"[Sample {i}] Raw output: {y_p!r} | Parsed rating: {parsed_rating} | Ground truth: {true_rating}"
        #     )

        if parsed_rating is None:
            logger.debug(
                f"Failed to parse rating from prediction {i}: {y_p!r} (true: {true_rating})"
            )
            squared_errors.append(25.0)
            continue

        try:
            pred_rating = int(parsed_rating)
        except (ValueError, TypeError) as exc:
            logger.debug(
                f"Error casting prediction {i}: {parsed_rating!r} (raw: {y_p!r}) - {exc}"
            )
            squared_errors.append(25.0)
            continue

        if pred_rating == true_rating:
            n_correct += 1

        error = pred_rating - true_rating
        squared_errors.append(error * error)

    sample_count = len(y_true_values)
    mse = sum(squared_errors) / sample_count if sample_count else 25.0
    rmse = math.sqrt(mse)

    rating_range = 5.0

    normalized_rmse = rmse / rating_range
    normalized_rmse = max(0.0, min(1.0, normalized_rmse))

    accuracy = n_correct / sample_count if sample_count else 0.0
    score = 0.5 * accuracy + 1 * (1 - normalized_rmse)

    return EvaluationScore(score)


class InitialPromptCandidates:
    """Callable to generate initial prompt candidates for SAMMO search."""

    def __init__(self, dtrain: DataTable, system_prompt: str):
        self.dtrain = dtrain
        self.system_prompt = system_prompt

    def __call__(self):
        """Build initial candidate prompts using guideline and suffix variations.

        Returns:
            MetaPrompt structure to be wrapped in Output by BagOfMutators
        """
        task_description = PROMPT_PARTS["task_descriptions"][
            INITIAL_PROMPT_TASK_DESCRIPTION_KEY
        ]
        examples = PROMPT_PARTS["examples"][INITIAL_PROMPT_EXAMPLES_KEY]
        output_format = PROMPT_PARTS["output_format"][INITIAL_PROMPT_OUTPUT_FORMAT_KEY]

        formatter = TextSegmentFormatter()

        instructions = MetaPrompt(
            [
                Paragraph(self.system_prompt, reference_id="system"),
                Paragraph("\n\n"),
                Paragraph(task_description, reference_id="guideline"),
                Paragraph("\n\n"),
                Paragraph(examples, reference_id="examples"),
                Paragraph("\n\n"),
                Paragraph(output_format, reference_id="output_format"),
                Paragraph("\n\n"),
                Paragraph(
                    "## Input\nRate the following text segment:\n\n",
                    reference_id="input",
                ),
                Paragraph(InputData()),
            ],
            render_as="raw",
            data_formatter=formatter,
        )

        return Output(
            instructions.with_extractor("raise"),
            minibatch_size=1,
            on_error="raise",
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
        choices=["accuracy", "mse", "accuracy_rmse"],
        default="accuracy_rmse",
        help="Metric to optimize (accuracy, mse, or accuracy_rmse) (default: accuracy_rmse)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Llama4-Scout-17b",
        help="Model name from MODEL_BY_NAME (default: Llama4-Scout-17b)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging of raw model inputs, outputs, and parsed values",
    )
    parser.add_argument(
        "--mutations-per-beam",
        type=int,
        default=6,
        help="Number of mutations per beam in the search (default: 6)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Search depth for beam search optimization (default: 4)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for beam search optimization (default: 4)",
    )

    args = parser.parse_args()

    global DEBUG_MODE, DEBUG_LOG_PATH, DEBUG_LOG_HANDLER
    DEBUG_MODE = args.debug

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if DEBUG_MODE:
        debug_log_dir = DATA_DIR / "output" / "debug_logs"
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        DEBUG_LOG_PATH = debug_log_dir / "optimization_debug.log"

        file_handler = logging.FileHandler(DEBUG_LOG_PATH, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        DEBUG_LOG_HANDLER = file_handler

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        logger.info(f"Debug logging enabled. Logs will be written to: {DEBUG_LOG_PATH}")
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    model_config = MODEL_BY_NAME.get(args.model)
    if not model_config:
        print(f"❌ Error: {args.model} not found in available models")
        sys.exit(1)

    vllm_agent = VLLMAgent(model_config=model_config)
    runner = VLLMBatchedRunner(vllm_agent)

    texts, ratings = load_train_dataset()

    if args.n_samples is not None and args.n_samples < len(texts):
        df = pd.DataFrame({"text": texts, "rating": ratings})
        df_sampled = df.sample(
            n=args.n_samples, stratify=df["rating"], random_state=args.seed
        )
        texts = df_sampled["text"].tolist()
        ratings = df_sampled["rating"].tolist()
        print(f"✓ Using {len(texts)} samples for optimization (stratified by rating)")

    d_train = DataTable.from_records(
        [{"input": text, "output": str(rating)} for text, rating in zip(texts, ratings)]
    )

    system_prompt = PROMPT_PARTS["system"]

    mutation_operators = BagOfMutators(
        InitialPromptCandidates(d_train, system_prompt),
        Paraphrase("#system"),
        Paraphrase("#guideline"),
        Paraphrase("#examples"),
        Paraphrase("#output_format"),
        Paraphrase("#input"),
        Rewrite(
            "#system",
            "Rewrite this system prompt to better steer the model toward accurate visual descriptiveness ratings while staying concise:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#guideline",
            "Rewrite this guideline to be clearer and more concise for rating visual descriptiveness:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#examples",
            "Rewrite these examples to be clearer and more concise while illustrating rating visual descriptiveness:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#output_format",
            "Rewrite this prompt output format to be clearer and more concise:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#guideline",
            "Rewrite this guideline to explicitly describe the rating scale boundaries (1-5) and clarify what makes text low vs high visual descriptiveness:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#examples",
            "Expand these examples to show more diverse cases covering the full rating scale spectrum:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#system",
            "Rewrite this system prompt to emphasize careful analysis of concrete details, sensory language, and imagery:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#guideline",
            "Rewrite this guideline to focus on specific linguistic markers that indicate visual descriptiveness (adjectives, concrete nouns, sensory verbs):\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#examples",
            "Rewrite these examples to highlight the specific features that justify each rating decision:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#output_format",
            "Rewrite this output format to encourage confidence scoring:\n\n{{{{text}}}}",
        ),
        Rewrite(
            "#guideline",
            "Rewrite this guideline to address edge cases like abstract vs concrete language, figurative language, and technical descriptions:\n\n{{{{text}}}}",
        ),
        sample_for_init_candidates=False,
    )

    if args.metric == "accuracy":
        metric_fn = accuracy_metric
        maximize = True
    elif args.metric == "accuracy_rmse":
        metric_fn = accuracy_rmse_metric
        maximize = True
    else:
        metric_fn = mean_squared_error_metric
        maximize = False

    class DebugBeamSearch(BeamSearch):
        """Beam search with debug logging."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._iteration_count = 0
            self._candidate_count = 0

        async def evaluate(
            self,
            candidates: list[Output],
            runner,
            objective,
            dataset: DataTable,
            colbar=None,
        ) -> list[dict]:
            if DEBUG_MODE and DEBUG_LOG_PATH:
                logger.info("\n" + "=" * 80)
                logger.info(
                    f"[ITERATION {self._iteration_count}] Evaluating {len(candidates)} candidate(s)"
                )
                logger.info("=" * 80)

                for i, candidate in enumerate(candidates):
                    self._candidate_count += 1
                    logger.info(
                        f"\n--- Candidate {self._candidate_count} (Iteration {self._iteration_count}, Index {i}) ---"
                    )
                    logger.info(f"Prompt preview:\n{str(candidate)}")

            return await super().evaluate(
                candidates, runner, objective, dataset, colbar
            )

    prompt_optimizer = DebugBeamSearch(
        runner,
        mutation_operators,
        metric_fn,
        maximize=maximize,
        mutations_per_beam=args.mutations_per_beam,
        depth=args.depth,
        beam_width=args.beam_width,
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

    if DEBUG_MODE and DEBUG_LOG_PATH:
        print(f"✓ Debug logs saved to: {DEBUG_LOG_PATH}")

    for handler in logging.root.handlers:
        handler.flush()
    logging.shutdown()


if __name__ == "__main__":
    main()
