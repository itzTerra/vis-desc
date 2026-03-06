import argparse
import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
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
from sammo.mutators import BagOfMutators, Rewrite
from sammo.search import BeamSearch

from models.llm.agents import VLLMAgent, MODEL_BY_NAME
from models.llm.prompts import PROMPT_PARTS, schema_for_suffix_key
from models.llm.run import parse_output
from utils import DATA_DIR
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

DEBUG_MODE = False
DEBUG_LOG_PATH: Path | None = None
DEBUG_LOG_HANDLER: logging.FileHandler | None = None

BATCH_SIZE = 16
STRUCTURED_OUTPUTS_ENABLED = True
STRUCTURED_SCHEMA_SUFFIX = "base"
INITIAL_PROMPT_EXAMPLES_KEY = "base"
INITIAL_PROMPT_OUTPUT_FORMAT_KEY = "base"
INITIAL_PROMPT_TASK_DESCRIPTION_KEY = "small"


@dataclass
class _BatchRequest:
    prompt: str
    system_prompt: str | None
    use_structured_outputs: bool
    structured_schema: dict | None
    seed: int | None
    randomness: float | None
    future: asyncio.Future
    schema_key: str


class RatingExtractor(Extractor):
    """Extract parsed rating from model output."""

    def _extract_from_single_value(self, text: str):
        rating, _ = parse_output(str(text))
        if DEBUG_MODE:
            logger.debug(f"Parsed score: {rating!r}")
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
        batch_size: int = BATCH_SIZE,
        max_batch_delay: float = 0.05,
        use_structured_outputs: bool = STRUCTURED_OUTPUTS_ENABLED,
        structured_schema_suffix: str = STRUCTURED_SCHEMA_SUFFIX,
        seed: int = 0,
    ):
        super().__init__()
        self.agent = vllm_agent
        self.batch_size = batch_size
        self.max_batch_delay = max_batch_delay
        self.use_structured_outputs = use_structured_outputs
        self.structured_schema = schema_for_suffix_key(structured_schema_suffix)
        self.seed = seed
        self._queue: list[_BatchRequest] = []
        self._queue_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        randomness: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> LLMResult:
        seed = self.seed
        is_mutation_request = any(
            keyword in prompt.lower() for keyword in ["rewrite", "paraphrase"]
        )

        user_prompt = prompt

        system_prompt = kwargs.get("system_prompt")
        if not is_mutation_request:
            if system_prompt is None and "\n\n" in prompt:
                system_prompt, user_prompt = prompt.split("\n\n", 1)

        use_structured = self.use_structured_outputs and not is_mutation_request
        schema = self.structured_schema if use_structured else None

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        request = _BatchRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            use_structured_outputs=use_structured,
            structured_schema=schema,
            seed=seed,
            randomness=randomness,
            future=future,
            schema_key=json.dumps(schema, sort_keys=True) if schema else "",
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
            randomness = requests[0].randomness if requests else None
            schema = requests[0].structured_schema if requests else None

            try:
                responses = await asyncio.to_thread(
                    self.agent.generate_batch,
                    prompts,
                    system_prompt=system_prompt,
                    use_structured_outputs=use_structured_outputs,
                    structured_schema=schema,
                    seed=seed,
                    temperature=randomness,
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


def _rewrite(part_id: str, objective: str) -> Rewrite:
    """Build a Rewrite mutator for the given prompt part with the given objective."""
    part_type = part_id.lstrip("#")
    prompt = (
        "You are a prompt optimization expert. Your task is to rewrite a prompt part.\n\n"
        '<prompt_part type="' + part_type + '">\n'
        "{{{{text}}}}\n"
        "</prompt_part>\n\n"
        "Rewrite objective: " + objective + "\n\n"
        "Return ONLY the rewritten prompt part, without any explanation, preamble, or commentary. "
        "Do not include XML tags in your response."
    )
    return Rewrite(part_id, prompt)


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
    # parser.add_argument(
    #     "--metric",
    #     type=str,
    #     choices=["accuracy", "mse", "accuracy_rmse"],
    #     default="accuracy_rmse",
    #     help="Metric to optimize (accuracy, mse, or accuracy_rmse) (default: accuracy_rmse)",
    # )
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
            logging.Formatter("%(asctime)s[%(levelname)s]: %(message)s")
        )
        DEBUG_LOG_HANDLER = file_handler

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s[%(levelname)s]: %(message)s")
        )

        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        logger.info(f"Debug logging enabled. Logs will be written to: {DEBUG_LOG_PATH}")
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s[%(levelname)s]: %(message)s")
        )
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    model_config = MODEL_BY_NAME.get(args.model)
    if not model_config:
        print(f"❌ Error: {args.model} not found in available models")
        sys.exit(1)

    vllm_agent = VLLMAgent(
        model_config=model_config, logger=logger if DEBUG_MODE else None
    )
    runner = VLLMBatchedRunner(vllm_agent, seed=args.seed)

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
        # Paraphrase("#system"),
        # Paraphrase("#guideline"),
        # Paraphrase("#examples"),
        # Paraphrase("#output_format"),
        # Paraphrase("#input"),
        # --- #system ---
        _rewrite(
            "#system",
            "Trim to the essential role statement for visual descriptiveness rating, removing any redundant phrasing.",
        ),
        _rewrite(
            "#system",
            "Reframe the model's role entirely—treat it as a visual perception analyst who measures how vividly text activates the mind's eye.",
        ),
        _rewrite(
            "#system",
            "Expand with explicit guidance on what dimensions to analyze: concrete details, sensory language, spatial descriptions, and imagery density.",
        ),
        # --- #guideline ---
        _rewrite(
            "#guideline",
            "Condense the rating criteria to the essential decision points only, cutting all redundancy.",
        ),
        _rewrite(
            "#guideline",
            "Reframe the rating task as a painter's assessment: how much raw visual material does this text give an artist to work from?",
        ),
        _rewrite(
            "#guideline",
            "Explicitly define each rating level (1–5) with boundary conditions and contrasting examples for low vs. high visual descriptiveness.",
        ),
        _rewrite(
            "#guideline",
            "Focus on specific linguistic markers of visual descriptiveness: descriptive adjectives, concrete nouns, sensory verbs, and spatial prepositions.",
        ),
        _rewrite(
            "#guideline",
            "Address edge cases—abstract language, figurative language, technical jargon, and minimalist prose—and how to rate each consistently.",
        ),
        # --- #examples ---
        _rewrite(
            "#examples",
            "Trim to the fewest examples needed to anchor the rating scale, keeping only the most representative cases.",
        ),
        _rewrite(
            "#examples",
            "Replace with contrastive pairs: the same scene described with minimal vs. rich visual language, side by side, highlighting what shifts the rating.",
        ),
        _rewrite(
            "#examples",
            "Expand to cover the full 1–5 spectrum with at least one example per level, annotating the specific features that justify each rating.",
        ),
        # --- #output_format ---
        _rewrite(
            "#output_format",
            "Reduce to the minimal format specification: just the required field(s) and their types.",
        ),
        _rewrite(
            "#output_format",
            "Restructure to require a brief chain-of-thought observation about the text's visual language before committing to a final rating.",
        ),
        _rewrite(
            "#output_format",
            "Add a confidence field and a required one-sentence justification alongside the numeric rating.",
        ),
        # --- #input ---
        _rewrite(
            "#input",
            "Shorten to a single directive line that clearly signals the text to rate follows.",
        ),
        _rewrite(
            "#input",
            "Reframe the prompt transition as an invitation to examine the passage through a visual lens, priming the model for perceptual analysis.",
        ),
        _rewrite(
            "#input",
            "Add explicit pre-rating instructions: read once for overall impression, then scan for specific visual markers before assigning the rating.",
        ),
        # --- generic cross-section rewrites ---
        _rewrite(
            "#system",
            "Adopt a more authoritative, instructional tone throughout—use imperative voice and eliminate hedging language.",
        ),
        _rewrite(
            "#system",
            "Switch to a neutral, analytical persona that avoids metaphor and focuses purely on operational criteria.",
        ),
        _rewrite(
            "#guideline",
            "Restructure as a numbered decision tree: evaluate each criterion in sequence and aggregate to a final score.",
        ),
        _rewrite(
            "#guideline",
            "Rewrite using 'if … then …' conditional rules for each rating level to make the decision boundary explicit.",
        ),
        _rewrite(
            "#guideline",
            "Simplify to a single scoring rubric table with columns: rating, defining feature, disqualifying feature.",
        ),
        _rewrite(
            "#examples",
            "Reorder examples from least to most visually descriptive to make the rating scale progression explicit.",
        ),
        _rewrite(
            "#examples",
            "Annotate each example with a one-sentence explanation of which specific textual feature drives its rating.",
        ),
        _rewrite(
            "#output_format",
            "Require the model to list up to three textual evidence quotes before the final rating.",
        ),
        _rewrite(
            "#output_format",
            "Strip all optional fields and enforce a single integer output with no surrounding text.",
        ),
        _rewrite(
            "#input",
            "Add a reminder immediately before the text that the model should attend only to visual and sensory language, not to narrative or emotional content.",
        ),
        sample_for_init_candidates=False,
    )

    metric_fn = mean_squared_error_metric
    maximize = False

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / f"optimization_candidates_{run_timestamp}.json"

    class DebugBeamSearch(BeamSearch):
        """Beam search with debug logging."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._iteration_count = 0
            self._candidate_count = 0
            self._all_candidates: list[dict] = []

        def _flush_candidates(self) -> None:
            sorted_candidates = sorted(
                self._all_candidates,
                key=lambda c: c["objective"],
                reverse=maximize,
            )
            with open(candidates_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": {
                            "model": args.model,
                            "seed": args.seed,
                            "beam_width": args.beam_width,
                            "depth": args.depth,
                            "mutations_per_beam": args.mutations_per_beam,
                            "maximize": maximize,
                            "total_candidates": len(sorted_candidates),
                        },
                        "candidates": sorted_candidates,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

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
                    f"[ITERATION {self._iteration_count}] {len(candidates)} candidate(s)"
                )
                logger.info("=" * 80)

                for i, candidate in enumerate(candidates):
                    self._candidate_count += 1
                    logger.info(
                        f"\n\t[{self._iteration_count}.{i}] Candidate {self._candidate_count}:"
                    )
                    logger.info(f"```\n{str(candidate)}\n```")

            results = await super().evaluate(
                candidates, runner, objective, dataset, colbar
            )

            for result in results:
                raw_predictions = result["predictions"].outputs.normalized_values()
                parsed_scores = [parse_output(str(p))[0] for p in raw_predictions]
                self._all_candidates.append(
                    {
                        "iteration": self._iteration_count,
                        "objective": result["objective"],
                        "parse_errors": result["parse_errors"],
                        "scores": parsed_scores,
                        "prompt": str(result["candidate"]),
                    }
                )

            self._iteration_count += 1
            self._flush_candidates()
            return results

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
    report_buffer = StringIO()
    with redirect_stdout(report_buffer):
        prompt_optimizer.show_report()
    report_text = report_buffer.getvalue().rstrip()
    if report_text:
        print(report_text)

    print("\n" + "=" * 60)
    print("🏆 Best Prompt")
    print("=" * 60)
    print(prompt_optimizer.best_prompt)

    output_with_results_path = output_dir / f"optimization_results_{run_timestamp}.txt"

    with open(output_with_results_path, "w", encoding="utf-8") as f:
        # Write run metadata
        f.write("Run Metadata\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Beam width: {args.beam_width}\n")
        f.write(f"Depth: {args.depth}\n")
        f.write(f"Mutations per beam: {args.mutations_per_beam}\n")
        f.write("\n")
        # Write optimization results
        f.write("Optimization Results\n")
        f.write("=" * 60 + "\n")
        if report_text:
            f.write(report_text + "\n\n")
        else:
            f.write("(No report output available)\n\n")
        # Write best prompt
        f.write("Best Prompt\n")
        f.write("=" * 60 + "\n")
        f.write(str(prompt_optimizer.best_prompt))
    print(f"✓ Results + prompt saved to: {output_with_results_path}")
    print(
        f"✓ All {len(prompt_optimizer._all_candidates)} candidate prompts saved to: {candidates_path}"
    )

    if DEBUG_MODE and DEBUG_LOG_PATH:
        print(f"✓ Debug logs saved to: {DEBUG_LOG_PATH}")

    for handler in logging.root.handlers:
        handler.flush()
    logging.shutdown()


if __name__ == "__main__":
    main()
