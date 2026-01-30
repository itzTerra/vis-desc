import argparse
import sys
import time
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from models.nli.classifiers import (
    NLIRoberta,
    NLIDebertaLarge,
    NLIModernBERTLarge,
    ORTNLIZeroshotClassifier,
)
from models.nli.common import (
    METRICS_DIR,
    NLIConfig,
    get_corrs,
    probs_to_score,
)
from utils import DATA_DIR, PersistentDict, get_device_name


BATCH_SIZE = 16

AVAILABLE_MODELS = {
    "roberta": NLIRoberta,
    "deberta_large": NLIDebertaLarge,
    "modernbert_large": NLIModernBERTLarge,
}

BINARY_CONFIGS = [
    # Original configs - general visual/detailed/descriptive
    NLIConfig(
        candidate_labels=["not detailed", "detailed"],
        hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
    ),
    NLIConfig(
        candidate_labels=["not descriptive", "descriptive"],
        hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
    ),
    NLIConfig(
        candidate_labels=["not visual", "visual"],
        hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
    ),
    NLIConfig(
        candidate_labels=["not detailed", "detailed"],
        hypothesis_template="This text is {} in terms of sensory details, imagery, and vivid descriptions.",
    ),
    NLIConfig(
        candidate_labels=["not descriptive", "descriptive"],
        hypothesis_template="This text is {} in terms of sensory details, imagery, and vivid descriptions.",
    ),
    NLIConfig(
        candidate_labels=["not visual", "visual"],
        hypothesis_template="This text is {} in terms of sensory details, imagery, and vivid descriptions.",
    ),
    NLIConfig(
        candidate_labels=["not visual", "visual"],
        hypothesis_template="This text is {} in terms of sensory details, imagery, characters, environment, and vivid descriptions.",
    ),
    NLIConfig(
        candidate_labels=["not visual", "visual"],
        hypothesis_template="This text is {} in terms of sensory details, imagery, characters, and vivid descriptions of foreground and background.",
    ),
    NLIConfig(
        candidate_labels=[
            "is difficult to visualize or abstract",
            "can be easily visualized with specific sensory details",
        ],
        hypothesis_template="This text {}",
    ),
    NLIConfig(
        candidate_labels=[
            "contains minimal or no visual description",
            "contains vivid visual description with multiple specific details",
        ],
        hypothesis_template="This text {}",
    ),
    NLIConfig(
        candidate_labels=[
            "describes abstract concepts without visual imagery",
            "describes a visually complete scene with setting and details",
        ],
        hypothesis_template="This text {}",
    ),
    NLIConfig(
        candidate_labels=[
            "provides little visual description",
            "provides rich visual description of people, places, or objects",
        ],
        hypothesis_template="This text {}",
    ),
    # New configs - abstract vs concrete
    NLIConfig(
        candidate_labels=["abstract", "concrete"],
        hypothesis_template="This text describes {} concepts.",
    ),
    NLIConfig(
        candidate_labels=["abstract and intangible", "concrete and visualizable"],
        hypothesis_template="The content in this text is {}.",
    ),
    # New configs - objects and characters
    NLIConfig(
        candidate_labels=[
            "lacks specific objects or characters",
            "contains specific objects or characters",
        ],
        hypothesis_template="This text {}.",
    ),
    NLIConfig(
        candidate_labels=["no visible elements", "visible objects, people, or places"],
        hypothesis_template="This text describes {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "does not describe appearance",
            "describes physical appearance",
        ],
        hypothesis_template="This text {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "without visual attributes",
            "with visual attributes like color, shape, or size",
        ],
        hypothesis_template="This text describes things {}.",
    ),
    # New configs - scene completeness
    NLIConfig(
        candidate_labels=["incomplete or fragmented", "a complete visual scene"],
        hypothesis_template="This text presents {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "lacks setting or environment",
            "includes setting and environment",
        ],
        hypothesis_template="This text {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "no spatial context",
            "clear spatial context and arrangement",
        ],
        hypothesis_template="This text provides {}.",
    ),
    NLIConfig(
        candidate_labels=["isolated elements", "elements within a scene"],
        hypothesis_template="This text describes {}.",
    ),
    # New configs - detail and vividness
    NLIConfig(
        candidate_labels=["generic and vague", "specific and detailed"],
        hypothesis_template="The visual descriptions in this text are {}.",
    ),
    NLIConfig(
        candidate_labels=["lacks sensory details", "rich in sensory details"],
        hypothesis_template="This text is {}.",
    ),
    NLIConfig(
        candidate_labels=["plain description", "vivid and evocative description"],
        hypothesis_template="This text uses {}.",
    ),
    NLIConfig(
        candidate_labels=["minimal detail", "extensive visual detail"],
        hypothesis_template="This text provides {}.",
    ),
    # New configs - atmospheric and environmental
    NLIConfig(
        candidate_labels=[
            "no atmosphere or mood",
            "conveys atmosphere through visual details",
        ],
        hypothesis_template="This text {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "lacks environmental context",
            "establishes environment with visual elements",
        ],
        hypothesis_template="This text {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "without lighting or time indicators",
            "includes lighting, time of day, or weather",
        ],
        hypothesis_template="This text is {}.",
    ),
    # New configs - foreground and background
    NLIConfig(
        candidate_labels=[
            "no depth or layers",
            "distinguishes foreground and background",
        ],
        hypothesis_template="This text {}.",
    ),
    NLIConfig(
        candidate_labels=["flat description", "layered scene with depth"],
        hypothesis_template="This text presents a {}.",
    ),
    # New configs - action and movement (visual actions from annotation guidelines)
    NLIConfig(
        candidate_labels=["static concepts", "observable actions or movements"],
        hypothesis_template="This text describes {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "does not depict visual actions",
            "depicts characters performing visual actions",
        ],
        hypothesis_template="This text {}.",
    ),
    # New configs - imageability focus
    NLIConfig(
        candidate_labels=["hard to form a mental image", "easy to form a mental image"],
        hypothesis_template="From this text, it is {}.",
    ),
    NLIConfig(
        candidate_labels=["low imageability", "high imageability"],
        hypothesis_template="This text has {}.",
    ),
    NLIConfig(
        candidate_labels=[
            "cannot be easily pictured",
            "can be easily pictured in mind",
        ],
        hypothesis_template="The content of this text {}.",
    ),
    # New configs - character vs scene focus
    NLIConfig(
        candidate_labels=[
            "no character description",
            "character appearance or clothing",
        ],
        hypothesis_template="This text includes {}.",
    ),
    NLIConfig(
        candidate_labels=["no scene description", "scene with location and objects"],
        hypothesis_template="This text includes {}.",
    ),
]

TERTIARY_CONFIGS = [
    NLIConfig(
        candidate_labels=["disagree", "agree", "strongly agree"],
        hypothesis_template="I {} that the text is visual in terms of sensory details, imagery, and vivid descriptions.",
    ),
    NLIConfig(
        candidate_labels=["low", "medium", "high"],
        hypothesis_template="The visual descriptiveness of this text is {}.",
    ),
    NLIConfig(
        candidate_labels=["none", "some", "rich"],
        hypothesis_template="The imagery in this text is {}.",
    ),
    NLIConfig(
        candidate_labels=["minimal", "moderate", "extensive"],
        hypothesis_template="The visual detail level of this text is: {}",
    ),
]

HEXA_CONFIGS = [
    NLIConfig(
        candidate_labels=[
            "no visual imagery",
            "simple object or person",
            "object or person in action",
            "detailed place or character",
            "complete scene with action",
            "fully immersive visual scene",
        ],
        hypothesis_template="This passage depicts {}",
    ),
    NLIConfig(
        candidate_labels=[
            "no visual description",
            "basic visible object or person",
            "object or person doing something",
            "detailed location or recognizable character",
            "rich visual scene with character, setting, and details",
            "vivid, complete scene with atmosphere and lighting",
        ],
        hypothesis_template="The visual richness of this text shows {}",
    ),
    NLIConfig(
        candidate_labels=[
            "zero visual imagery",
            "minimal visual content",
            "moderate visual scene",
            "good visual description",
            "rich visual scene",
            "fully immersive visual scene",
        ],
        hypothesis_template="The level of visual description in this text is: {}",
    ),
    NLIConfig(
        candidate_labels=["none", "minimal", "low", "moderate", "high", "vivid"],
        hypothesis_template="The imagery level of this text is: {}",
    ),
]

# Sanity checks for counts
assert len(BINARY_CONFIGS) == 38, (
    f"Expected 38 binary configs, got {len(BINARY_CONFIGS)}"
)
assert len(TERTIARY_CONFIGS) == 4, (
    f"Expected 4 tertiary configs, got {len(TERTIARY_CONFIGS)}"
)
assert len(HEXA_CONFIGS) == 4, f"Expected 4 hexa configs, got {len(HEXA_CONFIGS)}"

# Combined list used by CLI
AVAILABLE_CONFIGS = BINARY_CONFIGS + TERTIARY_CONFIGS + HEXA_CONFIGS

# UNDERSCORE_STUDY_CONFIGS = [
#     NLIConfig(
#         candidate_labels=["not detailed", "detailed"],
#         hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
#     ),
#     NLIConfig(
#         candidate_labels=["not_detailed", "detailed"],
#         hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
#     ),
#     NLIConfig(
#         candidate_labels=["not visual", "visual"],
#         hypothesis_template="This text is {} in terms of sensory details, imagery, characters, environment, and vivid descriptions.",
#     ),
#     NLIConfig(
#         candidate_labels=["not_visual", "visual"],
#         hypothesis_template="This text is {} in terms of sensory details, imagery, characters, environment, and vivid descriptions.",
#     ),
#     NLIConfig(
#         candidate_labels=["not descriptive", "descriptive"],
#         hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
#     ),
#     NLIConfig(
#         candidate_labels=["not_descriptive", "descriptive"],
#         hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
#     ),
# ]


class NLIPersistentMetrics(PersistentDict):
    """Persistent metrics storage for NLI model evaluations."""

    @classmethod
    def create_for_config(
        cls,
        config: NLIConfig,
    ) -> "NLIPersistentMetrics":
        """Create metrics file for a config combination using the new unified format.

        Filename format: tcode_lcode_yyyy-MM-dd-hh-mm-ss.json
        - tcode: lowercase first letters of words in hypothesis_template (ignoring '{}')
        - lcode: lowercase first letters of each candidate label string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        def _tcode(template: str) -> str:
            # Ignore non-letter tokens (e.g., '{}'), take first letter of each word
            words = re.findall(r"[A-Za-z]+", template)
            return "".join(w[0].lower() for w in words if w)

        def _lcode(labels: list[str]) -> str:
            # Take the first character of each label string
            return "".join(
                (lbl.strip()[0].lower()) for lbl in labels if lbl and lbl.strip()
            )

        filename = (
            METRICS_DIR
            / f"{_tcode(config.hypothesis_template)}_{_lcode(config.candidate_labels)}_{timestamp}.json"
        )
        metrics = cls(filename)
        metrics.setdefault("hypothesis_template", config.hypothesis_template)
        metrics.setdefault("candidate_labels", config.candidate_labels)
        metrics.setdefault("models", [])
        return metrics

    def add_model_result(self, model_result: dict) -> None:
        """Add evaluation results for a model.

        Args:
            model_result: Dict with model_name, probabilities, device, batch_size, performance, etc.
        """
        self["models"].append(model_result)
        self._dump()


def calculate_performance_metrics(
    latencies: list[float],
) -> dict:
    """
    Calculate performance metrics from latency measurements.

    Args:
        latencies: List of per-sample latencies in milliseconds

    Returns:
        dict with keys: macs, flops, throughput, latency_mean, latency_std
    """
    latencies_arr = np.array(latencies)
    throughput = 1000.0 / np.mean(latencies_arr)  # samples per second

    return {
        "throughput": float(throughput),
        "latency_mean": float(np.mean(latencies_arr)),
        "latency_std": float(np.std(latencies_arr)),
    }


def evaluate_model_on_config(
    model: ORTNLIZeroshotClassifier,
    config: NLIConfig,
    texts: list[str],
    metrics: "NLIPersistentMetrics",
    labels: list[float],
    dataset_name: str,
) -> None:
    """
    Evaluate a model on a configuration in batches and update metrics after each batch.

    Evaluates samples once in batches, measuring per-batch latency and updating
    persistent metrics after each batch.

    Args:
        model: NLI model to evaluate
        config: Configuration with labels and template
        texts: List of texts to evaluate
        metrics: NLIPersistentMetrics instance for storing results
        labels: Ground truth labels for each sample, used to compute scores from probabilities
    """
    time_start = datetime.now().isoformat()
    model_name = model.get_model_name()
    device = get_device_name()

    all_probabilities = []
    all_latencies = []

    with model.set_options(
        candidate_labels=config.candidate_labels,
        hypothesis_template=config.hypothesis_template,
    ) as m:
        # Warm-up
        _ = list(m.evaluate_segments(texts[: min(2, len(texts))]))

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            start = time.perf_counter()
            results = list(m.evaluate_segments(batch))
            end = time.perf_counter()

            batch_probabilities = [probs for _, probs in results]
            all_probabilities.extend(batch_probabilities)

            batch_latency_ms = (end - start) * 1000 / len(batch)
            all_latencies.extend([batch_latency_ms] * len(batch))

    scores = [probs_to_score(probs) for probs in all_probabilities]
    corr = get_corrs(all_probabilities, labels)
    perf = calculate_performance_metrics(all_latencies)
    time_end = datetime.now().isoformat()

    model_result = {
        "model_name": model_name,
        "dataset": dataset_name,
        "probabilities": all_probabilities,
        "scores": scores,
        "corr": corr,
        "device": device,
        "batch_size": BATCH_SIZE,
        "performance": perf,
        "time_start": time_start,
        "time_end": time_end,
    }

    metrics.add_model_result(model_result)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NLI models on configurations and save results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate default model (roberta) on config 0 with train set
  python nli_eval.py --dataset train

  # Evaluate all models on all configs with test set
  python nli_eval.py --dataset test --models all --configs -1

  # Evaluate on both train and test datasets
  python nli_eval.py --dataset both --models roberta --configs 0

  # Evaluate specific models on specific configs
  python nli_eval.py --dataset train --models roberta deberta_large --configs 0 1

  # Use custom data file
  python nli_eval.py --dataset train --data-file /path/to/custom.parquet
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["train", "test", "both"],
        help="Dataset to evaluate on (use 'both' to evaluate on both train and test)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()) + ["all"],
        default=["roberta"],
        help="Models to evaluate. Options: {} or 'all' (default: roberta)".format(
            ", ".join(AVAILABLE_MODELS.keys())
        ),
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        choices=list(range(len(AVAILABLE_CONFIGS))) + [-1],
        default=[0],
        help="Config indices to evaluate (0-based). Use -1 for all configs (default: 0)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to parquet file with 'text' and optional 'label' columns. "
        "Default: DATA_DIR/datasets/small/{dataset}.parquet",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configs and exit",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_models:
        print("Available models:")
        for i, model_key in enumerate(AVAILABLE_MODELS.keys(), 1):
            print(f"  {i}. {model_key}")
        sys.exit(0)

    if args.list_configs:
        print("Available configurations:")
        for i, config in enumerate(AVAILABLE_CONFIGS):
            print(f"  {i}. {config}")
        sys.exit(0)

    # Resolve models
    if "all" in args.models:
        models_to_eval = list(AVAILABLE_MODELS.keys())
    else:
        models_to_eval = args.models

    # Resolve configs
    if -1 in args.configs:
        configs_to_eval = list(range(len(AVAILABLE_CONFIGS)))
    else:
        configs_to_eval = args.configs

    # Resolve datasets to evaluate
    if args.dataset == "both":
        datasets_to_eval = ["train", "test"]
    else:
        datasets_to_eval = [args.dataset]

    # Load datasets
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
        labels = df.get("label", None)
        if labels is not None:
            labels = labels.tolist()

        datasets[dataset_name] = {"texts": texts, "labels": labels}
        print(f"✓ Loaded {len(texts)} samples from {dataset_name} set")
    print("\n📊 Evaluation Summary:")
    print(
        f"  Datasets to evaluate: {len(datasets_to_eval)} - {', '.join(datasets_to_eval)}"
    )
    print(f"  Models to evaluate: {len(models_to_eval)} - {', '.join(models_to_eval)}")
    print(f"  Configs to evaluate: {len(configs_to_eval)} - {configs_to_eval}")
    print(
        f"  Total evaluations: {len(datasets_to_eval) * len(models_to_eval) * len(configs_to_eval)}"
    )
    print(f"  Results will be saved to: {METRICS_DIR}\n")

    # Evaluate each config, then each model on that config, for each dataset
    total_evals = len(datasets_to_eval) * len(models_to_eval) * len(configs_to_eval)
    current_eval = 0
    successful_evals = 0
    metrics_files = []

    for config_idx in configs_to_eval:
        config = AVAILABLE_CONFIGS[config_idx]
        metrics = NLIPersistentMetrics.create_for_config(config)
        metrics_files.append(metrics.filename)

        for model_key in models_to_eval:
            for dataset_name in datasets_to_eval:
                current_eval += 1
                dataset_data = datasets[dataset_name]

                print(
                    f"\n  [{current_eval}/{total_evals}] Evaluating {model_key} on config {config_idx} with {dataset_name} set..."
                )
                print(f"      Template: {config.hypothesis_template}")
                print(f"      Labels: {', '.join(config.candidate_labels)}")

                try:
                    model_class = AVAILABLE_MODELS[model_key]
                    model = model_class()

                    # Evaluate model on configuration (updates metrics internally)
                    evaluate_model_on_config(
                        model,
                        config,
                        dataset_data["texts"],
                        metrics,
                        dataset_data["labels"],
                        dataset_name,
                    )
                    print("      ✓ Metrics updated and persisted")
                    successful_evals += 1

                except Exception as e:
                    print(
                        f"      ❌ Error evaluating {model_key} on config {config_idx} with {dataset_name}: {e}"
                    )
                    import traceback

                    traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"✓ Evaluation complete! ({successful_evals}/{total_evals} successful)")
    print("✓ Metrics files saved to:")
    for filepath in metrics_files:
        print(f"  - {filepath}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
