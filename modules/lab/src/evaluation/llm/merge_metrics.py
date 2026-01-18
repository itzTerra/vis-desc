import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return json.load(fh)


def filter_models_with_outputs(
    models: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [model for model in models if len(model.get("outputs", [])) > 0]


def merge_metric_files(files: Iterable[Path], unsafe: bool = False) -> dict[str, Any]:
    metrics_list = [load_metrics(path) for path in files]

    prompts = [m.get("prompt") for m in metrics_list if m.get("prompt")]
    unique_prompts = set(prompts)
    if len(unique_prompts) > 1 and not unsafe:
        raise ValueError("Cannot merge files with different prompts")

    prompt = prompts[0] if prompts else None
    prompt_token_count = next(
        (
            m.get("prompt_token_count")
            for m in metrics_list
            if m.get("prompt_token_count")
        ),
        None,
    )

    combined_models: list[dict[str, Any]] = []
    for m in metrics_list:
        combined_models.extend(list(m.get("models", [])))
    merged_models = filter_models_with_outputs(combined_models)

    return {
        "prompt": prompt,
        "prompt_token_count": prompt_token_count,
        "models": merged_models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge one or more LLM metric JSON files and drop empty model entries."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Paths to metric JSON files (2+ recommended)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for merged metrics (default: <first_file>_merged.json)",
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Allow merging files with different prompts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        first = args.files[0]
        output_path = first.with_name(f"{first.stem}_merged.json")

    merged = merge_metric_files(args.files, unsafe=args.unsafe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"Merged metrics saved to {output_path}")


if __name__ == "__main__":
    main()
