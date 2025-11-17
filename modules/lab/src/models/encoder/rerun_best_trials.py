#!/usr/bin/env python3

from typing import Literal
import argparse
import optuna
import sys
from datetime import datetime
from utils import DATA_DIR
from models.encoder.hyperparam_search import PROVIDER_MAPPING


def get_storage_url():
    """Get the Optuna storage URL."""
    return f"sqlite:///{(DATA_DIR / 'optuna' / 'optuna_db.sqlite3').as_posix()}"


def get_top_trials(study: optuna.Study, n_top: int = 5):
    """
    Get the top N trials from a study.

    Args:
        study: The Optuna study
        n_top: Number of top trials to retrieve

    Returns:
        List of top trials sorted by value (best first)
    """
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise ValueError(f"No completed trials found in study '{study.study_name}'")

    # Sort by value (ascending for minimize)
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)

    return sorted_trials[:n_top]


def rerun_best_trials(
    study_name: str,
    provider_name: str,
    embeddings: Literal["minilm", "mbert"],
    include_large: bool,
    n_top: int = 10,
    new_study_suffix: str = "_rerun",
):
    """
    Rerun the best trials from an existing study.

    Args:
        study_name: Name of the existing study
        provider_name: Name of the provider (ridge, svm, rf, catboost, finetuned-mbert)
        embeddings: Embedding type ('minilm' or 'mbert')
        include_large: Whether to include large dataset
        n_top: Number of top trials to rerun
        new_study_suffix: Suffix to add to the new study name
    """
    storage_url = get_storage_url()

    print(f"Loading study: {study_name}")
    try:
        existing_study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
        )
    except KeyError:
        print(f"Error: Study '{study_name}' not found")
        print("\nAvailable studies:")
        studies = optuna.get_all_study_summaries(storage=storage_url)
        for s in studies:
            print(f"  - {s.study_name}")
        sys.exit(1)

    print(f"Found {len(existing_study.trials)} total trials")

    top_trials = get_top_trials(existing_study, n_top=n_top)
    print(f"\nTop {len(top_trials)} trials:")
    for i, trial in enumerate(top_trials, 1):
        print(f"  {i}. Trial {trial.number}: MSE = {trial.value:.4f}")
        for key, value in trial.params.items():
            print(f"     {key}: {value}")

    if provider_name not in PROVIDER_MAPPING:
        print(f"Error: Unknown provider '{provider_name}'")
        print(f"Available providers: {list(PROVIDER_MAPPING.keys())}")
        sys.exit(1)

    provider_class = PROVIDER_MAPPING[provider_name]
    provider = provider_class(embeddings=embeddings, include_large=include_large)
    objective_fn = provider.get_objective_fn()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_study_name = f"{study_name}{new_study_suffix}_{timestamp}"
    print(f"\nCreating new study: {new_study_name}")

    new_study = optuna.create_study(
        study_name=new_study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=False,  # Create fresh study
    )

    print(f"\nEnqueuing {len(top_trials)} trials...")
    for trial in top_trials:
        new_study.enqueue_trial(trial.params)

    print(f"\nRunning {len(top_trials)} trials...")
    new_study.optimize(objective_fn, n_trials=len(top_trials))

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOriginal study '{study_name}':")
    print(f"  Best MSE: {existing_study.best_value:.4f}")

    print(f"\nNew study '{new_study_name}':")
    print(f"  Best MSE: {new_study.best_value:.4f}")
    print("  Best hyperparameters:")
    for key, value in new_study.best_params.items():
        print(f"    {key}: {value}")

    # Compare with original best
    improvement = existing_study.best_value - new_study.best_value
    if improvement > 0:
        print(f"\n✓ Improvement: {improvement:.4f} (lower MSE)")
    elif improvement < 0:
        print(f"\n✗ Regression: {abs(improvement):.4f} (higher MSE)")
    else:
        print("\n= No change in MSE")

    return new_study


def list_studies():
    """List all available studies in the database."""
    storage_url = get_storage_url()
    studies = optuna.get_all_study_summaries(storage=storage_url)

    if not studies:
        print("No studies found in the database")
        return

    print(f"Found {len(studies)} studies:\n")
    for study in studies:
        print(f"  {study.study_name}")
        print(f"    Trials: {study.n_trials}")
        if study.best_trial:
            print(f"    Best MSE: {study.best_trial.value:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerun best trials from an Optuna study (e.g., after code changes)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available studies")

    # Rerun command
    rerun_parser = subparsers.add_parser("rerun", help="Rerun best trials from a study")
    rerun_parser.add_argument(
        "study_name",
        type=str,
        help="Name of the existing study to rerun best trials from",
    )
    rerun_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        required=True,
        choices=list(PROVIDER_MAPPING.keys()),
        help="Provider name (ridge, svm, rf, catboost, finetuned-mbert)",
    )
    rerun_parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        required=True,
        choices=["minilm", "mbert"],
        help="Embedding type to use: 'minilm' or 'mbert'",
    )
    rerun_parser.add_argument(
        "--lg",
        action="store_true",
        help="Include large dataset in training",
    )
    rerun_parser.add_argument(
        "-n",
        "--n-top",
        type=int,
        default=10,
        help="Number of top trials to rerun (default: 10)",
    )
    rerun_parser.add_argument(
        "--suffix",
        type=str,
        default="_rerun",
        help="Suffix to add to new study name (default: '_rerun')",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_studies()
    elif args.command == "rerun":
        rerun_best_trials(
            study_name=args.study_name,
            provider_name=args.provider,
            embeddings=args.embeddings,
            include_large=args.lg,
            n_top=args.n_top,
            new_study_suffix=args.suffix,
        )
    else:
        parser.print_help()
        sys.exit(1)
