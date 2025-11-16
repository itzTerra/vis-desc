import optuna
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_study.py <study_name>")
        sys.exit(1)

    study_name = sys.argv[1]
    optuna.delete_study(
        study_name=study_name,
        storage=f"sqlite:///{(Path(__file__).parent / 'optuna_db.sqlite3').as_posix()}",
    )
