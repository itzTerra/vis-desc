# AI Coding Instructions for vis-desc

## Project Overview
vis-desc is a research app for scoring visual descriptiveness/imageability of text. It consists of:
- **Lab module** (`modules/lab/`): Python-based data processing, ML model training, and feature extraction using BookNLP, sentence-transformers, and scikit-learn.
- **Illustrator-app** (`modules/illustrator-app/`): Web application with Django Ninja API backend and Nuxt.js frontend for PDF processing, text segmentation, and real-time evaluation.

Data flow: PDF upload → text extraction → segmentation → ML scoring → visualization with generated images.

## Key Workflows
- **Model Training**: Use `python3 src/models/encoder/training.py -e <encoder> --train --test <model>` (e.g., `-e minilm --train --test catboost`). Encoders: minilm, mbert. Models: ridge, svm, rf, catboost.
- **Inter-Annotator Agreement**: `python3 modules/lab/src/agreement.py <file> --details [--old <files>]` for IAA calculation with ratings 0-5, adjusted by visual_action.
- **Hyperparameter Tuning**: Optuna studies in `modules/lab/data/optuna/`. Delete with `uv run python3 data/optuna/delete_study.py <name>`.
- **Package Management**: Use `uv` for Python dependencies (e.g., `uv run python ...`), `pnpm` for Node.js.

## Architecture Patterns
- **Feature Extraction**: `text2features.py` uses BookNLP for token-level features (POS, lemma, events) and sentence-transformers for embeddings.
- **Evaluation**: API uses Dramatiq for async segment processing via Redis/Channels. Evaluators loaded from ONNX models.
- **Data Storage**: Annotations in Label Studio format, processed books in `data/books/`, models in `data/models/`.

## Coding Conventions
- Use dataclasses for data structures (e.g., `TokenData` in `text2features.py`).
- Type hints throughout, especially for NDArray, Dict, List.
- Seed setting with `set_seed(42)` for reproducibility.
- Stratified K-fold cross-validation for training.
- Custom metrics: MSE, accuracy, precision/recall/F1, confusion matrix.

## Dependencies
- Python: torch, sentence-transformers, catboost, optuna, booknlp (custom fork).
- JS: Nuxt 4, TailwindCSS, DaisyUI, Vue PDF Embed.

Reference: `modules/lab/pyproject.toml`, `modules/illustrator-app/services/frontend/package.json`.
