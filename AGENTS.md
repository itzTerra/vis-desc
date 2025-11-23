# AGENTS.md

## Project Overview

vis-desc is a research application for scoring visual descriptiveness/imageability of text. It consists of two main modules:

- **Lab module** (`modules/lab/`): Python-based data processing, machine learning model training, and feature extraction using libraries like BookNLP, sentence-transformers, and scikit-learn.
- **Illustrator-app** (`modules/illustrator-app/`): Web application with a Django Ninja API backend and Nuxt.js frontend for PDF processing, text segmentation, and real-time evaluation.

Data flow: PDF upload → text extraction → segmentation → ML scoring → visualization with generated images.

Key technologies: Python (uv), Node.js (pnpm), Django, Nuxt.js, Redis, Docker.

## Setup Commands

### Python Environment (Lab Module)
- Install uv: `pip install uv`
- Install dependencies: `cd modules/lab && uv sync`
- Download spaCy model: `uv run python -m spacy download en_core_web_sm`

### Node.js Environment (Illustrator App)
- Install pnpm: `npm install -g pnpm`
- Install frontend dependencies: `cd modules/illustrator-app/services/frontend && pnpm install`
- Install API dependencies: `cd modules/illustrator-app/services/api && uv sync`

### Database and Services
- Start Redis: Use Docker or local Redis instance
- For full app: `cd modules/illustrator-app && docker-compose up`

## Development Workflow

### Model Training
- Train encoder models: `cd modules/lab && uv run python3 src/models/encoder/training.py -e <encoder> --train --test <model>`
  - Encoders: minilm, mbert
  - Models: ridge, svm, rf, catboost
  - Example: `uv run python3 src/models/encoder/training.py -e minilm --train --test catboost`

### Inter-Annotator Agreement (IAA)
- Calculate IAA: `cd modules/lab && uv run python3 src/agreement.py <file> --details [--old <files>]`
- Generate CSV: `uv run python3 src/agreement.py <annotationFile> --csv <outputCsvFile> --from-csv <inputCsvFile>`

### Hyperparameter Tuning
- Run Optuna studies: Use scripts in `modules/lab/data/optuna/`
- Delete study: `cd modules/lab && uv run python3 data/optuna/delete_study.py <name>`

### Frontend Development
- Start dev server: `cd modules/illustrator-app/services/frontend && pnpm dev`
- Generate types: `pnpm gen-types` (requires API running)

### API Development
- Run API server: `cd modules/illustrator-app/services/api && uv run python manage.py runserver`
- With Dramatiq: `uv run python manage.py rundramatiq`

## Testing Instructions

### Model Testing
- Test models during training: Include `--test` flag in training commands
- Evaluate models: Run training scripts with test datasets

### Code Quality
- Run pre-commit hooks: `cd modules/illustrator-app && pre-commit run --all-files`
- Ruff linting: `cd modules/lab && uv run ruff check . --fix`
- Ruff formatting: `uv run ruff format .`

### Integration Testing
- Use Docker Compose for full stack testing: `cd modules/illustrator-app && docker-compose up`
- Test PDF processing and segmentation workflows

## Code Style Guidelines

- **Python**: Use type hints throughout, especially for NDArray, Dict, List. Use dataclasses for data structures. Seed with `set_seed(42)` for reproducibility. Stratified K-fold cross-validation.
- **JavaScript/TypeScript**: Follow ESLint rules (when enabled). Use Nuxt 4 conventions.
- **File Organization**: Feature extraction in `text2features.py`, evaluators loaded from ONNX models.
- **Imports**: Use absolute imports, avoid relative imports where possible.
- **Naming**: Descriptive variable names, follow PEP 8 for Python.

## Build and Deployment

### Frontend Build
- Build production: `cd modules/illustrator-app/services/frontend && pnpm build`
- Generate static: `pnpm generate`
- Preview: `pnpm preview`

### API Build
- Use Docker: `cd modules/illustrator-app && docker-compose build`

### Full Deployment
- Production compose: `docker-compose -f docker-compose.prod.yml up`
- Debug mode: `docker-compose -f docker-compose.debug.yml up`

## Pull Request Guidelines

- Title format: [module] Brief description (e.g., [lab] Add new encoder support)
- Run pre-commit hooks before committing
- Ensure tests pass (model training with --test)
- Update dependencies in pyproject.toml or package.json as needed

## Additional Notes

- **Package Managers**: Use `uv` for Python, `pnpm` for Node.js
- **Custom Dependencies**: BookNLP from custom fork: `git = "https://github.com/itzTerra/booknlp"`
- **Data Storage**: Annotations in Label Studio format, processed books in `data/books/`
- **Performance**: Use GPU for training if available (check `gpu_info.py`)
- **Troubleshooting**: Ensure Redis is running for async tasks. Check Docker logs for API issues.
