---
name: format-lint
description: Runs the formatter and linter configured in the project to fix warnings after making code edits.
---

# Format and Lint

All commands are to be executed from project root dir.

**Run ESLint** (Typescript, Vue):
```bash
pnpm lint
```

ESLint config: `eslint.config.mjs`

**Run Ruff Format** (Python):
```bash
docker compose run --rm --user root api uv run ruff format .
```

**Run Ruff Lint** (Python):
```bash
docker compose run --rm --user root api uv run ruff check . --fix
```

Ruff config: `services/api/pyproject.toml` (under `[tool.ruff]`)
