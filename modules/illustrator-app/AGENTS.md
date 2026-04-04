# AGENTS.md

## Project Overview

Illustrator App is a full-stack application for analyzing and visualizing book content using natural language processing. It combines a Django REST API backend with a Nuxt 3 frontend, using Docker for containerization and orchestration.

### Tech Stack
- **Frontend**: Nuxt 3, Vue 3, TypeScript, Tailwind CSS, DaisyUI
- **Backend**: Django 5.2, Django Ninja (REST API), Python 3.11+
- **Infrastructure**: Docker Compose, Redis, PostgreSQL
- **Package Manager**: pnpm (frontend), uv (backend)
- **Task Queue**: Dramatiq with Redis backend
- **WebSockets**: Django Channels with Daphne
- **LLM Integration**: Hugging Face Transformers, ONNX models

## Project Structure

```
illustrator-app/
├── services/
│   ├── api/              # Django backend
│   │   ├── api/          # Django project settings
│   │   ├── core/         # Main application logic
│   │   ├── manage.py
│   │   ├── pyproject.toml
│   │   ├── build.sh
│   │   └── run.sh
│   └── frontend/         # Nuxt 3 application
│       ├── app/          # Nuxt app directory
│       ├── nuxt.config.ts
│       └── package.json
├── compose/              # Docker configurations
│   ├── api/
│   └── frontend/
├── data/                 # Data files
├── docker-compose.yml
├── docker-compose.debug.yml
├── docker-compose.prod.yml
└── package.json          # Root eslint config
```

## Dependency Management

**Backend (Python/API)**:
- Uses `uv` for dependency management with `pyproject.toml`
- Sync dependencies: `docker compose run --rm --user root api uv sync`
- Special sources configured:
  - PyTorch CPU index for efficient ML dependencies
  - BookNLP from custom GitHub repository
  - spaCy English model from GitHub releases

**Frontend (Node/Nuxt)**:
- Uses `pnpm` as package manager
- Install: `docker compose run --rm frontend pnpm install`
- Lock file: `pnpm-lock.yaml` (commit this when dependencies change)

## Development Workflow

### Starting the Development Stack

**API only**:
```bash
docker compose up api
```

**Frontend only**:
```bash
docker compose up frontend
```

### Frontend Development

Start the dev server:
```bash
docker compose up frontend
```

The frontend runs on `http://localhost:3000` with hot module replacement (HMR) enabled. File changes trigger automatic rebuilds.

**Dev commands inside container**:
```bash
docker compose run --rm frontend pnpm dev      # Start dev server
docker compose run --rm frontend pnpm build    # Production build
```

### Running Commands in Containers

**API (as root)**:
```bash
docker compose run --rm --user root api bash
```

**Frontend**:
```bash
docker compose run --rm frontend sh
```

### Syncing API Schema to Frontend

After backend endpoint or schema changes, sync the OpenAPI types to frontend:
```bash
# Ensure API service is running
docker compose run --rm frontend pnpm gen-types
```

This fetches the OpenAPI schema from `http://api:8000/api/openapi.json` and generates TypeScript types in `app/types/schema.d.ts`.

### Pre-commit Validation

Pre-commit runs linting and formatting checks:

**Staged files only**:
```bash
pre-commit run
```

**All files**:
```bash
pre-commit run --all-files
```

**Bypass pre-commit**:
```bash
git commit -m "..." --no-verify
```

## Code Style

### Frontend (TypeScript/Vue)

**File Organization**:
- Components: `app/components/`
- Pages: `app/pages/`
- Composables: `app/composables/`
- Utils: `app/utils/`
- Plugins: `app/plugins/`
- Types: `app/types/`

**Conventions**:
- Use PascalCase for Vue component files and exports
- Use camelCase for utility functions and variables
- TypeScript strict mode enabled
- Vue 3 Composition API with TypeScript setup

**Import patterns**:
- Use absolute imports with `@/` alias
- ESLint plugin-import-x configured for TypeScript resolution
- Avoid wildcard imports (`import *`)

### Backend (Python/Django)

**File Organization**:
- Models: defined in `core/` app
- API endpoints: `core/api.py`
- Tasks: `core/tasks.py`
- WebSocket consumers: `core/consumers.py`
- Utilities: `core/utils.py` and `core/tools/`
- Migrations: `core/migrations/`

**Conventions**:
- Type hints recommended for function signatures
- Django naming conventions (models singular, tables plural)
- Use Django ORM for database operations

**Pre-commit checks**:
- Python formatting and linting (configured in `.pre-commit-config.yaml`)

## Additional Notes

### Architecture Highlights

- **Frontend-Backend Communication**: REST API with OpenAPI schema auto-generated types
- **Real-time Features**: Django Channels for WebSocket support via Daphne
- **Async Tasks**: Dramatiq with Redis broker for background job processing
- **NLP Processing**: Integration with Hugging Face models and ONNX runtime for efficient inference
- **Model Caching**: Pre-downloaded models cached in `services/api/model_cache/`

#### Frontend Component Architecture

**Props and Events**: See [HeatmapViewer API documentation](../services/frontend/README.md#heatmapviewer-component) for detailed props and events reference.

### Performance Considerations

- ONNX models used for optimized inference (MiniLM, ModernBERT, RoBERTa)
- Redis for caching and task queue
- Channels for WebSocket scalability
- Multi-process/thread Dramatiq workers for parallel task processing

### Important Files

- **Frontend Config**: `services/frontend/nuxt.config.ts`
- **Backend Config**: `services/api/api/settings/` (base, local, production)
- **Orchestration**: `docker-compose.yml` and compose-specific overrides
- **Root ESLint**: `eslint.config.mjs` (applies to frontend)
- **Pre-commit Config**: `.pre-commit-config.yaml`

### Common Development Workflows

**Adding dependencies**:
- **Backend**: Edit `services/api/pyproject.toml`, then `docker compose run --rm --user root api uv sync`
- **Frontend**: Edit `services/frontend/package.json`, then `docker compose run --rm frontend pnpm install`
- Commit lock files (`pnpm-lock.yaml` and `uv.lock` if generated)
