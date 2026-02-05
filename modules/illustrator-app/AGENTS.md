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

## Setup Commands

### Initial Setup
1. Install Docker: https://docs.docker.com/install/
2. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   # or with uv:
   uv tool install pre-commit
   ```
3. Install pnpm dependencies:
   ```bash
   pnpm install
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
5. Build Docker containers:
   ```bash
   docker compose build
   ```
6. Install frontend dependencies in container:
   ```bash
   docker compose run --rm frontend pnpm install
   ```
7. **Linux Only**: Increase file descriptor limits for Vite development server
   - Add `DefaultLimitNOFILE=65535` to `/etc/systemd/user.conf` and `/etc/systemd/system.conf`
   - Add `* hard nofile 65535` and `* soft nofile 65535` to `/etc/security/limits.conf`

### Dependency Management

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

**All services**:
```bash
docker compose up
```

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
docker compose run --rm frontend pnpm generate # Static generation
docker compose run --rm frontend pnpm preview  # Preview production build
docker compose run --rm frontend pnpm postinstall # Generate Nuxt artifacts
```

### API Development

Start the debug API:
```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up api
```

This enables Python debugging on port 5678.

**Dev commands inside container**:
```bash
docker compose run --rm --user root api bash   # Interactive shell
docker compose run --rm api python manage.py shell
docker compose run --rm api python manage.py migrate
docker compose run --rm api python manage.py makemigrations
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

### Dramatiq Task Queue

Run Dramatiq worker manually:
```bash
docker compose run --rm --user root api python manage.py rundramatiq -p 2 -t 2
```

Options:
- `-p`: Number of worker processes
- `-t`: Number of worker threads per process

The main API container runs this automatically with 1 process and 1 thread. Adjust for development workloads as needed.

## Testing Instructions

### Frontend Tests
Currently no dedicated test command; coverage using Vue component testing or E2E tests would go here.

### API Tests
```bash
# Run Django tests (if configured)
docker compose run --rm api python manage.py test
```

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
- Follow PEP 8 style guide
- Type hints recommended for function signatures
- Django naming conventions (models singular, tables plural)
- Use Django ORM for database operations

**Pre-commit checks**:
- Python formatting and linting (configured in `.pre-commit-config.yaml`)
- Import sorting
- Trailing whitespace removal

## Build and Deployment

### Development Environments

**Local development** (default):
```bash
ENVIRONMENT=local docker compose build
```

Env files: `.envs/local/.api` and `.envs/local/.frontend`

**Debug mode**:
```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml build api
```

Enables Python remote debugging via debugpy on port 5678.

**Production**:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
```

Uses production optimizations and configurations.

### Build Commands

**Build all services**:
```bash
docker compose build
```

**Build specific service**:
```bash
docker compose build api
docker compose build frontend
```

### Environment Variables

Configuration uses `.envs/` directory with environment-specific files:

```
.envs/
├── local/
│   ├── .api
│   └── .frontend
├── debug/
│   ├── .api
│   └── .frontend
└── prod/
    ├── .api
    └── .frontend
```

Use `ENVIRONMENT` variable to select which set to use:
```bash
ENVIRONMENT=local docker compose up
ENVIRONMENT=debug docker compose -f docker-compose.yml -f docker-compose.debug.yml up
ENVIRONMENT=prod docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### Port Configuration

Default ports (configurable via environment variables):
- **Frontend**: 3000 (`FE_PORT`)
- **API**: 8000 (`API_PORT`)
- **Redis**: 6379
- **Python Debugger**: 5678 (debug mode only)
- **Vite HMR**: 24678

## Troubleshooting

### Common Issues

**Dependencies not found**:
```bash
# Backend
docker compose run --rm --user root api uv sync

# Frontend
docker compose run --rm frontend pnpm install
```

**Docker build failures**:
```bash
docker compose build --no-cache
```

**Container networking issues**:
- DNS configured to use 8.8.8.8 and 8.8.4.4
- All services on default network

**API not responding**:
- Ensure Redis is running: `docker compose ps redis`
- Check API logs: `docker compose logs api`
- Verify port mapping: `docker compose port api 8000`

**Frontend not rebuilding**:
- Linux users: Check file descriptor limits (see Setup section)
- Check Vite HMR port 24678 is accessible
- Restart frontend: `docker compose restart frontend`

**Dramatiq tasks not processing**:
- Verify Redis connection
- Check task logs: `docker compose logs api`
- Ensure `ENABLE_DRAMATIQ` is not set to "off"

### Debug Mode

Enable Python debugging:
```bash
docker compose -f docker-compose.yml -f docker-compose.debug.yml up api
```

Attach debugger to port 5678. Debugpy is configured in the API Dockerfile.

### Logs

View logs:
```bash
docker compose logs [service]  # Single service
docker compose logs -f         # All services, follow mode
docker compose logs api        # API only
docker compose logs frontend   # Frontend only
```

## Key Commands Reference

| Task | Command |
|------|---------|
| Start all services | `docker compose up` |
| Start frontend only | `docker compose up frontend` |
| Start API with debug | `docker compose -f docker-compose.yml -f docker-compose.debug.yml up api` |
| Run bash in API | `docker compose run --rm --user root api bash` |
| Run shell in frontend | `docker compose run --rm frontend sh` |
| Build services | `docker compose build` |
| View logs | `docker compose logs -f [service]` |
| Lint code | `pre-commit run --all-files` |
| Sync API types to frontend | `docker compose run --rm frontend pnpm gen-types` |
| Run Dramatiq worker | `docker compose run --rm --user root api python manage.py rundramatiq -p 2 -t 2` |
| Migrate database | `docker compose run --rm api python manage.py migrate` |
| Make migrations | `docker compose run --rm api python manage.py makemigrations` |

## Additional Notes

### Architecture Highlights

- **Frontend-Backend Communication**: REST API with OpenAPI schema auto-generated types
- **Real-time Features**: Django Channels for WebSocket support via Daphne
- **Async Tasks**: Dramatiq with Redis broker for background job processing
- **NLP Processing**: Integration with Hugging Face models and ONNX runtime for efficient inference
- **Model Caching**: Pre-downloaded models cached in `services/api/model_cache/`

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

**Feature development**:
1. Create feature branch
2. Run `docker compose up` to start all services
3. Make changes to frontend/API
4. Frontend auto-reloads on file changes
5. Sync types if API changes: `docker compose run --rm frontend pnpm gen-types`
6. Test via API client or frontend UI
7. Run `pre-commit run --all-files` before committing
8. Push and create PR

**Debugging API issues**:
1. Start debug API: `docker compose -f docker-compose.yml -f docker-compose.debug.yml up api`
2. Attach debugger to port 5678
3. Set breakpoints in Python code
4. Make requests from frontend to hit breakpoints

**Adding dependencies**:
- **Backend**: Edit `services/api/pyproject.toml`, then `docker compose run --rm --user root api uv sync`
- **Frontend**: Edit `services/frontend/package.json`, then `docker compose run --rm frontend pnpm install`
- Commit lock files (`pnpm-lock.yaml` and `uv.lock` if generated)
