## Install
1. [Install Docker](https://docs.docker.com/install/)
2. Install pre-commit
   1. `pip install pre-commit` / `uv tool install pre-commit` OR (optional) Install uv venv locally `uv sync`
   2. Install node_modules `pnpm install`
   3. `pre-commit install`
3. Build Docker containers `docker compose build`
4. Install frontend node_modules `docker compose run --rm frontend pnpm install`
5. *Linux*: [Increase file descriptor limit](https://vitejs.dev/guide/troubleshooting.html#requests-are-stalled-forever)
   1. Add `DefaultLimitNOFILE=65535` into `/etc/systemd/user.conf` and `/etc/systemd/system.conf`
   2. Add `* hard nofile 65535` and `* soft nofile 65535`  into `/etc/security/limits.conf`
6. Start stack `docker compose up`
   1. API only: `docker compose up api`
   2. FE only: `docker compose up frontend`

## Running commands in a Docker container
API: `docker compose run --rm --user root api bash`

FE: `docker compose run --rm frontend sh`

## API

**Sync endpoint types from backend to frontend** (after endpoint contract or schema changes)  
(!) API service needs to be up, as this command makes a request to endpoint serving the OpenAPI schema  
```sh
docker compose run --rm frontend pnpm gen-types
```

### Dramatiq

```bash
docker compose run --rm --user root api python manage.py rundramatiq -p 4 -t 2
```

## Pre-commit
Staged only  
`pre-commit run`

All  
`pre-commit run --all-files`

Bypass pre-commit  
`git commit -m "..." --no-verify`

## Troubleshooting
`docker compose run --rm -u root api uv sync`

`docker compose run --rm frontend pnpm install`

`docker compose build`
