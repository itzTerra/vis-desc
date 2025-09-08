# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ARG APT_CACHE_PREFIX=bookworm-apt
ENV TZ=Europe/Prague

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN --mount=type=cache,id=${APT_CACHE_PREFIX}-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=${APT_CACHE_PREFIX}-lib,target=/var/lib/apt,sharing=locked \
    apt update && apt --no-install-recommends install -y \
    build-essential \
    tzdata \
    wget \
    gnupg2 \
    curl  \
    # Git dependencies
    git openssh-client expect \
    libxml2-dev libxslt-dev libxmlsec1-dev pkg-config libxmlsec1-openssl \
    gettext \
    gnupg \
    rsync \
    unzip \
    nano \
    gcc \
    lsb-release


ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

FROM base AS builder

ENV UV_CACHE_DIR=/app/.caches/uv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    VENV=/home/django/.venv

ENV PATH=${VENV}/bin:${PATH} \
    UV_PROJECT_ENVIRONMENT=${VENV}

COPY ./services/api/pyproject.toml ./services/api/uv.lock /app/

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=./services/api/uv.lock,target=./services/api/uv.lock \
    --mount=type=bind,source=./services/api/pyproject.toml,target=./services/api/pyproject.toml \
    uv sync --frozen --no-install-project \
    $(if [ "$TARGET" = "release" ]; then echo "--no-dev"; else echo "--compile-bytecode"; fi)


# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD ./services/api ./
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

# Dev
FROM builder AS dev

CMD ["sh", "-c", "uv run manage.py migrate; uv run manage.py runserver 0.0.0.0:8000;"]

# Production
FROM builder AS release

ENV DJANGO_SETTINGS_MODULE=project.settings \
    PYTHONUNBUFFERED=1 \
    PORT=8000

CMD ["sh", "-c", "uv run python manage.py migrate && exec uv run daphne -b 0.0.0.0 -p $PORT project.asgi:application"]
