FROM node:lts-alpine AS base

# Update dependencies
RUN apk update \
    # Install git
    && apk add --no-cache openssh git curl \
    && corepack enable \
    && corepack prepare pnpm@10.12.2 --activate

# Set up the working directory
WORKDIR /app

# Specify the host variable
ENV HOST=0.0.0.0

# Expose the Nuxt port
EXPOSE 3000

# Export the Vite websocket port
EXPOSE 24678

FROM base AS dev

COPY .git ./
COPY package.json ./

WORKDIR /app

CMD ["pnpm", "dev"]

# --------------------------------------------

FROM base AS builder

# Set up the working directory
WORKDIR /app
COPY ./services/frontend/ ./

# Install build dependencies
RUN pnpm config set store /root/.pnpm-store
RUN --mount=type=cache,target=/root/.pnpm-store,sharing=locked \
    pnpm install --prod

COPY .git ./
COPY package.json ./

# Build the app
RUN pnpm build

# --------------------------------------------

FROM node:lts-alpine AS release

RUN apk update && apk add --no-cache curl

# Set up the working directory
WORKDIR /app

# Copy the built app
COPY --from=builder /app/.output .

# Specify the host variable
ENV HOST=0.0.0.0
ENV PORT=3000

ENTRYPOINT ["node", "./server/index.mjs"]
