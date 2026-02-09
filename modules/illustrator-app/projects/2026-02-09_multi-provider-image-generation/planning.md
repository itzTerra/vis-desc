---
type: improvement
size: M
---

# Multi-Provider Image Generation

## Overview
Add a second image generation provider (Cloudflare Workers AI) with priority-based provider selection and automatic failover. Providers are enabled by credential presence only. Generation remains synchronous.

## Goals
- Provide a second provider (Cloudflare Workers AI) alongside Pollinations.
- Improve resilience with automatic failover to a second provider when the first fails.
- Keep generation synchronous and simple for the frontend.
- Use a simple, stateless priority-based provider order (Pollinations → Cloudflare).

## User Stories

### As a user, I want image generation to keep working when a provider fails so that I can still get an image without retrying manually.

**Acceptance criteria:**
- [ ] If the first provider fails, the system tries the other enabled provider once.
- [ ] If both providers fail, the API returns a clear error message.

### As an operator, I want providers to be automatically available when credentials are present.

**Acceptance criteria:**
- [ ] Each provider is enabled if its API key environment variable is set and non-empty.
- [ ] Providers are skipped gracefully if credentials are missing.
- [ ] The API returns an error when no providers are available.

### As a user, I want consistent behavior in the editor while generation stays synchronous so that I do not see new loading states or background jobs.

**Acceptance criteria:**
- [ ] The frontend waits for the response as it does today.
- [ ] No new background job or websocket state tracking is introduced.

## Requirements

### Functional
- Add Cloudflare Workers AI provider alongside Pollinations.
- Select providers in priority order (Pollinations → Cloudflare) with automatic failover.
- Skip providers with missing or empty credentials.
- Define all provider credentials in Django settings (`services/api/api/settings/base.py`) using `env.str(..., default=None)` pattern.
- Return HTTP status codes only for error detection:
  - 200 + binary PNG on success
  - 400 + plain text on validation error (empty/invalid prompt)
  - 500 + plain text on provider failure (all providers failed or none available)
- Make Cloudflare model configurable via `CLOUDFLARE_MODEL` setting (default: `@cf/black-forest-labs/flux-2-klein-4b`).

### Non-functional
- Keep generation synchronous.
- No Dramatiq or websocket changes.
- Maintain existing API endpoint contracts for success responses (binary PNG).
- Per-provider request timeout (30 seconds default, configurable).
- Stateless, no distributed state or locking required.
- No startup-time provider registration; check credentials on each request.

## Scope

### Included
- Backend provider integration and selection logic.
- Stateless, priority-based provider selection with automatic failover.
- Credential presence detection for automatic provider enabling.
- Per-provider timeout configuration.
- Error handling improvements for generation endpoints.
- Configuration template updates for new provider settings (Cloudflare creds and model).

### Excluded
- Asynchronous processing or background queues.
- Frontend UI changes beyond error messaging behavior.
- Provider usage analytics.
- Persistent provider performance metrics.

---

## Current State
- Image generation uses a single Pollinations provider in `get_image_bytes()` and `get_image_url()`.
- `/api/gen-image` and `/api/gen-image-bytes` always call Pollinations.
- The frontend `ImageEditor` calls `/api/gen-image-bytes` and handles blob responses synchronously.
- Configuration includes `POLLINATIONS_API_KEY` in the local API template only.

## Key Files
- services/api/api/settings/base.py - Django settings with provider credentials and timeouts.
- services/api/core/tools/text2image.py - ImageProvider abstract class, provider implementations, and generation logic.
- services/api/core/api.py - Image generation endpoints.
- services/frontend/app/components/ImageEditor.vue - Frontend generation call and error handling.
- .envs/.local/.api.template - Local API env template with optional provider credentials.
- .envs/.production/.api.template - Production API env template.

## Existing Patterns
- Settings are defined in `services/api/api/settings/base.py` using `env.str()` and `env.int()` with `default=None` for optional values.
- Credentials checked via simple `if api_key:` pattern at provider initialization.
- Provider classes initialized with credentials and expose `get_image_bytes()` and `is_available()` methods.

## Provider Architecture

### ImageProvider Abstract Base Class
All providers inherit from `ImageProvider` with these methods:
- `get_image_bytes(text: str) -> bytes`: Returns PNG image bytes from text prompt
- `is_available() -> bool`: Returns True if credentials are configured; checked at provider initialization

### Provider Implementations

**PollinationsProvider**:
- Credentials: `POLLINATIONS_API_KEY` from Django settings
- Default available if key is set and non-empty
- Method: `get_image_bytes()` calls Pollinations API synchronously

**CloudflareProvider**:
- Credentials: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, and `CLOUDFLARE_MODEL` from Django settings
- Default available if all three are set and non-empty
- **Endpoint**: `POST https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{model}`
- **Auth**: Bearer token in `Authorization: Bearer {CLOUDFLARE_API_TOKEN}` header
- **Request body**: JSON `{"prompt": "text description"}`
- **Response (200)**: JSON `{"result": {"image": "<base64 encoded image>"}}`
- Method: `get_image_bytes()` decodes base64 from response

### Provider Selection & Failover
- Fixed priority order: Pollinations → Cloudflare
- Attempt providers in order; use first available one
- On failure, try next provider in priority order
- If all fail or none available, return error response

### Timeouts
- Configurable per request via `IMAGE_GENERATION_TIMEOUT_SECONDS` (default 30 seconds)
- Applied at HTTP request level via `requests.post(..., timeout=...)`

## Decisions

### Priority-Based Provider Selection
**Decision**: Use a fixed priority order for provider selection: Pollinations (primary) → Cloudflare (fallback).
- No distributed state or locking required.
- Stateless and simple across all worker processes.
- Both providers available simultaneously if credentials are set.

**Rationale**: Simplicity and resilience; same weighted fairness acceptable for free-tier quotas; easy to reason about across scaled deployments.

### Credential Enablement
**Decision**: Enable providers solely by credential presence—if API key is set and non-empty, provider is available.

**Rationale**: Simpler operational model; no separate enable flags; avoids misconfiguration.

### Credential Initialization
**Decision**: Define provider settings in Django settings with `default=None` (Optional[str]).
- Check `if api_key:` at provider class initialization.
- Skip unavailable providers gracefully in selection logic.

**Rationale**: Simple, consistent with codebase patterns; no startup failures from missing optional credentials.

### Failover with Priority Order
**Decision**: When a request arrives:
1. Check Pollinations credential and attempt generation if available.
2. If Pollinations fails or unavailable, try Cloudflare if credential is present.
3. If both fail or unavailable, return 500 with error details.

**Rationale**: Provides reliable failover without complexity; Pollinations is stable and preferred, Cloudflare as backup.

### Request Timeouts
**Decision**: 30-second default timeout per provider request; configurable per provider via Django settings.

**Rationale**: Prevents hung requests; reasonable for image generation; avoids long frontend waits.

### Error Response Contract (Status Code Only)
**Decision**: Error responses return HTTP status code only; frontend reads status code to determine success/failure.

- **Success (200)**: Returns binary PNG image body (standard multipart/form-data or octet-stream)
- **Validation Error (400)**: Invalid or empty prompt; returns plain text error message (e.g., "Prompt cannot be empty\n")
- **Provider Error (500)**: All providers failed or no providers available; returns plain text error message (e.g., "All image generation providers failed\n")

Frontend behavior:
- Status >= 400 indicates error; body (if any) is for logging only
- Frontend treats error as generic "Image generation failed" message
- Response body parsing optional; status code is authoritative

**Rationale**: Simplify frontend error handling (single status check); HTTP semantics for success/failure; avoid complex response body parsing; enable generic error messages without parsing provider-specific details.
---

## Provider Instantiation & Usage

### Initialization
Providers are instantiated at generation time (not at startup). Each provider reads credentials from Django settings in its `__init__` method:

```python
class PollinationsProvider(ImageProvider):
    def __init__(self):
        self.api_key = settings.POLLINATIONS_API_KEY
        self.available = bool(self.api_key)
    
    def is_available(self) -> bool:
        return self.available

class CloudflareProvider(ImageProvider):
    def __init__(self):
        self.account_id = settings.CLOUDFLARE_ACCOUNT_ID
        self.api_token = settings.CLOUDFLARE_API_TOKEN
        self.model = settings.CLOUDFLARE_MODEL
        self.available = bool(self.account_id and self.api_token)
    
    def is_available(self) -> bool:
        return self.available
```

### Generation Flow
When `/api/gen-image-bytes` receives a request:

1. Validate prompt (return 400 if invalid)
2. Create provider instances: `providers = [PollinationsProvider(), CloudflareProvider()]`
3. For each provider in priority order:
   - Check `provider.is_available()`
   - If available, call `provider.get_image_bytes(prompt)`
   - If successful, return 200 + PNG bytes
   - If fails, catch exception, log error, continue to next
4. If all fail or none available, return 500 + plain text error

**No startup registration or global state required.** Each request independently checks credentials.
