# Tasks

## Phase 1: Add Settings to Django Configuration
**Goal**: Define provider credentials and timeout in Django settings using `env.str()` pattern.

**File**: `services/api/api/settings/base.py`

- [x] Add imports: `from environ import Env; env = Env()`
- [x] Add settings (all optional, default None):
  ```python
  POLLINATIONS_API_KEY = env.str("POLLINATIONS_API_KEY", default=None)
  CLOUDFLARE_ACCOUNT_ID = env.str("CLOUDFLARE_ACCOUNT_ID", default=None)
  CLOUDFLARE_API_TOKEN = env.str("CLOUDFLARE_API_TOKEN", default=None)
  CLOUDFLARE_MODEL = env.str("CLOUDFLARE_MODEL", default="@cf/black-forest-labs/flux-2-klein-4b")
  IMAGE_GENERATION_TIMEOUT_SECONDS = env.int("IMAGE_GENERATION_TIMEOUT_SECONDS", default=30)
  ```
- [x] Update `text2image.py` imports: `from django.conf import settings` (remove any `api.env` references)

**Atomic task**: <100 LoC. Adds settings and removes env module usage.

---

## Phase 2: Create ImageProvider Abstract Base Class
**Goal**: Define provider interface for all image generation providers.

**File**: `services/api/core/tools/text2image.py`

- [x] Create abstract class `ImageProvider`:
  ```python
  from abc import ABC, abstractmethod

  class ImageProvider(ABC):
      @abstractmethod
      def get_image_bytes(self, text: str) -> bytes:
          """Return PNG image bytes from text prompt"""
          pass

      def is_available(self) -> bool:
          """Return True if provider has required credentials configured"""
          pass
  ```
- [x] Add `__init__` method that initializes credentials and sets `self.available` based on credential presence
- [x] Add basic exception handling structure for provider-specific errors

**Atomic task**: <100 LoC. Defines interface contract. ✅ COMPLETE

---

## Phase 3: Create PollinationsProvider Class
**Goal**: Implement Pollinations as a provider class inheriting from `ImageProvider`.

**File**: `services/api/core/tools/text2image.py`

- [ ] Create class `PollinationsProvider(ImageProvider)`:
  - `__init__`: Initialize with credentials from `settings.POLLINATIONS_API_KEY`; set `self.available = bool(api_key)`
  - `get_image_bytes(text)`: Call existing Pollinations API with timeout from settings; return PNG bytes
  - Handle timeouts and API errors; raise `ProviderError(provider_name, error_detail)`
- [ ] Refactor existing `get_image_bytes()` function logic into provider class method
- [ ] Preserve existing URL-building and request logic; only move to class

**Atomic task**: <150 LoC. Wraps existing Pollinations logic in provider class.

---

## Phase 4: Create CloudflareProvider Class
**Goal**: Implement Cloudflare Workers AI as a provider class.

**File**: `services/api/core/tools/text2image.py`

- [ ] Create class `CloudflareProvider(ImageProvider)`:
  - `__init__`: Initialize with credentials from settings (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_MODEL`); set `self.available = bool(all_three_keys)`
  - `get_image_bytes(text)`:
    - POST to `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`
    - Bearer auth header with token
    - JSON body `{"prompt": text}`
    - Extract and decode base64 image: `response["result"]["image"]`
    - Return PNG bytes
  - Handle missing credentials, API errors, malformed responses; raise `ProviderError`
  - Apply timeout from settings
- [ ] Create minimal custom exception: `class ProviderError(Exception): pass`

**Atomic task**: <180 LoC. Full provider implementation with error handling.

---

## Phase 5: Implement Provider Selection & Generation
**Goal**: Attempt providers in fixed priority order; failover on error.

**File**: `services/api/core/tools/text2image.py`

- [ ] Create function `generate_image_bytes(prompt: str) -> bytes`:
  - Initialize providers in priority order: `[PollinationsProvider(), CloudflareProvider()]`
  - Validate prompt: raise ValueError if empty/None
  - Iterate providers; check `is_available()` and attempt `get_image_bytes()`
  - Catch `ProviderError`; log provider name + error; try next
  - Return first success
  - If all fail or none available: raise final exception with all attempted provider names and errors
- [ ] Provider initialization includes credential discovery; no separate `get_available_providers()` needed
- [ ] Log provider name and error on failure for debugging

**Atomic task**: <120 LoC. Selection logic with simple failover.

---

## Phase 6: Update API Endpoints for Status-Code-Only Errors
**Goal**: Return binary PNG on success; 400/500 status code on error (body optional).

**File**: `services/api/core/api.py`

- [ ] Update `/api/gen-image-bytes` endpoint:
  - Validate prompt: if empty/None, return `Response(status=400, content="Prompt cannot be empty")`
  - Call `generate_image_bytes(prompt)` with timeout wrapper
  - On `ValueError` (validation): return 400 + plain text
  - On any provider exception (all failed/none available): return 500 + plain text (e.g., "Image generation failed")
  - On success: return binary PNG with `media_type="image/png"`
- [ ] Update `/api/gen-image` endpoint similarly (return JSON with URL or error status)
- [ ] Remove JSON error body; use status code only for error detection

**Atomic task**: <120 LoC. Endpoint updates with status-code error responses.

---

## Phase 7: Update Environment Templates
**Goal**: Add optional Cloudflare and timeout settings to `.envs` templates.

**Files**: `.envs/.local/.api.template`, `.envs/.production/.api.template`

- [ ] Add (all optional):
  ```
  # Pollinations (optional - set key to enable)
  POLLINATIONS_API_KEY=

  # Cloudflare Workers AI (all required if using - set all three to enable)
  CLOUDFLARE_ACCOUNT_ID=
  CLOUDFLARE_API_TOKEN=
  CLOUDFLARE_MODEL=@cf/black-forest-labs/flux-2-klein-4b

  # Image Generation Timeout (seconds)
  IMAGE_GENERATION_TIMEOUT_SECONDS=30
  ```
- [ ] Update `.envs/.production/.api.template` with guidance comments about setting via secrets engine

**Atomic task**: <50 LoC. Template updates.

---

## Phase 8: Update Frontend Error Handling
**Goal**: Frontend reads HTTP status code; shows generic error message.

**File**: `services/frontend/app/components/ImageEditor.vue`

- [ ] Update image generation handler:
  - Call `/api/gen-image-bytes` as before
  - Check response status on error: if `>= 400`, treat as error
  - Display user message: "Image generation failed. Please try again."
  - Log response body to console (if present) for debugging
  - No special handling needed for 4xx vs 5xx; both show same message
- [ ] Verify blob handling on success (200) unchanged
- [ ] Test that Blob construction still works with successful binary responses

**Atomic task**: <80 LoC. Error handling using status code only.

---

## Phase 9: Integration and Testing
**Goal**: Verify all providers work correctly with failover.

**Tests**:
- [ ] Manual test: Pollinations enabled only → generation succeeds
- [ ] Manual test: Cloudflare enabled only → generation succeeds
- [ ] Manual test: Pollinations fails → tries Cloudflare → succeeds
- [ ] Manual test: Both providers unavailable → returns 500 with plain text error
- [ ] Manual test: Empty prompt → returns 400 with plain text error
- [ ] Manual test: Concurrent requests → all succeed with no race conditions
- [ ] Frontend test: Error message displays on 400/500 response
- [ ] Frontend test: Blob/image displays correctly on 200 response

**Atomic task**: <50 LoC (test code). Validation and edge cases.
