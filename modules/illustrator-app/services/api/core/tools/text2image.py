from abc import ABC, abstractmethod
from urllib.parse import quote
import base64
import logging
import requests

from django.conf import settings

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when an image provider fails to generate an image.

    Args:
        provider: identifier of provider (e.g. 'pollinations')
        message: human-readable error message
    """

    def __init__(self, provider: str | None = None, message: str | None = None):
        self.provider = provider
        self.message = message or ""
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.provider:
            return f"{self.provider}: {self.message}"
        return self.message


class ImageProvider(ABC):
    """Abstract base class for text-to-image generation providers."""

    def __init__(self):
        self.available = False

    @abstractmethod
    def get_image_bytes(self, text: str) -> bytes:
        """Generate and return PNG image bytes from text prompt."""

    def is_available(self) -> bool:
        """Return True if provider has required credentials configured."""
        return self.available


class PollinationsProvider(ImageProvider):
    """Text-to-image provider using Pollinations API."""

    def __init__(self):
        """Initialize provider with credentials from settings."""
        api_key = settings.POLLINATIONS_API_KEY
        self.api_key = api_key
        self.available = bool(api_key)

    def get_image_bytes(self, text: str) -> bytes:
        """Generate and return PNG image bytes from text using Pollinations API."""
        if not self.available:
            raise ProviderError("pollinations", "API key not configured")

        try:
            width = 512
            height = 512
            model = "flux"
            seed = -1

            url = f"https://gen.pollinations.ai/image/{quote(text)}"
            params = {"width": width, "height": height, "seed": seed, "model": model}
            timeout = settings.IMAGE_GENERATION_TIMEOUT_SECONDS

            response = requests.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout,
            )

            if response.status_code != 200:
                raise ProviderError("pollinations", f"HTTP {response.status_code}")
            return response.content

        except requests.exceptions.Timeout:
            raise ProviderError("pollinations", "Request timeout")
        except requests.exceptions.RequestException as e:
            raise ProviderError("pollinations", f"Request failed: {str(e)}")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError("pollinations", str(e))


class CloudflareProvider(ImageProvider):
    """Text-to-image provider using Cloudflare Workers AI."""

    def __init__(self):
        """Initialize provider with credentials from settings."""
        account_id = settings.CLOUDFLARE_ACCOUNT_ID
        api_token = settings.CLOUDFLARE_API_TOKEN
        model = settings.CLOUDFLARE_MODEL

        self.account_id = account_id
        self.api_token = api_token
        self.model = model
        self.available = bool(account_id and api_token and model)

    def get_image_bytes(self, text: str) -> bytes:
        """Generate and return PNG image bytes from text using Cloudflare Workers AI."""
        if not self.available:
            raise ProviderError(
                "cloudflare", "Account ID, API token, or model not configured"
            )

        try:
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
            # Send as multipart/form-data (Cloudflare expects a multipart body)
            # Use requests' `files` to let it generate the boundary and Content-Type header.
            width = 512
            height = 512
            timeout = settings.IMAGE_GENERATION_TIMEOUT_SECONDS

            files = {
                "prompt": (None, text),
                "width": (None, str(width)),
                "height": (None, str(height)),
            }

            headers = {"Authorization": f"Bearer {self.api_token}"}

            response = requests.post(
                url,
                files=files,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code != 200:
                raise ProviderError(
                    "cloudflare", f"HTTP {response.status_code}: {response.text}"
                )

            # Cloudflare may return JSON (with base64 image) or raw binary image bytes.
            content_type = response.headers.get("content-type", "")

            # If response is JSON, parse and extract base64 image string as before.
            if "application/json" in content_type:
                response_data = response.json()

                if not response_data.get("success"):
                    error_messages = response_data.get("errors", [])
                    raise ProviderError("cloudflare", f"API error: {error_messages}")

                result = response_data.get("result", {})
                image_base64 = result.get("image")

                if not image_base64:
                    raise ProviderError("cloudflare", "No image data in response")

                image_bytes = base64.b64decode(image_base64)
                return image_bytes

            # Otherwise assume binary image data was returned directly.
            return response.content

        except requests.exceptions.Timeout:
            raise ProviderError("cloudflare", "Request timeout")
        except requests.exceptions.RequestException as e:
            raise ProviderError("cloudflare", f"Request failed: {str(e)}")
        except base64.binascii.Error as e:
            raise ProviderError("cloudflare", f"Invalid base64 data: {str(e)}")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError("cloudflare", str(e))


def get_image_providers() -> list[ImageProvider]:
    """Instantiate and return list of available image providers in priority order."""
    provider_classes = {
        "pollinations": PollinationsProvider,
        "cloudflare": CloudflareProvider,
    }
    order_setting = settings.IMAGE_GENERATION_PROVIDERS
    if isinstance(order_setting, (list, tuple)):
        order = [str(p).strip().lower() for p in order_setting if str(p).strip()]
    elif isinstance(order_setting, str) and order_setting.strip():
        order = [p.strip().lower() for p in order_setting.split(",") if p.strip()]
    else:
        # Setting provided but empty/blank -> treat as explicit empty list (no providers)
        order = []

    providers = []
    seen = set()
    for key in order:
        cls = provider_classes.get(key)
        if cls and key not in seen:
            providers.append(cls())
            seen.add(key)

    return providers


def generate_image_bytes(prompt: str) -> bytes:
    """
    Generate image bytes from text prompt using available providers.

    Attempts providers in priority order, failing over to next on error.
    Returns first successful result.

    Parameters:
        prompt (str): Text prompt for image generation.

    Returns:
        bytes: PNG image bytes.

    Raises:
        ValueError: If prompt is empty, None, or whitespace only.
        ProviderError: If all providers fail or none are available.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty or whitespace")

    providers = get_image_providers()
    errors = []

    for provider in providers:
        provider_name = provider.__class__.__name__

        if not provider.is_available():
            logger.debug(f"{provider_name} is not available (missing credentials)")
            continue

        try:
            logger.info(f"Attempting image generation with {provider_name}")
            image_bytes = provider.get_image_bytes(prompt)
            logger.info(f"Successfully generated image with {provider_name}")
            return image_bytes
        except ProviderError as e:
            error_msg = f"{provider_name}: {str(e)}"
            logger.warning(f"Provider failed - {error_msg}")
            errors.append(error_msg)

    if not errors:
        raise ProviderError("all", "No providers are available (missing credentials)")

    error_details = "; ".join(errors)
    raise ProviderError("all", f"All providers failed: {error_details}")


def generate_image_bytes_batch(
    texts: list[str], max_batch_size: int | None = None
) -> list[dict]:
    """
    Generate images for a list of text prompts.

    Returns a list of dicts preserving input order. Each dict is either:
      {"ok": True, "image_b64": "..."}
    or
      {"ok": False, "error": "..."}

    Enforces a server-side max batch size (from settings if not provided).
    """
    if max_batch_size is None:
        max_batch_size = getattr(settings, "IMAGE_GENERATION_MAX_BATCH_SIZE", 50)

    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list of strings")

    if len(texts) > max_batch_size:
        raise ValueError(
            f"Batch size {len(texts)} exceeds maximum allowed {max_batch_size}"
        )

    results: list[dict] = []
    for idx, t in enumerate(texts):
        # Validate element type before calling provider code to avoid
        # AttributeError from non-string inputs and to return clear errors.
        if not isinstance(t, str):
            results.append({"ok": False, "error": "text must be a string"})
            continue

        try:
            image_bytes = generate_image_bytes(t)
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            results.append({"ok": True, "image_b64": image_b64})
        except (ValueError, ProviderError) as e:
            results.append({"ok": False, "error": str(e)})
        except Exception:
            results.append({"ok": False, "error": "generation failed"})

    return results
