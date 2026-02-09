from abc import ABC, abstractmethod
from enum import Enum
import requests
from urllib.parse import quote
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.template.defaultfilters import slugify
import time

from django.conf import settings


class ProviderError(Exception):
    """Raised when an image provider fails to generate an image."""

    pass


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


class Provider(str, Enum):
    POLLINATIONS = "pollinations"


PROVIDER_SETTINGS = {
    Provider.POLLINATIONS: {
        "width": 512,
        "height": 512,
        "model": "flux",
        "api_key": settings.POLLINATIONS_API_KEY,
        "seed": -1,  # Random seed
    }
}


def upload_image(request, image: bytes, filename: str) -> str:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestr}/{filename}"
    path = default_storage.save(filename, ContentFile(image))
    image_url = request.build_absolute_uri(default_storage.url(path))
    return image_url


def get_image_bytes(text: str, provider: Provider) -> bytes:
    provider_config = PROVIDER_SETTINGS[provider]

    match provider:
        case Provider.POLLINATIONS:
            width = provider_config["width"]
            height = provider_config["height"]
            model = provider_config["model"]
            seed = provider_config["seed"]
            api_key = provider_config["api_key"]

            url = f"https://gen.pollinations.ai/image/{quote(text)}"
            params = {"width": width, "height": height, "seed": seed, "model": model}
            response = requests.get(
                url, params=params, headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code != 200:
                raise Exception(f"Failed to fetch image from {url}: {response.text}")
            return response.content
        case _:
            raise Exception(f"Unsupported provider: {provider}")

    raise Exception("There was an error generating the image.")


def get_image_url(request, text: str, provider: Provider) -> str:
    content = get_image_bytes(text, provider)
    text_slug = slugify(text)[:10]
    return upload_image(request, content, f"{text_slug}.png")
