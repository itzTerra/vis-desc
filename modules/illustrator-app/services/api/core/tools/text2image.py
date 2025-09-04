from enum import Enum
import requests
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.template.defaultfilters import slugify
import time

class Provider(str, Enum):
    POLLINATIONS = "pollinations"


provider_settings = {
    Provider.POLLINATIONS: {"width": 512, "height": 512, "model": "flux"}
}


def upload_image(request, image: bytes, filename: str) -> str:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestr}/{filename}"
    path = default_storage.save(filename, ContentFile(image))
    image_url = request.build_absolute_uri(default_storage.url(path))
    return image_url


def get_image_bytes(text: str, provider: Provider) -> bytes:
    settings = provider_settings[provider]
    width = settings["width"]
    height = settings["height"]
    model = settings["model"]
    seed = 42

    if provider == Provider.POLLINATIONS:
        url = f"https://pollinations.ai/p/{text}?width={width}&height={height}&seed={seed}&model={model}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image from {url}")
    return response.content


def get_image_url(request, text: str, provider: Provider) -> str:
    content = get_image_bytes(text, provider)
    text_slug = slugify(text)[:10]
    return upload_image(request, content, f"{text_slug}.png")
