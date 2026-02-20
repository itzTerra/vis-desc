"""LLM integration utilities for text enhancement."""

import requests
from django.conf import settings
import time


LLM_MODEL_NAME = "gpt-oss-120b"

SYSTEM_PROMPT = """You are an expert at extracting visual descriptions from text for generating images.

Your task is to analyze the given text segment and extract ONLY the visual descriptions that would be useful for a text-to-image generation model. Follow these rules strictly:

1. Extract only visual descriptions (objects, people, scenery, colors, textures, compositions, etc.)
2. Keep as much detail as possible while being concise
3. Do NOT add any meta keywords, hashtags, or creative additions
4. Do NOT add narrative elements, emotions, or abstract concepts
5. Do NOT add quality instructions (like "high quality", "photorealistic", etc.)
6. Only use information that is explicitly in the original text
7. Preserve specific details like numbers, colors, materials, actions, and positions

Output only the extracted visual description, nothing else.

Examples:

Input: "The old lighthouse stood on the rocky cliff, its white paint peeling in long strips, while the grey stone tower below remained proud and sturdy. Waves crashed violently against the jagged rocks at its base."
Output: "Old white lighthouse with peeling paint on a rocky cliff, grey stone tower, white paint peeling in strips, jagged rocks below, waves crashing against rocks"

Input: "She said, 'Do you see it now?' as the narrow alley opened into a quiet courtyard. Tall, weathered brick walls rose on either side, their surfaces broken by rusted fire escapes and a few cracked windows. A thin ribbon of late afternoon light cut across the cobblestones, and a lone bicycle leaned against a dented metal gate painted a faded green."
Output: "Quiet courtyard with tall weathered brick walls, rusted fire escapes, cracked windows, cobblestone ground, thin ribbon of late afternoon light, lone bicycle leaning against a dented faded green metal gate"
"""


def enhance_text_with_llm(text: str) -> str:
    """
    Enhance text by extracting visual descriptions using an LLM.

    Args:
        text: The input text segment to enhance

    Returns:
        Enhanced text containing visual descriptions suitable for text-to-image generation

    Raises:
        ValueError: If LLM API configuration is missing
        requests.RequestException: If the API request fails
    """
    DEFAULT_PROMPT_KEYWORDS = "concept art, highly detailed, 4K, UHD, cinematic lighting, vivid, vibrant, artstation"
    api_key = settings.EINFRA_API_KEY
    base_url = settings.EINFRA_BASE_URL

    if not api_key or not base_url:
        raise ValueError(
            "EINFRA_API_KEY and EINFRA_BASE_URL must be configured in environment variables"
        )

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    max_retries = settings.LLM_API_MAX_RETRIES
    delay = settings.LLM_API_RETRY_DELAY_SECONDS
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            if not content or not isinstance(content, str) or content.strip() == "":
                raise ValueError("LLM API response is missing content")
            enhanced_text = content.strip()
            return f"{enhanced_text}, {DEFAULT_PROMPT_KEYWORDS}"
        except Exception:
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise
