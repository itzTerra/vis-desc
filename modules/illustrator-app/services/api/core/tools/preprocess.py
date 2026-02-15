from __future__ import annotations

PREPROCESS_TRANSLATION_TABLE = str.maketrans(
    {
        "\r": None,
        "’": "'",
        "“": '"',
        "”": '"',
        "？": "?",
        "！": "!",
    }
)


def preprocess(text: str) -> str:
    """Normalize text for downstream NLP processing."""
    return text.translate(PREPROCESS_TRANSLATION_TABLE)
