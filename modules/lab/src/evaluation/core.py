from __future__ import annotations


from typing import Any


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
        "\\": r"\\textbackslash{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def format_number(val: Any, decimals: int = 4) -> str:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val) if val is not None else ""
    return f"{num:.{decimals}f}"
