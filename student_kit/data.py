from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if limit is not None and len(items) >= limit:
                    break
    return items


def get_role_content(messages: list[dict[str, str]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return message.get("content", "")
    return ""


def build_prompt(messages: list[dict[str, str]]) -> str:
    system = get_role_content(messages, "system").strip()
    user = get_role_content(messages, "user").strip()
    return (
        f"{system}\n\n"
        f"Logo description:\n{user}\n\n"
        "Output exactly one complete SVG document. No markdown, no prose.\n"
        "SVG:\n"
    )


def get_target_svg(messages: list[dict[str, str]]) -> str:
    return get_role_content(messages, "assistant").strip()

