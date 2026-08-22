"""Shared LLM plumbing for the extraction methods.

Reuses the evaluator's provider routing (`shared.evaluator._call_ai`) so extraction
runs on the same model the deployed prototype does. Nothing here loads a .env
automatically, so we load one ourselves.
"""

from __future__ import annotations

import json
from pathlib import Path

_ENV_LOADED = False


def ensure_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    from dotenv import load_dotenv
    # A repo-root .env. This used to also read the prototype's
    # ui_prototype/backend/.env, which no longer exists here — the app lives in
    # its own repository now, and a research run should not depend on a
    # checkout of it being present.
    from shared.paths import REPO_ROOT
    load_dotenv(REPO_ROOT / ".env")
    _ENV_LOADED = True


def call_llm(prompt: str, max_tokens: int = 1500, model: str = None) -> str:
    """One completion. `max_tokens` used to be accepted and silently ignored —
    every call was capped at the evaluator's 1200, which quietly truncated the
    testing step whenever a run carried many principles.
    """
    ensure_env()
    from shared.evaluator import _call_ai
    return _call_ai(prompt, model=model, max_tokens=max_tokens)


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ``` fences / prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)
