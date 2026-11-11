"""Portion 3 (Yousef) — Explainable AI (Grad-CAM, Routing SHAP & GTF LLM Translation)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = ROOT_DIR / "data" / "exports"
DEFAULT_LLM_MODEL_PATH = ROOT_DIR / "data" / "models" / "mistral-7b.gguf"

__all__ = [
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_LLM_MODEL_PATH",
    "generate",
    "gradcam",
    "llm",
    "shap_explainer",
    "validate",
]

_SUBMODULES = {"generate", "gradcam", "llm", "shap_explainer", "validate"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

