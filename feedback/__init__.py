"""Portion 4 (Abdulrahman) — Active Learning Feedback, Drift Monitor & Continual Adaptation."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "adapt",
    "calibrate",
    "config",
    "drift",
    "feedback",
    "gate",
    "replay",
    "selection",
    "status",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
