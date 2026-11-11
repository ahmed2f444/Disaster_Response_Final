"""Portion 3 (Yousef) — GTF-Constrained Local Mistral-7B Narrative Translator.

Restates pre-computed SHAP (routes) and Grad-CAM (buildings) facts in plain English
using `data/models/mistral-7b.gguf` via `llama-cpp-python` when available and enabled.
Never computes routes, never estimates its own confidence, and cleanly yields `""`
when disabled or unavailable so deterministic templates cover synchronous/edge paths.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

from explain.validate import coerce_shap_dict

try:
    from llama_cpp import Llama

    _LLAMA_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    _LLAMA_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LLM_MODEL_PATH = ROOT_DIR / "data" / "models" / "mistral-7b.gguf"

_LLM_LOCK = threading.Lock()
_LLM_CALL_LOCK = threading.Lock()
_LLM_INSTANCE: Any = None
_LLM_LOAD_FAILED = False


def is_llm_available(model_path: str | Path | None = None) -> bool:
    """Return True if `llama_cpp` is installed and the GGUF model checkpoint exists."""
    path = Path(model_path or os.getenv("AEGIS_LLM_MODEL_PATH", str(DEFAULT_LLM_MODEL_PATH)))
    return bool(_LLAMA_AVAILABLE and path.is_file())


def is_llm_enabled(model_path: str | Path | None = None) -> bool:
    """Return True when local LLM inference is both available and enabled by runtime config."""
    raw = os.getenv("AEGIS_LLM_ENABLED")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"} and is_llm_available(model_path)
    # Default off inside automated pytest runs unless explicitly enabled, and on for live runtime
    if "pytest" in sys.modules:
        return False
    return is_llm_available(model_path)


def get_llm(model_path: str | Path | None = None) -> Any | None:
    """Thread-safe lazy singleton loader for local Mistral-7B with safe VRAM/CPU bounds."""
    global _LLM_INSTANCE, _LLM_LOAD_FAILED
    if not _LLAMA_AVAILABLE or _LLM_LOAD_FAILED:
        return None
    target_path = Path(model_path or os.getenv("AEGIS_LLM_MODEL_PATH", str(DEFAULT_LLM_MODEL_PATH)))
    if not target_path.is_file():
        return None

    with _LLM_LOCK:
        if _LLM_INSTANCE is not None:
            return _LLM_INSTANCE
        if _LLM_LOAD_FAILED:
            return None
        try:
            gpu_layers = int(os.getenv("AEGIS_LLM_GPU_LAYERS", "0"))
            n_ctx = int(os.getenv("AEGIS_LLM_CTX", "1024"))
            n_threads = int(os.getenv("AEGIS_LLM_THREADS", str(max(1, min(8, (os.cpu_count() or 4) // 2)))))
            _LLM_INSTANCE = Llama(
                model_path=str(target_path),
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mmap=True,
                verbose=False,
            )
            return _LLM_INSTANCE
        except Exception as exc:
            _LOGGER.warning("Failed to initialize local Mistral GGUF model (%s): %s", target_path, exc)
            _LLM_LOAD_FAILED = True
            return None


def build_route_prompt(
    shap_values: dict[str, Any] | list[dict[str, Any]],
    mask_altered: bool,
    mask_reason: str,
    uncertainty: float,
    route: dict[str, Any] | None = None,
) -> str:
    """Construct Ground-Truth-First prompt for route SHAP narration (Stage 6 & Sec. 7)."""
    shap_dict = coerce_shap_dict(shap_values)
    top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    features_text = ", ".join(f"{name} ({val:.2f})" for name, val in top_features) or "none"

    route_id = str((route or {}).get("route_id") or (route or {}).get("plan_id") or "PLAN")
    eta = float((route or {}).get("eta_seconds") or (route or {}).get("whatif_eta_sec") or (route or {}).get("total_eta_sec") or 0.0)
    edge_ids: list[str] = []
    for sub in (route or {}).get("routes", []) or []:
        if isinstance(sub, dict):
            edge_ids.extend(str(e) for e in sub.get("edge_ids", []))
    unique_edges = ", ".join(list(dict.fromkeys(edge_ids))[:8]) or "corridor"

    mask_text = f"Yes ({mask_reason})" if mask_altered else "No (primary PPO proposal verified clear)"
    return (
        "[INST] You are a disaster response translator. Restate ONLY the verified facts below in 2 plain-English sentences.\n"
        "STRICT RULES:\n"
        "- Do NOT add any number not listed below.\n"
        "- Do NOT use percentage symbols (%) or words like confidence, certainty, confident, very, highly, definitely, maybe, likely, probably.\n"
        "- Do NOT suggest overriding the PPO decision or safety mask.\n"
        "FACTS:\n"
        f"- Route ID: {route_id}\n"
        f"- Computed ETA: {eta:.1f}s\n"
        f"- Corridors: {unique_edges}\n"
        f"- Top SHAP factors (pre-mask PPO): {features_text}\n"
        f"- Safety mask altered route: {mask_text}\n"
        f"- System uncertainty: {uncertainty:.2f}\n"
        "Restate only these facts. [/INST]"
    )


def build_building_prompt(
    building: dict[str, Any],
    gradcam_info: dict[str, Any] | None = None,
) -> str:
    """Construct Ground-Truth-First prompt for building Grad-CAM narration (Stage 3 & Sec. 7)."""
    bld_id = str(building.get("building_id") or "BLD")
    dmg_cls = str(building.get("damage_class", building.get("class_label", "minor"))).lower()
    conf = float(building.get("confidence", 0.85))
    uncertainty = round(max(0.0, min(1.0, 1.0 - conf)), 2)

    info = gradcam_info or {}
    hotspot = str(info.get("hotspot_summary") or "Decoder Grad-CAM highlights structural regions within the building footprint.")
    peak = float(info.get("peak_saliency", 0.0))

    return (
        "[INST] You are a disaster response translator. Restate ONLY the verified facts below in 1-2 plain-English sentences.\n"
        "STRICT RULES:\n"
        "- State the exact damage class listed below and do NOT mention other damage classes.\n"
        "- Do NOT use percentage symbols (%) or words like confidence, certainty, confident, very, highly, definitely, maybe, likely.\n"
        "- Do NOT invent numbers.\n"
        "FACTS:\n"
        f"- Building ID: {bld_id}\n"
        f"- Damage class: {dmg_cls}\n"
        f"- Grad-CAM focus: {hotspot}\n"
        f"- Peak saliency weight: {peak:.2f}\n"
        f"- System uncertainty: {uncertainty:.2f}\n"
        "Restate only these facts. [/INST]"
    )


def _invoke_llm(llm: Any, prompt: str, max_tokens: int = 140) -> str:
    with _LLM_CALL_LOCK:
        output = llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["[/INST]"])
    if isinstance(output, dict):
        choices = output.get("choices") or []
        if choices and isinstance(choices[0], dict):
            return str(choices[0].get("text") or "").strip()
    return str(output or "").strip()


def generate_explanation(
    shap_values: dict[str, Any] | list[dict[str, Any]],
    mask_altered: bool,
    mask_reason: str,
    uncertainty: float,
    *,
    route: dict[str, Any] | None = None,
    llm_instance: Any | None = None,
    max_tokens: int = 140,
) -> str:
    """Restate route SHAP facts using Mistral-7B; returns `""` if model is unavailable."""
    llm = llm_instance if llm_instance is not None else get_llm()
    if llm is None:
        return ""
    prompt = build_route_prompt(
        shap_values=shap_values,
        mask_altered=mask_altered,
        mask_reason=mask_reason,
        uncertainty=uncertainty,
        route=route,
    )
    try:
        return _invoke_llm(llm, prompt, max_tokens=max_tokens)
    except Exception as exc:
        _LOGGER.warning("LLM route generation failed cleanly: %s", exc)
        return ""


def generate_building_explanation(
    building: dict[str, Any],
    gradcam_info: dict[str, Any] | None = None,
    *,
    llm_instance: Any | None = None,
    max_tokens: int = 120,
) -> str:
    """Restate building Grad-CAM facts using Mistral-7B; returns `""` if unavailable."""
    llm = llm_instance if llm_instance is not None else get_llm()
    if llm is None:
        return ""
    prompt = build_building_prompt(building=building, gradcam_info=gradcam_info)
    try:
        return _invoke_llm(llm, prompt, max_tokens=max_tokens)
    except Exception as exc:
        _LOGGER.warning("LLM building generation failed cleanly: %s", exc)
        return ""
