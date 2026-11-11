"""Portion 4 — Responder Feedback Validation, Noisy-Label Guard & Unified Queue Helpers."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from db import (
    DEFAULT_DB_PATH,
    count_responder_recent_actions,
    get_active_model_version,
    get_feedback_logs,
    get_feedback_stats,
    get_second_opinion_count,
    log_feedback,
    normalize_damage_class,
)
from feedback import config

_RESPONDER_BASE_RELIABILITY: dict[str, float] = {
    "CMD_LOCAL_01": 1.0,
    "QA_RESP": 0.95,
}


def compute_noisy_label_assessment(
    predicted_label: str,
    corrected_label: str,
    responder_action: str,
    responder_id: str,
) -> dict[str, Any]:
    """Assess severity jump, second-opinion need, and responder reliability weight."""
    action_norm = str(responder_action or "confirm").strip().lower()
    if action_norm != "correct":
        base_rel = _RESPONDER_BASE_RELIABILITY.get(responder_id, 0.9)
        return {
            "severity_jump": 0,
            "requires_second_opinion": False,
            "reliability_weight": base_rel,
        }

    pred_tier, _ = normalize_damage_class(predicted_label)
    corr_tier, _ = normalize_damage_class(corrected_label)
    jump = abs(corr_tier - pred_tier)
    requires_second = jump >= config.SECOND_OPINION_TIER_JUMP
    base_rel = _RESPONDER_BASE_RELIABILITY.get(responder_id, 0.9)
    # Down-weight large jumps until confirmed by a second responder (Portion 4 Plan Sec. 2)
    weight = round(max(0.4, base_rel - (0.15 if requires_second else 0.0)), 3)
    return {
        "severity_jump": jump,
        "requires_second_opinion": requires_second,
        "reliability_weight": weight,
    }


def _parse_bounded_float(raw: Any, field_name: str, default: float, min_val: float, max_val: float | None = None) -> float:
    val_raw = default if raw is None else raw
    if isinstance(val_raw, bool):
        raise ValueError(f"{field_name} cannot be boolean")
    try:
        val = float(val_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(val):
        raise ValueError(f"{field_name} must be finite")
    if val < min_val or (max_val is not None and val > max_val):
        bound_desc = f"[{min_val}, {max_val}]" if max_val is not None else f">= {min_val}"
        raise ValueError(f"{field_name} must be within {bound_desc}")
    return val


def normalize_and_validate_feedback(
    payload: dict[str, Any],
    allow_identical_correction: bool = False,
) -> dict[str, Any]:
    """Normalize canonical damage labels and enforce Portion 4 correction invariants."""
    target_id = str(payload.get("target_id") or "").strip()
    if not target_id:
        raise ValueError("target_id must be non-empty")

    responder_id = str(payload.get("responder_id") or "FIELD_RESPONDER_01").strip()
    if not responder_id:
        raise ValueError("responder_id must be non-empty")

    action_raw = payload.get("responder_action", "confirm")
    action_norm = getattr(action_raw, "value", str(action_raw)).strip().lower().split(".")[-1]
    if action_norm not in {"confirm", "correct", "skip"}:
        raise ValueError(f"Unsupported responder_action: {action_raw!r}")

    pred_raw = str(payload.get("prediction") or "").strip()
    if not pred_raw:
        raise ValueError("prediction must be non-empty")
    _, pred_norm = normalize_damage_class(pred_raw)

    if config.VALID_LABELS and pred_norm not in config.VALID_LABELS:
        raise ValueError(f"prediction must be one of {sorted(config.VALID_LABELS)}")

    corr_raw = str(payload.get("corrected_label") or "").strip()
    corr_norm = ""
    if action_norm == "correct":
        if not corr_raw:
            raise ValueError("corrected_label is required when responder_action is 'correct'")
        _, corr_norm = normalize_damage_class(corr_raw)
        if not allow_identical_correction and corr_norm == pred_norm:
            raise ValueError("corrected_label must differ from predicted label")
        if config.VALID_LABELS and corr_norm not in config.VALID_LABELS:
            raise ValueError(f"corrected_label must be one of {sorted(config.VALID_LABELS)}")
    elif action_norm == "confirm" and corr_raw:
        _, corr_norm = normalize_damage_class(corr_raw)
        if corr_norm != pred_norm:
            raise ValueError("corrected_label cannot contradict prediction when responder_action is 'confirm'")

    uncertainty = _parse_bounded_float(payload.get("uncertainty"), "uncertainty", 0.0, 0.0, 1.0)
    time_taken_sec = _parse_bounded_float(payload.get("time_taken_sec"), "time_taken_sec", 1.5, 0.0)

    noisy = compute_noisy_label_assessment(pred_norm, corr_norm or pred_norm, action_norm, responder_id)
    return {
        **payload,
        "target_id": target_id,
        "responder_id": responder_id,
        "responder_action": action_norm,
        "prediction": pred_norm,
        "corrected_label": corr_norm,
        "uncertainty": uncertainty,
        "time_taken_sec": time_taken_sec,
        "severity_jump": noisy["severity_jump"],
        "requires_second_opinion": noisy["requires_second_opinion"],
        "reliability_weight": noisy["reliability_weight"],
    }


def check_responder_rate_limit(
    responder_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[bool, int]:
    """Return (exceeded_warning, count_last_hour) without blocking active operations."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    recent_count = count_responder_recent_actions(responder_id=responder_id, since_utc=cutoff, db_path=db_path)
    return recent_count >= config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR, recent_count


def record_responder_feedback(
    payload: dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Validate, check responder load, resolve active model/adapter, and persist to SQLite WAL."""
    normalized = normalize_and_validate_feedback(payload)
    rate_warn, recent_count = check_responder_rate_limit(normalized["responder_id"], db_path=db_path)

    active = get_active_model_version(db_path=db_path)
    if active and normalized.get("adapter_version") in (None, "", "none"):
        normalized["model_version"] = active.get("model_version", normalized.get("model_version", "ResNet18-UNet-xBD-v1.0"))
        normalized["adapter_version"] = active.get("adapter_version", "none")

    saved = log_feedback(normalized, db_path=db_path)
    return {
        **saved,
        "prediction": normalized["prediction"],
        "corrected_label": normalized["corrected_label"],
        "uncertainty": normalized["uncertainty"],
        "responder_id": normalized["responder_id"],
        "rate_limited_warning": rate_warn,
        "responder_hourly_count": recent_count + 1,
    }


def get_feedback_queue(
    limit: int = 100,
    target_id: str | None = None,
    action: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Retrieve responder feedback entries and aggregate statistics via db.py."""
    items = get_feedback_logs(limit=limit, target_id=target_id, action=action, db_path=db_path)
    stats = dict(get_feedback_stats(db_path=db_path))
    stats["second_opinion_count"] = get_second_opinion_count(db_path=db_path)
    return {"items": items, "stats": stats}
