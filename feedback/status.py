"""Portion 4 — Shadow Mode, Safe Model Promotion, Rollback & Adaptation Overview."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from db import (
    DEFAULT_DB_PATH,
    get_active_model_version,
    get_adaptation_runs,
    get_feedback_stats,
    get_second_opinion_count,
    list_model_versions,
    record_adaptation_run,
    record_model_transition,
    register_model_version,
    set_active_model_version,
)
from feedback.drift import compute_drift_report
from feedback.replay import DualReplayBuffer


def promote_or_stage_adapter(
    model_version: str,
    adapter_version: str,
    disaster_id: str,
    weights_path: str,
    gate_report: dict[str, Any],
    promote: bool = True,
    tier: str = "tier_a",
    trigger_reason: str = "drift_or_manual",
    drift_distance: float = 0.0,
    actor_role: str = "PORTION4:ADAPTATION",
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Register candidate adapter in shadow mode, and promote only if all 5 safety checks pass."""
    now_utc = datetime.now(timezone.utc).isoformat()
    gate_passed = bool(gate_report.get("passed", False))

    register_model_version(
        {
            "model_version": model_version,
            "adapter_version": adapter_version,
            "disaster_id": disaster_id,
            "weights_path": weights_path,
            "status": "shadow",
            "metrics": gate_report,
        },
        db_path=db_path,
    )

    promoted = False
    has_weights_ref = bool(str(weights_path or "").strip())
    if promote and gate_passed and has_weights_ref:
        set_active_model_version(model_version, adapter_version, db_path=db_path)
        record_model_transition(
            model_version=model_version,
            adapter_version=adapter_version,
            action="MODEL_PROMOTED",
            gate_checks=gate_report.get("checks", {}),
            actor_role=actor_role,
            db_path=db_path,
        )
        promoted = True
    else:
        record_model_transition(
            model_version=model_version,
            adapter_version=adapter_version,
            action="MODEL_SHADOW_STAGED" if gate_passed else "MODEL_GATE_REJECTED",
            gate_checks=gate_report.get("checks", {}),
            actor_role=actor_role,
            db_path=db_path,
        )

    run_id = f"ADAPT-{int(datetime.now(timezone.utc).timestamp())}-{uuid.uuid4().hex[:6]}-{adapter_version}"
    record_adaptation_run(
        {
            "run_id": run_id,
            "tier": tier,
            "disaster_id": disaster_id,
            "model_version": model_version,
            "adapter_version": adapter_version,
            "trigger_reason": trigger_reason,
            "drift_distance": drift_distance,
            "gate_passed": gate_passed,
            "promoted": promoted,
            "gate_report": gate_report,
            "started_at": now_utc,
            "completed_at": now_utc,
        },
        db_path=db_path,
    )

    return {
        "run_id": run_id,
        "model_version": model_version,
        "adapter_version": adapter_version,
        "gate_passed": gate_passed,
        "promoted": promoted,
        "status": "active" if promoted else ("shadow" if gate_passed else "rejected"),
        "gate_report": gate_report,
    }


def rollback_adapter(
    reason: str = "Operator rollback requested",
    actor_role: str = "COMMANDER:CMD_LOCAL_01",
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """One-click rollback from active adapter to prior archived version or base ('none')."""
    active = get_active_model_version(db_path=db_path)
    versions = list_model_versions(limit=20, db_path=db_path)

    if active is None:
        return {
            "rolled_back": False,
            "previous_adapter_version": "none",
            "active_model_version": "ResNet18-UNet-xBD-v1.0",
            "active_adapter_version": "none",
            "message": "Already running base model weights (no active adapter).",
        }

    prev_adapter = active["adapter_version"]
    model_ver = active["model_version"]
    register_model_version(
        {
            "model_version": model_ver,
            "adapter_version": prev_adapter,
            "disaster_id": active.get("disaster_id", "HOUSTON_HARVEY_2017"),
            "weights_path": active.get("weights_path", ""),
            "status": "rolled_back",
            "metrics": active.get("metrics", {}),
        },
        db_path=db_path,
    )

    fallback = next(
        (v for v in versions if v["status"] == "archived" and v["adapter_version"] != prev_adapter),
        None,
    )
    if fallback is not None:
        set_active_model_version(fallback["model_version"], fallback["adapter_version"], db_path=db_path)
        next_adapter = fallback["adapter_version"]
        next_model = fallback["model_version"]
    else:
        next_adapter = "none"
        next_model = model_ver

    audit = record_model_transition(
        model_version=next_model,
        adapter_version=next_adapter,
        action="MODEL_ROLLBACK",
        gate_checks={"rolled_back_from": prev_adapter, "reason": reason},
        actor_role=actor_role,
        db_path=db_path,
    )

    return {
        "rolled_back": True,
        "previous_adapter_version": prev_adapter,
        "active_model_version": next_model,
        "active_adapter_version": next_adapter,
        "audit_hash": audit.get("curr_hash", ""),
        "message": f"Rolled back adapter {prev_adapter} -> {next_adapter}.",
    }


def get_adaptation_overview(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return active adapter, version history, replay buffer summary, and multi-signal drift."""
    active = get_active_model_version(db_path=db_path)
    stats = get_feedback_stats(db_path=db_path)
    drift_rep = compute_drift_report(feedback_stats=stats, db_path=db_path)
    replay_summary = DualReplayBuffer(db_path=db_path).get_summary()
    runs = get_adaptation_runs(limit=10, db_path=db_path)
    versions = list_model_versions(limit=10, db_path=db_path)

    return {
        "active_model_version": active["model_version"] if active else "ResNet18-UNet-xBD-v1.0",
        "active_adapter_version": active["adapter_version"] if active else "none",
        "retraining_scope": "segmentation_only",
        "routing_policy_coupled": False,
        "second_opinion_count": get_second_opinion_count(db_path=db_path),
        "drift": drift_rep,
        "feedback_stats": stats,
        "replay_summary": replay_summary,
        "registered_versions": versions,
        "recent_runs": runs,
    }
