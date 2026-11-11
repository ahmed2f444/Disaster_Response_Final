"""Portion 4 — Shadow Mode, Safe Model Promotion, Rollback & Adaptation Overview."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db import (
    DEFAULT_DB_PATH,
    append_audit_event,
    get_active_model_version,
    get_adaptation_runs,
    get_buildings,
    get_feedback_logs,
    get_feedback_stats,
    get_second_opinion_count,
    list_model_versions,
    record_adaptation_run,
    record_model_transition,
    register_model_version,
    set_active_model_version,
)
from feedback import config
from feedback.drift import compute_drift_report
from feedback.replay import DualReplayBuffer
from feedback.selection import select_labeling_candidates

_LOGGER = logging.getLogger(__name__)
_DRIFT_TRIGGER_LOCK = threading.Lock()
_DRIFT_IN_FLIGHT: set[str] = set()
_DRIFT_LAST_FB_COUNT: dict[str, int] = {}
_DRIFT_THREADS: list[threading.Thread] = []


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


def join_drift_candidate_threads(timeout: float = 60.0) -> None:
    """Wait for any active background drift-triggered candidate training threads to complete."""
    with _DRIFT_TRIGGER_LOCK:
        pending = [t for t in _DRIFT_THREADS if t.is_alive()]
    for th in pending:
        th.join(timeout=timeout)


def _count_verified_building_labels(db_path: str) -> int:
    bld_ids = {str(b["building_id"]) for b in get_buildings(db_path=db_path)}
    verified: set[str] = set()
    for row in get_feedback_logs(limit=5000, db_path=db_path):
        tid = str(row.get("target_id") or "")
        act = str(row.get("responder_action") or "").lower()
        if tid in bld_ids and act in {"confirm", "correct"}:
            verified.add(tid)
    return len(verified)


def handle_drift_candidate_trigger(
    db_path: str = DEFAULT_DB_PATH,
    *,
    drift_distance: float | None = None,
    feedback_stats: dict[str, Any] | None = None,
    disaster_id: str = "HOUSTON_HARVEY_2017",
    spawn_training: bool = True,
    async_training: bool = True,
    adapter_root: str | Path | None = None,
    force_trigger: bool = False,
) -> dict[str, Any]:
    """Step-1 drift trigger: queue 1-tap labels and start candidate training (`promote=False`) only.

    Guarantees:
    - Never promotes or replaces the live segmentation model (`promote=False` always).
    - Promotion stays gated behind `evaluate_safety_gate` (5 checks -> shadow -> gated promotion)
      and logged in the SHA-256 audit chain.
    - Routing / MAPPO weights and road graphs are never touched (`routing_policy_coupled=False`).
    """
    stats = feedback_stats if feedback_stats is not None else get_feedback_stats(db_path=db_path)
    total_fb = int(stats.get("total_count") or 0)
    if drift_distance is None:
        drift_rep = compute_drift_report(feedback_stats=stats, db_path=db_path)
        drift_val = float(drift_rep["drift_distance"])
    else:
        drift_val = float(drift_distance)

    drift_detected = drift_val >= float(config.DRIFT_THRESHOLD)
    pending_trigger = drift_detected or total_fb >= 50
    if not pending_trigger:
        return {
            "triggered": False,
            "drift_detected": False,
            "candidate_started": False,
            "promoted": False,
            "auto_promote": False,
            "routing_policy_coupled": False,
            "queued_label_count": 0,
            "queued_label_ids": [],
            "status": "idle",
        }

    db_key = str(Path(db_path).resolve())
    with _DRIFT_TRIGGER_LOCK:
        if db_key in _DRIFT_IN_FLIGHT or (not force_trigger and _DRIFT_LAST_FB_COUNT.get(db_key) == total_fb):
            return {
                "triggered": False,
                "drift_detected": drift_detected,
                "candidate_started": False,
                "promoted": False,
                "auto_promote": False,
                "routing_policy_coupled": False,
                "queued_label_count": 0,
                "queued_label_ids": [],
                "status": "already_triggered",
            }
        _DRIFT_LAST_FB_COUNT[db_key] = total_fb

    queue_res = select_labeling_candidates(budget=12, db_path=db_path)
    queued_ids = [str(c["building_id"]) for c in (queue_res.get("candidates") or [])]
    verified_cnt = _count_verified_building_labels(db_path)

    if not spawn_training or verified_cnt < 2:
        append_audit_event(
            actor_role="PORTION4:DRIFT_TRIGGER",
            event_type="DRIFT_LABEL_QUEUE_TRIGGERED",
            payload={
                "drift_distance": round(drift_val, 4),
                "total_feedback_samples": total_fb,
                "verified_label_count": verified_cnt,
                "queued_label_count": len(queued_ids),
                "queued_label_ids": queued_ids[:8],
                "auto_promote": False,
                "retraining_scope": "segmentation_only",
                "routing_policy_coupled": False,
            },
            db_path=db_path,
        )
        return {
            "triggered": True,
            "drift_detected": drift_detected,
            "candidate_started": False,
            "promoted": False,
            "auto_promote": False,
            "routing_policy_coupled": False,
            "queued_label_count": len(queued_ids),
            "queued_label_ids": queued_ids,
            "status": "queued_labels",
        }

    append_audit_event(
        actor_role="PORTION4:DRIFT_TRIGGER",
        event_type="DRIFT_CANDIDATE_TRIGGERED",
        payload={
            "drift_distance": round(drift_val, 4),
            "total_feedback_samples": total_fb,
            "verified_label_count": verified_cnt,
            "queued_label_count": len(queued_ids),
            "auto_promote": False,
            "retraining_scope": "segmentation_only",
            "routing_policy_coupled": False,
        },
        db_path=db_path,
    )

    def _run_candidate() -> dict[str, Any] | None:
        try:
            from feedback.adapt import run_feedback_driven_tier_a

            return run_feedback_driven_tier_a(
                disaster_id=disaster_id,
                epochs=2,
                promote=False,  # Safety-gate invariant: drift trigger NEVER auto-promotes
                tier="tier_a",
                trigger_reason="auto_drift_candidate",
                db_path=db_path,
                adapter_root=adapter_root,
            )
        except Exception as exc:
            _LOGGER.warning("Drift candidate adaptation failed cleanly for %s: %s", db_key, exc)
            return None
        finally:
            with _DRIFT_TRIGGER_LOCK:
                _DRIFT_IN_FLIGHT.discard(db_key)

    if async_training:
        with _DRIFT_TRIGGER_LOCK:
            _DRIFT_IN_FLIGHT.add(db_key)
            _DRIFT_THREADS[:] = [t for t in _DRIFT_THREADS if t.is_alive()]
            th = threading.Thread(target=_run_candidate, name="p4-drift-candidate", daemon=True)
            _DRIFT_THREADS.append(th)
        th.start()
        return {
            "triggered": True,
            "drift_detected": drift_detected,
            "candidate_started": True,
            "promoted": False,
            "auto_promote": False,
            "routing_policy_coupled": False,
            "queued_label_count": len(queued_ids),
            "queued_label_ids": queued_ids,
            "status": "candidate_training_started",
        }

    with _DRIFT_TRIGGER_LOCK:
        _DRIFT_IN_FLIGHT.add(db_key)
    cand_res = _run_candidate()
    return {
        "triggered": True,
        "drift_detected": drift_detected,
        "candidate_started": cand_res is not None,
        "promoted": False,
        "auto_promote": False,
        "routing_policy_coupled": False,
        "queued_label_count": len(queued_ids),
        "queued_label_ids": queued_ids,
        "status": (cand_res or {}).get("status", "candidate_failed"),
    }
