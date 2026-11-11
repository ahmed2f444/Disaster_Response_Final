"""Portion 4 — Uncertainty + Diversity Active Learning Label Selector.

Implements non-blocking responder query selection (Portion 4 Plan Sec. 2):
- Ranks buildings by Bayesian uncertainty (1.0 - confidence)
- Combines spatial/embedding k-center diversity so batches avoid near-duplicates
- Deduplicates neighboring footprints within MIN_NEIGHBOR_SEPARATION_DEG
- Prioritizes buildings needing second opinion (>=2 damage-tier jump)
- Reports responder hourly workload caps without blocking operations
"""

from __future__ import annotations

import math
from typing import Any

from db import DEFAULT_DB_PATH, get_buildings, get_feedback_logs, normalize_damage_class
from feedback import config
from feedback.feedback import check_responder_rate_limit


def _safe_float(raw: Any, default: float, lo: float | None = None, hi: float | None = None) -> float:
    if raw is None or isinstance(raw, bool):
        val = default
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = default
        if not math.isfinite(val):
            val = default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def _deg_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(
        float(a.get("centroid_lat", 0.0)) - float(b.get("centroid_lat", 0.0)),
        float(a.get("centroid_lon", 0.0)) - float(b.get("centroid_lon", 0.0)),
    )


def _min_distance_to_selected(cand: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 1.0
    return min(_deg_distance(cand, item) for item in selected)


def select_labeling_candidates(
    budget: int = 12,
    responder_id: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    min_neighbor_deg: float | None = None,
) -> dict[str, Any]:
    """Select an uncertainty + diversity batch of buildings for 1-tap responder review."""
    max_per_class = max(1, int(config.SELECTION_MAX_PER_CLASS))
    effective_budget = max(1, min(int(budget), max_per_class * 4))
    neighbor_tol = max(0.0, float(config.MIN_NEIGHBOR_SEPARATION_DEG if min_neighbor_deg is None else min_neighbor_deg))

    buildings = get_buildings(db_path=db_path)
    # get_feedback_logs returns newest-first; scan reversed (chronological) so confirmations clear prior flags
    logs = get_feedback_logs(limit=5000, db_path=db_path)

    verified_targets: set[str] = set()
    pending_second_opinion: set[str] = set()

    for row in reversed(logs):
        tid = str(row.get("target_id") or "")
        act = str(row.get("responder_action") or "").lower()
        if not tid:
            continue
        if int(row.get("requires_second_opinion") or 0) == 1:
            pending_second_opinion.add(tid)
            verified_targets.discard(tid)
        elif act in {"confirm", "correct"}:
            pending_second_opinion.discard(tid)
            verified_targets.add(tid)

    # Exclude cleanly verified buildings unless they still owe a second opinion
    pool: list[dict[str, Any]] = []
    for b in buildings:
        bid = str(b["building_id"])
        needs_second = bid in pending_second_opinion
        if bid in verified_targets and not needs_second:
            continue
        try:
            _, cls_norm = normalize_damage_class(b.get("damage_class", "no_damage") or "no_damage")
        except ValueError:
            cls_norm = "no_damage"
        conf = _safe_float(b.get("confidence"), 0.9, 0.0, 1.0)
        unc = round(1.0 - conf, 4)
        pool.append(
            {
                "building_id": bid,
                "damage_class": cls_norm,
                "confidence": round(conf, 4),
                "uncertainty": unc,
                "centroid_lat": _safe_float(b.get("centroid_lat"), 0.0),
                "centroid_lon": _safe_float(b.get("centroid_lon"), 0.0),
                "status_flag": b.get("status_flag", "CALIBRATED"),
                "embedding_ref": b.get("embedding_ref", ""),
                "requires_second_opinion": needs_second,
            }
        )

    # Greedy uncertainty + k-center diversity with spatial neighbor deduplication & class caps
    selected: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    remaining = list(pool)
    while remaining and len(selected) < effective_budget:
        best_idx = -1
        best_score = -1.0
        for idx, cand in enumerate(remaining):
            cls_name = cand["damage_class"]
            if class_counts.get(cls_name, 0) >= max_per_class:
                continue
            min_dist = _min_distance_to_selected(cand, selected)
            if selected and min_dist < neighbor_tol:
                continue
            div_norm = min(1.0, min_dist / 0.05) if selected else 1.0
            second_bonus = 0.25 if cand["requires_second_opinion"] else 0.0
            score = 0.65 * cand["uncertainty"] + 0.35 * div_norm + second_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        chosen = dict(remaining.pop(best_idx))
        chosen["acquisition_score"] = round(best_score, 4)
        class_counts[chosen["damage_class"]] = class_counts.get(chosen["damage_class"], 0) + 1
        selected.append(chosen)

    rate_warn, hourly_count = (False, 0)
    if responder_id:
        rate_warn, hourly_count = check_responder_rate_limit(responder_id, db_path=db_path)

    return {
        "candidates": selected,
        "pool_size": len(pool),
        "budget": effective_budget,
        "responder_rate_warning": rate_warn,
        "responder_hourly_count": hourly_count,
    }

