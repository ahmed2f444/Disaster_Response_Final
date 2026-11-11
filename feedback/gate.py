"""Portion 4 — Five-Check Adaptation Safety Gate & PyTorch Validation Evaluator.

Enforces Unified System Document Stage 10 & Portion 4 Plan Sec. 5:
1. Per-class F1 check (with "destroyed" class reported separately)
2. Forgetting check on old-disaster benchmark (<= 2.0 F1 points drop)
3. Geographic holdout validation (non-overlapping spatial regions)
4. Uncertainty calibration preservation (ECE delta check)
5. Edge inference latency check against 30-minute full-area budget

Fails CLOSED if required validation checks/benchmarks are absent or violated.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np
import torch

from damage.model import IGNORE_INDEX, NUM_CLASSES
from feedback import config

_CLASS_NAMES = {1: "no_damage", 2: "minor", 3: "major", 4: "destroyed"}


def _finite_or_none(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def compute_pixel_metrics(
    model: torch.nn.Module,
    samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate real PyTorch model per-class F1, ECE, and mean tile latency."""
    if not samples:
        raise ValueError("Evaluation requires at least one (image_6ch, target_mask) sample")

    bins = max(1, int(n_bins))
    model.eval()
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    all_confs: list[np.ndarray] = []
    all_correct: list[np.ndarray] = []
    latencies_ms: list[float] = []

    with torch.no_grad():
        for x_item, y_item in samples:
            xb = x_item.unsqueeze(0).to(device) if x_item.ndim == 3 else x_item.to(device)
            yb = y_item.to(device)
            t0 = time.perf_counter()
            logits = model(xb)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            probs = torch.softmax(logits.float(), dim=1)[0]
            conf_map, pred_map = probs.max(dim=0)
            valid = yb != IGNORE_INDEX
            if not torch.any(valid):
                continue

            pred_v = pred_map[valid].cpu().numpy()
            true_v = yb[valid].cpu().numpy()
            conf_v = conf_map[valid].cpu().numpy()

            all_confs.append(conf_v)
            all_correct.append((pred_v == true_v).astype(np.float32))
            for c in range(NUM_CLASSES):
                p_c = pred_v == c
                t_c = true_v == c
                tp[c] += int(np.sum(p_c & t_c))
                fp[c] += int(np.sum(p_c & (~t_c)))
                fn[c] += int(np.sum((~p_c) & t_c))

    bg_denom = (2 * tp[0]) + fp[0] + fn[0]
    background_f1 = round(float((2 * tp[0]) / bg_denom), 4) if bg_denom > 0 else None

    per_class_f1: dict[str, float] = {}
    present_f1s: list[float] = []
    for c, name in _CLASS_NAMES.items():
        denom = (2 * tp[c]) + fp[c] + fn[c]
        if denom > 0:
            f1 = float((2 * tp[c]) / denom)
            per_class_f1[name] = round(f1, 4)
            present_f1s.append(f1)
        else:
            per_class_f1[name] = 0.0

    macro_f1 = round(float(np.mean(present_f1s)) if present_f1s else 0.0, 4)
    ece = 0.0
    if all_confs:
        confs = np.concatenate(all_confs)
        corr = np.concatenate(all_correct)
        edges = np.linspace(0.0, 1.0, bins + 1)
        for idx in range(bins):
            mask = (confs >= edges[idx]) & (confs <= edges[idx + 1] if idx == bins - 1 else confs < edges[idx + 1])
            if np.any(mask):
                ece += (float(np.sum(mask)) / len(confs)) * abs(float(np.mean(corr[mask])) - float(np.mean(confs[mask])))

    return {
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "destroyed_f1": per_class_f1["destroyed"],
        "background_f1": background_f1,
        "ece": round(float(ece), 4),
        "tile_latency_ms": round(float(np.mean(latencies_ms)), 3),
    }


def evaluate_safety_gate(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    max_forgetting_drop: float | None = None,
    max_ece_delta: float | None = None,
    max_tile_ms: float | None = None,
) -> dict[str, Any]:
    """Run all five Stage 10 safety gate checks and fail closed on missing/failing checks."""
    drop_tol = max(0.0, float(config.MAX_FORGETTING_F1_DROP if max_forgetting_drop is None else max_forgetting_drop))
    ece_tol = max(0.0, float(config.MAX_CALIBRATION_ECE_DELTA if max_ece_delta is None else max_ece_delta))
    tile_tol = max(1.0, float(config.MAX_TILE_INFERENCE_MS if max_tile_ms is None else max_tile_ms))

    base_f1_map = baseline_metrics.get("per_class_f1") or {}
    cand_f1_map = candidate_metrics.get("per_class_f1") or {}
    cand_dest = _finite_or_none(cand_f1_map.get("destroyed"))
    if not cand_f1_map or cand_dest is None:
        c1: dict[str, Any] = {"passed": False, "reason": "Missing finite candidate per_class_f1 or destroyed class metric"}
    else:
        missing_cls = [k for k in base_f1_map if k not in cand_f1_map]
        bad_cls = [k for k, v in cand_f1_map.items() if _finite_or_none(v) is None]
        if missing_cls or bad_cls:
            c1 = {
                "passed": False,
                "reason": f"Incomplete/non-finite per_class_f1 classes: missing={missing_cls}, invalid={bad_cls}",
            }
        else:
            base_dest = _finite_or_none(base_f1_map.get("destroyed", 0.0)) or 0.0
            base_vals = [_finite_or_none(v) or 0.0 for v in base_f1_map.values()] if base_f1_map else [0.0]
            cand_vals = [_finite_or_none(v) or 0.0 for v in cand_f1_map.values()]
            base_macro = float(np.mean(base_vals))
            cand_macro = float(np.mean(cand_vals))
            base_bg = _finite_or_none(baseline_metrics.get("background_f1"))
            cand_bg = _finite_or_none(candidate_metrics.get("background_f1"))
            bg_ok = True
            if base_bg is not None and cand_bg is not None:
                bg_ok = cand_bg >= max(0.50, base_bg - 0.10)
            c1_ok = (cand_dest >= base_dest) and (cand_macro >= base_macro) and bg_ok
            c1 = {
                "passed": c1_ok,
                "baseline_macro_f1": round(base_macro, 4),
                "candidate_macro_f1": round(cand_macro, 4),
                "baseline_destroyed_f1": round(base_dest, 4),
                "candidate_destroyed_f1": round(cand_dest, 4),
            }
            if base_bg is not None and cand_bg is not None:
                c1["baseline_background_f1"] = round(base_bg, 4)
                c1["candidate_background_f1"] = round(cand_bg, 4)

    # Check 2: Old-disaster forgetting tolerance (<= 2 F1 points)
    base_old = _finite_or_none(baseline_metrics.get("old_disaster_macro_f1"))
    cand_old = _finite_or_none(candidate_metrics.get("old_disaster_macro_f1"))
    if base_old is None or cand_old is None:
        c2 = {"passed": False, "reason": "Missing finite old-disaster benchmark F1 for forgetting check"}
    else:
        forgetting_drop = round(base_old - cand_old, 4)
        c2 = {
            "passed": forgetting_drop <= drop_tol,
            "forgetting_drop": forgetting_drop,
            "tolerance": drop_tol,
        }

    # Check 3: Non-overlapping geographic holdout validation
    hold_cand = _finite_or_none(candidate_metrics.get("geographic_holdout_f1"))
    hold_base = _finite_or_none(candidate_metrics.get("geographic_holdout_baseline_f1"))
    train_ids = set(str(x) for x in (candidate_metrics.get("train_region_ids") or candidate_metrics.get("train_building_ids") or []))
    hold_ids = set(str(x) for x in (candidate_metrics.get("holdout_region_ids") or candidate_metrics.get("holdout_building_ids") or []))
    has_overlap = bool(candidate_metrics.get("geographic_holdout_overlap", False)) or bool(train_ids & hold_ids)
    if hold_cand is None or hold_base is None:
        c3 = {"passed": False, "reason": "Missing non-overlapping geographic holdout validation metrics"}
    elif has_overlap:
        c3 = {"passed": False, "reason": "Geographic holdout overlaps training region or building set"}
    else:
        c3 = {
            "passed": hold_cand >= hold_base,
            "baseline_holdout_f1": round(hold_base, 4),
            "candidate_holdout_f1": round(hold_cand, 4),
        }

    # Check 4: Calibration preservation (lower ECE is better; reject only calibration degradation > tolerance)
    base_ece = _finite_or_none(baseline_metrics.get("ece"))
    cand_ece = _finite_or_none(candidate_metrics.get("ece"))
    if base_ece is None or cand_ece is None:
        c4 = {"passed": False, "reason": "Missing finite ECE calibration metrics"}
    else:
        ece_degradation = round(cand_ece - base_ece, 4)
        c4 = {
            "passed": ece_degradation <= ece_tol,
            "baseline_ece": round(base_ece, 4),
            "candidate_ece": round(cand_ece, 4),
            "ece_delta": ece_degradation,
            "tolerance": ece_tol,
        }

    # Check 5: Edge latency budget (per tile + projected 600-tile sector < 30 min = 1800s)
    lat_ms = _finite_or_none(candidate_metrics.get("tile_latency_ms"))
    if lat_ms is None or lat_ms <= 0.0:
        c5 = {"passed": False, "reason": "Missing positive finite edge tile_latency_ms"}
    else:
        projected_sector_sec = round((lat_ms * 600.0) / 1000.0, 2)
        c5 = {
            "passed": (lat_ms <= tile_tol) and (projected_sector_sec <= 1800.0),
            "tile_latency_ms": round(lat_ms, 3),
            "max_tile_ms": tile_tol,
            "projected_600_tile_sec": projected_sector_sec,
        }

    checks = {
        "per_class_f1": c1,
        "forgetting_check": c2,
        "geographic_holdout": c3,
        "calibration_preserved": c4,
        "edge_latency": c5,
    }
    all_passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "passed": all_passed,
        "destroyed_f1": cand_dest if cand_dest is not None else 0.0,
        "checks": checks,
    }

