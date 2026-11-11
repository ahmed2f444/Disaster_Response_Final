"""Portion 3 (Yousef) — 7-Feature Routing SHAP Explainer (Unified System Doc Sec. 4.2 & 9).

Attributes PPO routing proposals (`ppo_proposed_path`, Mismatch 6) across all seven
Unified System Document routing factors:
    hazard_distance, travel_time, road_capacity, closure_state,
    vehicle_restriction, incident_severity, unit_capability

Uses `shap.KernelExplainer` against a deterministic Houston-East corridor reference
distribution (exact 2^7 coalition evaluation when `shap` is available) and falls back
cleanly to exact linear Shapley attribution if `shap` is unavailable.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False

CANONICAL_SHAP_FEATURES: tuple[str, ...] = (
    "hazard_distance",
    "travel_time",
    "road_capacity",
    "closure_state",
    "vehicle_restriction",
    "incident_severity",
    "unit_capability",
)

# Linear PPO-surrogate valuation weights over normalized [0, 1] feature vector
_VALUE_WEIGHTS = np.array([-0.40, -0.35, 0.22, -0.50, -0.30, 0.18, 0.25], dtype=np.float64)
_VALUE_BIAS = 0.50
_HAZARD_SCALE_M = 3500.0
_TRAVEL_SCALE_S = 720.0

_EXPLAINER_LOCK = threading.Lock()
_CACHED_EXPLAINER: Any = None


def _build_reference_background() -> np.ndarray:
    """Deterministic 16-sample open-corridor baseline matrix with full column variance."""
    rng = np.random.default_rng(42)
    # Columns: hazard_prox, travel_time, road_capacity, closure_state, vehicle_restriction, severity, capability
    base = np.array(
        [
            [0.08, 0.22, 0.96, 0.00, 0.00, 0.45, 1.00],
            [0.12, 0.26, 0.92, 0.00, 0.00, 0.50, 1.00],
            [0.15, 0.30, 0.88, 0.05, 0.00, 0.55, 0.95],
            [0.10, 0.24, 0.94, 0.00, 0.05, 0.40, 1.00],
            [0.20, 0.35, 0.85, 0.10, 0.05, 0.60, 0.90],
            [0.05, 0.18, 1.00, 0.00, 0.00, 0.35, 1.00],
            [0.18, 0.32, 0.82, 0.08, 0.04, 0.50, 0.85],
            [0.25, 0.40, 0.78, 0.12, 0.06, 0.65, 0.80],
            [0.09, 0.21, 0.95, 0.02, 0.01, 0.42, 0.98],
            [0.14, 0.28, 0.90, 0.04, 0.02, 0.48, 0.92],
            [0.22, 0.36, 0.80, 0.15, 0.08, 0.58, 0.88],
            [0.11, 0.25, 0.93, 0.00, 0.03, 0.46, 0.96],
            [0.16, 0.29, 0.87, 0.06, 0.04, 0.52, 0.94],
            [0.07, 0.20, 0.98, 0.00, 0.00, 0.38, 0.99],
            [0.19, 0.33, 0.84, 0.09, 0.05, 0.62, 0.86],
            [0.13, 0.27, 0.91, 0.03, 0.02, 0.49, 0.93],
        ],
        dtype=np.float64,
    )
    jitter = rng.normal(0.0, 0.005, size=base.shape)
    return np.clip(base + jitter, 0.0, 1.0)


_REFERENCE_BACKGROUND = _build_reference_background()


def build_value_function():
    """Surrogate valuation function mimicking PPO route preference over the 7 features."""

    def value_fn(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr @ _VALUE_WEIGHTS + _VALUE_BIAS

    return value_fn


def _get_kernel_explainer() -> Any:
    global _CACHED_EXPLAINER
    if not _SHAP_AVAILABLE:
        return None
    with _EXPLAINER_LOCK:
        if _CACHED_EXPLAINER is None:
            _CACHED_EXPLAINER = shap.KernelExplainer(build_value_function(), _REFERENCE_BACKGROUND)
        return _CACHED_EXPLAINER


def _extract_feature_dict(payload: dict[str, Any], mask_altered: bool | None = None) -> tuple[dict[str, Any], bool]:
    """Resolve raw routing feature values from either a Route object or a feature dict."""
    effective_mask = bool(payload.get("mask_altered", False)) if mask_altered is None else bool(mask_altered)
    if isinstance(payload.get("routes"), list) and payload["routes"]:
        sub_routes = [r for r in payload["routes"] if isinstance(r, dict)]
        primary = next((r for r in sub_routes if r.get("mask_altered")), sub_routes[0] if sub_routes else {})
        if isinstance(primary.get("features"), dict):
            return primary["features"], effective_mask

    if any(
        k in payload
        for k in (
            "travel_time_s",
            "travel_time_seconds",
            "travel_time",
            "hazard_distance_m",
            "hazard_distance_meters",
            "hazard_distance",
        )
    ):
        return payload, effective_mask

    # Derive fallback feature snapshot when only high-level Route summary fields exist
    road_states = [rs for rs in payload.get("road_states", []) if isinstance(rs, dict)]
    blocked_cnt = sum(1 for rs in road_states if str(rs.get("state", "")).lower() == "blocked")
    haz_cnt = sum(1 for rs in road_states if str(rs.get("state", "")).lower() == "hazardous")
    min_cap = min((float(rs.get("capacity", 1.0)) for rs in road_states), default=1.0)
    restr_cnt = sum(
        1 for rs in road_states if str(rs.get("vehicle_restriction", "none") or "none").lower() not in ("", "none")
    )
    derived = {
        "travel_time_s": float(payload.get("eta_seconds") or payload.get("total_eta_sec") or 360.0),
        "hazard_distance_m": 150.0 if (blocked_cnt or haz_cnt) else 1500.0,
        "road_capacity": min_cap,
        "closure_penalty": float(blocked_cnt) + 0.5 * float(haz_cnt) + (0.5 if effective_mask and blocked_cnt == 0 else 0.0),
        "vehicle_restriction_penalty": float(restr_cnt) * 0.5,
        "incident_severity": 0.75,
        "unit_capability_match": 1.0,
    }
    return derived, effective_mask


def extract_routing_feature_vector(
    routing_features_or_route: dict[str, Any],
    mask_altered: bool | None = None,
) -> np.ndarray:
    """Convert raw routing features or Route payload into a normalized `(7,)` vector."""
    raw, effective_mask = _extract_feature_dict(routing_features_or_route, mask_altered=mask_altered)

    if "hazard_distance_m" in raw or "hazard_distance_meters" in raw:
        hd_m = float(raw.get("hazard_distance_m", raw.get("hazard_distance_meters", _HAZARD_SCALE_M)))
        hazard_prox = max(0.0, 1.0 - min(max(0.0, hd_m), _HAZARD_SCALE_M) / _HAZARD_SCALE_M)
    else:
        hazard_prox = min(1.0, max(0.0, float(raw.get("hazard_distance", 0.2))))

    if "travel_time_s" in raw or "travel_time_seconds" in raw:
        tt_s = float(raw.get("travel_time_s", raw.get("travel_time_seconds", 0.0)))
        travel_norm = min(1.0, max(0.0, tt_s / _TRAVEL_SCALE_S))
    else:
        travel_norm = min(1.0, max(0.0, float(raw.get("travel_time", 0.3))))

    cap_raw = float(raw.get("road_capacity", 1.0))
    cap_norm = min(1.0, max(0.0, cap_raw / 1800.0)) if cap_raw > 1.5 else min(1.0, max(0.0, cap_raw))

    if "closure_penalty" in raw:
        clos_pen = float(raw.get("closure_penalty", 0.0))
        closure_norm = min(1.0, max(0.0, clos_pen * 0.4 + (0.30 if effective_mask and clos_pen <= 0.0 else 0.0)))
    else:
        closure_norm = min(1.0, max(0.0, float(raw.get("closure_state", 0.35 if effective_mask else 0.0))))

    if "vehicle_restriction_penalty" in raw:
        restr_pen = float(raw.get("vehicle_restriction_penalty", 0.0))
        restr_norm = min(1.0, max(0.0, restr_pen * 0.5))
    else:
        restr_norm = min(1.0, max(0.0, float(raw.get("vehicle_restriction", 0.0))))

    sev_norm = min(1.0, max(0.0, float(raw.get("incident_severity", raw.get("incident_severity_score", 0.5)))))
    cap_match_norm = min(1.0, max(0.0, float(raw.get("unit_capability_match", raw.get("unit_capability", 1.0)))))

    return np.array(
        [hazard_prox, travel_norm, cap_norm, closure_norm, restr_norm, sev_norm, cap_match_norm],
        dtype=np.float64,
    )


def compute_shap(
    routing_features: dict[str, Any],
    mask_altered: bool | None = None,
) -> dict[str, float]:
    """Compute signed 7-feature Shapley values on `ppo_proposed_path` features."""
    vec = extract_routing_feature_vector(routing_features, mask_altered=mask_altered)
    explainer = _get_kernel_explainer()

    if explainer is not None:
        raw_vals = explainer.shap_values(vec.reshape(1, -1), nsamples=128, silent=True)
        if isinstance(raw_vals, list):
            raw_vals = raw_vals[0]
        phi = np.asarray(raw_vals, dtype=np.float64).reshape(-1)
    else:
        # Exact analytical linear Shapley attribution when shap package is unavailable
        phi = _VALUE_WEIGHTS * (vec - _REFERENCE_BACKGROUND.mean(axis=0))

    return {name: round(float(val), 6) for name, val in zip(CANONICAL_SHAP_FEATURES, phi)}


def compute_shap_attributions(
    routing_features_or_route: dict[str, Any],
    mask_altered: bool | None = None,
) -> list[dict[str, Any]]:
    """Return sorted `ShapAttribution` entries (`feature`, `value`, `impact`, `attribution`)."""
    signed = compute_shap(routing_features_or_route, mask_altered=mask_altered)
    vec = extract_routing_feature_vector(routing_features_or_route, mask_altered=mask_altered)

    # Combine KernelSHAP marginal deviation magnitude with base factor salience so both
    # positive capacity/capability matches and active hazard/closure penalties read clearly.
    salience: dict[str, float] = {}
    for idx, name in enumerate(CANONICAL_SHAP_FEATURES):
        shap_mag = abs(signed[name])
        feat_val = float(vec[idx])
        if name == "road_capacity":
            feat_salience = (1.0 - feat_val) * 0.12
        elif name in {"incident_severity", "unit_capability"}:
            feat_salience = feat_val * 0.08
        else:
            feat_salience = feat_val * 0.14
        salience[name] = max(0.015, shap_mag + feat_salience)

    total = sum(salience.values()) or 1.0
    items: list[dict[str, Any]] = []
    for name, score in sorted(salience.items(), key=lambda kv: (-kv[1], kv[0])):
        val = round(min(1.0, max(0.0, (score / total) * 1.6)), 3)
        items.append(
            {
                "feature": name,
                "value": val,
                "impact": val,
                "attribution": val,
                "raw_shap": round(signed[name], 4),
            }
        )
    return items

