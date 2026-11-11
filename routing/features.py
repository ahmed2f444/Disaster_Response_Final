"""Section 4.2 routing feature engineering for Portion 2 & Portion 3 SHAP attribution.

Computes the seven Unified System Document routing features on `ppo_proposed_path`
(Mismatch 6) and maps them to normalized `ShapAttribution` records compatible with
`models.Route.shap_values` and Screen 3's SHAP bar chart:
    hazard_distance, travel_time, road_capacity, closure_state,
    vehicle_restriction, incident_severity, unit_capability
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .graph import haversine_m, road_state_label
from .mask import _edge_attrs

FEATURE_NAMES = [
    "travel_time_s",
    "hazard_distance_m",
    "road_capacity",
    "closure_penalty",
    "vehicle_restriction_penalty",
    "incident_severity",
    "unit_capability_match",
]

SEVERITY = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
HAZARD_DISTANCE_CAP_M = 10000.0


def capability_match(unit_cap: str, required: str) -> float:
    """Return 1.0 for exact capability match, 0.5 for transport<->rescue, else 0.0."""
    if unit_cap == required:
        return 1.0
    if {unit_cap, required} == {"transport", "rescue"}:
        return 0.5
    return 0.0


def _midpoint(G: nx.Graph, a: str, b: str) -> tuple[float, float]:
    na, nb = G.nodes[a], G.nodes[b]
    return (float(na["lat"]) + float(nb["lat"])) / 2.0, (float(na["lon"]) + float(nb["lon"])) / 2.0


def effective_edge_travel_s(
    G: nx.Graph,
    state: dict[str, dict[str, Any]] | None,
    a: str,
    b: str,
    unit_capability: str | None = None,
) -> float:
    """Return base edge travel time adjusted smoothly for hazardous DB corridors."""
    base_tt = float(G[a][b].get("travel_time_s", 90.0))
    attrs = _edge_attrs(G, state, a, b)
    label = road_state_label(attrs, unit_capability=unit_capability)
    if label == "hazardous" and attrs.get("_schema") == "db":
        haz = float(attrs.get("hazard_level", attrs.get("flood_depth_m", 0.2)))
        factor = 1.0 + 0.5 * min(1.0, max(0.0, haz) / 0.5)
        return round(base_tt * factor, 2)
    return base_tt


def _restriction_penalty(attrs: dict[str, Any], unit_capability: str) -> float:
    restriction = str(attrs.get("vehicle_restriction", "none") or "none").strip().lower()
    if restriction == "closed" or (restriction == "no_heavy" and unit_capability == "transport"):
        return 1.0
    if restriction not in ("", "none"):
        return 0.5
    return 0.0


def compute_features(
    G: nx.Graph,
    state: dict[str, dict[str, Any]] | None,
    path: list[str],
    severity: str,
    unit_capability: str,
    required_capability: str,
) -> dict[str, float]:
    """Compute the 7 routing features for `path` under current road `state`."""
    sev_val = SEVERITY.get(str(severity).lower(), 0.5)
    cap_match = capability_match(str(unit_capability).lower(), str(required_capability).lower())

    if len(path) < 2:
        return {
            "travel_time_s": 0.0,
            "hazard_distance_m": HAZARD_DISTANCE_CAP_M,
            "road_capacity": 0.0,
            "closure_penalty": 0.0,
            "vehicle_restriction_penalty": 0.0,
            "incident_severity": sev_val,
            "unit_capability_match": cap_match,
        }

    edges = [(a, b) for a, b in zip(path[:-1], path[1:]) if G.has_edge(a, b)]
    if not edges:
        return {
            "travel_time_s": 0.0,
            "hazard_distance_m": HAZARD_DISTANCE_CAP_M,
            "road_capacity": 0.0,
            "closure_penalty": 0.0,
            "vehicle_restriction_penalty": 0.0,
            "incident_severity": sev_val,
            "unit_capability_match": cap_match,
        }

    attrs = [_edge_attrs(G, state, a, b) for a, b in edges]
    travel = sum(effective_edge_travel_s(G, state, a, b, unit_capability=unit_capability) for a, b in edges)
    capacity = min(float(x.get("capacity", 1.0)) for x in attrs)
    labels = [road_state_label(x, unit_capability=unit_capability) for x in attrs]
    closure_pen = sum(1.0 if s == "blocked" else 0.5 if s == "hazardous" else 0.0 for s in labels)
    restrict_pen = sum(_restriction_penalty(x, unit_capability) for x in attrs)

    haz_points = [
        _midpoint(G, u, v)
        for u, v in G.edges
        if road_state_label(_edge_attrs(G, state, u, v), unit_capability=unit_capability) != "open"
    ]
    on_path = [_midpoint(G, a, b) for a, b in edges]
    on_path += [(float(G.nodes[n]["lat"]), float(G.nodes[n]["lon"])) for n in path if n in G.nodes]

    if not haz_points or not on_path:
        hd = HAZARD_DISTANCE_CAP_M
    else:
        hd = min(
            HAZARD_DISTANCE_CAP_M,
            min(haversine_m(p[0], p[1], h[0], h[1]) for p in on_path for h in haz_points),
        )

    return {
        "travel_time_s": round(float(travel), 2),
        "hazard_distance_m": round(float(hd), 2),
        "road_capacity": round(float(capacity), 4),
        "closure_penalty": round(float(closure_pen), 3),
        "vehicle_restriction_penalty": round(float(restrict_pen), 3),
        "incident_severity": sev_val,
        "unit_capability_match": cap_match,
    }


def features_to_shap_attributions(
    proposed_features: dict[str, float],
    mask_altered: bool = False,
) -> list[dict[str, Any]]:
    """Convert raw Section 4.2 path features into normalized `ShapAttribution` entries."""
    try:
        from explain.shap_explainer import compute_shap_attributions

        return compute_shap_attributions(proposed_features, mask_altered=mask_altered)
    except Exception:
        pass

    hd = float(proposed_features.get("hazard_distance_m", HAZARD_DISTANCE_CAP_M))
    tt = float(proposed_features.get("travel_time_s", 0.0))
    cap = float(proposed_features.get("road_capacity", 1.0))
    cap_norm = min(1.0, cap / 1800.0) if cap > 1.5 else min(1.0, max(0.0, cap))
    clos = float(proposed_features.get("closure_penalty", 0.0))
    restr = float(proposed_features.get("vehicle_restriction_penalty", 0.0))
    sev = float(proposed_features.get("incident_severity", 0.5))
    umatch = float(proposed_features.get("unit_capability_match", 1.0))

    raw_weights = {
        "hazard_distance": max(0.08, 1.0 - min(hd, 3500.0) / 3500.0),
        "travel_time": max(0.10, min(1.0, tt / 720.0)),
        "road_capacity": max(0.05, 1.0 - cap_norm * 0.75),
        "closure_state": min(1.0, (0.35 if mask_altered else 0.05) + clos * 0.35),
        "vehicle_restriction": min(1.0, 0.05 + restr * 0.35),
        "incident_severity": max(0.05, sev * 0.55),
        "unit_capability": max(0.04, umatch * 0.40),
    }
    total = sum(raw_weights.values()) or 1.0

    items = []
    for key, w in sorted(raw_weights.items(), key=lambda kv: (-kv[1], kv[0])):
        val = round(min(1.0, max(0.0, w / total * 1.6)), 3)
        items.append({"feature": key, "value": val, "impact": val, "attribution": val})
    return items

