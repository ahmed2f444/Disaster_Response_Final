"""Portion 2 — MARL/PPO Disaster Response Routing & Safety Mask Package.

Exposes graph adapters, Gymnasium routing environment, PPO trainer, Section 4.2
routing features, and multi-unit dispatch / What-If counterfactual entrypoints.
"""

from .env import RoadGraphIndex, RoutingEnv, build_obs, state_blocked
from .features import FEATURE_NAMES, SEVERITY, capability_match, compute_features, features_to_shap_attributions
from .graph import (
    DEFAULT_GRAPH_PATH,
    HOUSTON_GRAPH_PATH,
    build_from_osm,
    build_graph_from_db,
    edge_id,
    export_houston_graphml,
    haversine_m,
    is_blocked,
    load_graph,
    road_state_label,
    save_graph,
    state_from_db,
    synthetic_graph,
    update_road_hazards,
)
from .mask import apply_mask, safe_view
from .generate import (
    DEFAULT_FLEET_UNITS,
    RoutingEngine,
    compute_multi_unit_routes,
    nearest_node,
    run_counterfactual,
    warm_engine,
)

__all__ = [
    "DEFAULT_FLEET_UNITS",
    "DEFAULT_GRAPH_PATH",
    "FEATURE_NAMES",
    "HOUSTON_GRAPH_PATH",
    "RoadGraphIndex",
    "RoutingEngine",
    "RoutingEnv",
    "SEVERITY",
    "apply_mask",
    "build_from_osm",
    "build_graph_from_db",
    "build_obs",
    "capability_match",
    "compute_features",
    "compute_multi_unit_routes",
    "edge_id",
    "export_houston_graphml",
    "features_to_shap_attributions",
    "haversine_m",
    "is_blocked",
    "load_graph",
    "nearest_node",
    "road_state_label",
    "run_counterfactual",
    "safe_view",
    "save_graph",
    "state_blocked",
    "state_from_db",
    "synthetic_graph",
    "update_road_hazards",
    "warm_engine",
]
