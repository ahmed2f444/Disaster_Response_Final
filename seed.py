"""Portion 5 (Ahmed) — Standalone Deterministic Sector Seeder.

Initializes `data/poc_mission.db` in SQLite WAL mode with Houston-East (`xBD` Harvey)
baseline records using ONLY `db.py` helper functions (zero raw SQL and zero
dependencies on teammate P1–P4 modules):
- 220 building footprints (~20% flagged LOW_CONFIDENCE < 0.65) with Grad-CAM & embedding refs
- 26 road segments across 13 intersections with Section 4.2 attributes & statuses
- 3 active emergency incidents
- 1 baseline multi-unit Route plan (`ppo_proposed_path` and `masked_path`)
- Sample Route & Building Explanation objects
- Initial responder feedback logs & genesis SHA-256 audit entries
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db import (
    DEFAULT_DB_PATH,
    append_audit_event,
    derive_road_display_state,
    init_db,
    insert_building,
    insert_explanation,
    insert_incident,
    insert_road_edge,
    insert_route,
    log_feedback,
)

# 13 tactical intersections across the xBD Hurricane Harvey (houston-east) corridor
NODES: dict[int, tuple[float, float]] = {
    0: (-95.198, 29.830),  # Far West Logistics Depot (MED-01, TRN-01)
    1: (-95.158, 29.857),  # North Highway Interchange
    2: (-95.095, 29.849),  # NE USAR Fire Station (RSC-01)
    3: (-95.190, 29.798),  # SW Industrial Parkway
    4: (-95.134, 29.819),  # Buffalo Bayou Bridgehead (Low Elevation Basin)
    5: (-95.078, 29.810),  # Far East Beltway Overpass
    6: (-95.166, 29.769),  # South-SW Shelter Evacuation Point (TRN-01 Target)
    7: (-95.122, 29.774),  # South Bayou District (RSC-01 Target)
    8: (-95.084, 29.786),  # SE Trauma Medical Campus (MED-01 Target)
    9: (-95.170, 29.827),  # NW Midtown Fork
    10: (-95.118, 29.839),  # NE Ridge Junction
    11: (-95.152, 29.792),  # South-Central Spur
    12: (-95.106, 29.798),  # SE Bayou Overlook
}


def _quad_bezier(u_node: int, ctrl: tuple[float, float], v_node: int, steps: int = 4) -> list[list[float]]:
    x0, y0 = NODES[u_node]
    cx, cy = ctrl
    x1, y1 = NODES[v_node]
    pts: list[list[float]] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1.0 - t
        px = mt * mt * x0 + 2.0 * mt * t * cx + t * t * x1
        py = mt * mt * y0 + 2.0 * mt * t * cy + t * t * y1
        pts.append([round(px, 5), round(py, 5)])
    return pts


# Smooth curved highway, connector & bridge geometries (non-grid, realistic bends)
EDGE_CURVES: dict[str, list[list[float]]] = {
    "E-01": _quad_bezier(0, (-95.184, 29.851), 1),
    "E-02": _quad_bezier(1, (-95.126, 29.860), 2),
    "E-03": _quad_bezier(3, (-95.164, 29.814), 4),
    "E-04": _quad_bezier(4, (-95.104, 29.822), 5),
    "E-05": _quad_bezier(6, (-95.144, 29.765), 7),
    "E-06": _quad_bezier(7, (-95.100, 29.774), 8),
    "E-07": _quad_bezier(0, (-95.203, 29.813), 3),
    "E-08": _quad_bezier(3, (-95.184, 29.779), 6),
    "E-09": _quad_bezier(1, (-95.140, 29.840), 4),
    "E-10": _quad_bezier(4, (-95.137, 29.795), 7),
    "E-11": _quad_bezier(2, (-95.077, 29.832), 5),
    "E-12": _quad_bezier(5, (-95.073, 29.796), 8),
    "E-13": _quad_bezier(0, (-95.184, 29.832), 9, steps=3),
    "E-14": _quad_bezier(9, (-95.168, 29.844), 1, steps=3),
    "E-15": _quad_bezier(9, (-95.178, 29.810), 3, steps=3),
    "E-16": _quad_bezier(9, (-95.151, 29.826), 4, steps=3),
    "E-17": _quad_bezier(1, (-95.136, 29.847), 10, steps=3),
    "E-18": _quad_bezier(10, (-95.106, 29.842), 2, steps=3),
    "E-19": _quad_bezier(10, (-95.095, 29.827), 5, steps=3),
    "E-20": _quad_bezier(9, (-95.158, 29.808), 11, steps=3),
    "E-21": _quad_bezier(3, (-95.171, 29.791), 11, steps=3),
    "E-22": _quad_bezier(11, (-95.156, 29.779), 6, steps=3),
    "E-23": _quad_bezier(11, (-95.139, 29.781), 7, steps=3),
    "E-24": _quad_bezier(5, (-95.093, 29.808), 12, steps=3),
    "E-25": _quad_bezier(12, (-95.118, 29.786), 7, steps=3),
    "E-26": _quad_bezier(12, (-95.094, 29.790), 8, steps=3),
}

# (edge_id, u_node, v_node, base_flood_depth_m) — 0 blocked by default; hazards elevate dynamically
GRID_EDGES: list[tuple[str, int, int, float]] = [
    ("E-01", 0, 1, 0.04),
    ("E-02", 1, 2, 0.06),
    ("E-03", 3, 4, 0.36),  # HAZARDOUS Bayou basin (blocks dynamically on >= +0.14m surge or closure)
    ("E-04", 4, 5, 0.24),  # HAZARDOUS (>= 0.15m)
    ("E-05", 6, 7, 0.11),
    ("E-06", 7, 8, 0.08),
    ("E-07", 0, 3, 0.05),
    ("E-08", 3, 6, 0.07),
    ("E-09", 1, 4, 0.21),  # HAZARDOUS (>= 0.15m)
    ("E-10", 4, 7, 0.34),  # HAZARDOUS Bayou bridgehead (blocks dynamically on >= +0.16m surge or closure)
    ("E-11", 2, 5, 0.05),
    ("E-12", 5, 8, 0.12),
    ("E-13", 0, 9, 0.04),
    ("E-14", 9, 1, 0.05),
    ("E-15", 9, 3, 0.08),
    ("E-16", 9, 4, 0.19),  # HAZARDOUS (>= 0.15m)
    ("E-17", 1, 10, 0.06),
    ("E-18", 10, 2, 0.03),
    ("E-19", 10, 5, 0.09),
    ("E-20", 9, 11, 0.32),  # HAZARDOUS Bayou crossing (blocks dynamically on >= +0.18m surge or closure)
    ("E-21", 3, 11, 0.09),
    ("E-22", 11, 6, 0.06),
    ("E-23", 11, 7, 0.22),  # HAZARDOUS (>= 0.15m)
    ("E-24", 5, 12, 0.07),
    ("E-25", 12, 7, 0.09),
    ("E-26", 12, 8, 0.05),
]

CANONICAL_LABELS = ("no_damage", "minor", "major", "destroyed")

DEFAULT_ROUTING_SHAP = [
    {"feature": "hazard_distance", "value": 0.48, "impact": 0.48, "attribution": 0.48},
    {"feature": "travel_time", "value": 0.35, "impact": 0.35, "attribution": 0.35},
    {"feature": "road_capacity", "value": 0.17, "impact": 0.17, "attribution": 0.17},
    {"feature": "closure_state", "value": 0.14, "impact": 0.14, "attribution": 0.14},
    {"feature": "vehicle_restriction", "value": 0.09, "impact": 0.09, "attribution": 0.09},
    {"feature": "incident_severity", "value": 0.07, "impact": 0.07, "attribution": 0.07},
    {"feature": "unit_capability", "value": 0.05, "impact": 0.05, "attribution": 0.05},
]


def _make_building_polygon(
    lon: float,
    lat: float,
    half_w: float = 0.00085,
    half_h: float = 0.00065,
    angle_rad: float = 0.0,
) -> dict[str, Any]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    ring = [
        [round(lon + dx * cos_a - dy * sin_a, 5), round(lat + dx * sin_a + dy * cos_a, 5)]
        for dx, dy in corners
    ]
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _near_arterial(lon: float, lat: float, min_clearance: float = 0.0020) -> bool:
    for pts in EDGE_CURVES.values():
        for idx in range(len(pts) - 1):
            if _point_to_segment_dist(lon, lat, pts[idx][0], pts[idx][1], pts[idx + 1][0], pts[idx + 1][1]) < min_clearance:
                return True
    return False


def _generate_city_block_centroids(num_buildings: int, rng: random.Random) -> list[tuple[float, float]]:
    """Generate spacious, non-overlapping urban neighborhood clusters outside arterial corridors."""
    cluster_centers = [
        (-95.182, 29.839, 0.011, 0.009),
        (-95.154, 29.840, 0.011, 0.009),
        (-95.134, 29.849, 0.011, 0.008),
        (-95.106, 29.826, 0.011, 0.009),
        (-95.176, 29.782, 0.011, 0.008),
        (-95.146, 29.776, 0.010, 0.008),
        (-95.118, 29.796, 0.010, 0.008),
        (-95.092, 29.798, 0.010, 0.008),
        (-95.186, 29.818, 0.009, 0.008),
        (-95.121, 29.825, 0.010, 0.008),
        (-95.152, 29.804, 0.009, 0.007),
        (-95.092, 29.816, 0.009, 0.007),
    ]
    selected: list[tuple[float, float]] = []
    min_sep_sq = 0.0024 ** 2

    def _is_clear(lon: float, lat: float) -> bool:
        if _near_arterial(lon, lat):
            return False
        return all(((lon - sx) ** 2 + (lat - sy) ** 2) >= min_sep_sq for sx, sy in selected)

    attempts = 0
    while len(selected) < num_buildings and attempts < 14000:
        attempts += 1
        cx, cy, rx, ry = cluster_centers[attempts % len(cluster_centers)]
        lon = round(cx + rng.gauss(0.0, rx * 0.65), 5)
        lat = round(cy + rng.gauss(0.0, ry * 0.65), 5)
        if not (-95.202 <= lon <= -95.078 and 29.768 <= lat <= 29.858):
            continue
        if _is_clear(lon, lat):
            selected.append((lon, lat))

    while len(selected) < num_buildings:
        lon = round(rng.uniform(-95.196, -95.084), 5)
        lat = round(rng.uniform(29.772, 29.852), 5)
        if _near_arterial(lon, lat, min_clearance=0.0014):
            continue
        if all(((lon - sx) ** 2 + (lat - sy) ** 2) >= (0.0018 ** 2) for sx, sy in selected):
            selected.append((lon, lat))

    return selected[:num_buildings]


def _edge_path_to_coords(node_path: list[int], edge_ids: list[str]) -> list[list[float]]:
    """Stitch curved road coordinates cleanly in the exact node_path traversal direction."""
    coords: list[list[float]] = []
    for idx, edge_id in enumerate(edge_ids):
        raw_pts = EDGE_CURVES.get(edge_id)
        if not raw_pts:
            continue
        seg = [list(pt) for pt in raw_pts]
        start_node = node_path[idx]
        start_xy = [NODES[start_node][0], NODES[start_node][1]]
        dist_first = (seg[0][0] - start_xy[0]) ** 2 + (seg[0][1] - start_xy[1]) ** 2
        dist_last = (seg[-1][0] - start_xy[0]) ** 2 + (seg[-1][1] - start_xy[1]) ** 2
        if dist_last < dist_first:
            seg.reverse()
        for pt_idx, pt in enumerate(seg):
            if not coords or pt_idx > 0:
                coords.append(pt)
    return coords or [[NODES[n][0], NODES[n][1]] for n in node_path]


def _coords_to_latlon(coords: list[list[float]]) -> list[dict[str, float]]:
    return [{"lat": c[1], "lon": c[0]} for c in coords]


def _build_baseline_plan(road_states: list[dict[str, Any]], created_utc: str) -> dict[str, Any]:
    """Build deterministic Day-1 baseline Route contract object using fixed sector nodes."""
    unit_specs = [
        {
            "unit_id": "MED-01",
            "unit_class": "MEDICAL",
            "origin_node": 0,
            "target_node": 8,
            "node_path": [0, 1, 2, 5, 8],
            "edge_ids": ["E-01", "E-02", "E-11", "E-12"],
            "edge_statuses": ["OPEN", "OPEN", "OPEN", "OPEN"],
            "ppo_nodes": [0, 1, 2, 5, 8],
            "ppo_edge_ids": ["E-01", "E-02", "E-11", "E-12"],
            "eta_sec": 360.0,
        },
        {
            "unit_id": "TRN-01",
            "unit_class": "TRANSPORT",
            "origin_node": 0,
            "target_node": 6,
            "node_path": [0, 3, 6],
            "edge_ids": ["E-07", "E-08"],
            "edge_statuses": ["OPEN", "OPEN"],
            "ppo_nodes": [0, 3, 6],
            "ppo_edge_ids": ["E-07", "E-08"],
            "eta_sec": 180.0,
        },
        {
            "unit_id": "RSC-01",
            "unit_class": "RESCUE",
            "origin_node": 2,
            "target_node": 7,
            "node_path": [2, 5, 8, 7],
            "edge_ids": ["E-11", "E-12", "E-06"],
            "edge_statuses": ["OPEN", "OPEN", "OPEN"],
            "ppo_nodes": [2, 5, 8, 7],
            "ppo_edge_ids": ["E-11", "E-12", "E-06"],
            "eta_sec": 270.0,
        },
    ]

    unit_routes: list[dict[str, Any]] = []
    for spec in unit_specs:
        safe_coords = _edge_path_to_coords(spec["node_path"], spec["edge_ids"])
        ppo_coords = _edge_path_to_coords(spec["ppo_nodes"], spec["ppo_edge_ids"])
        unit_routes.append(
            {
                "unit_id": spec["unit_id"],
                "unit_class": spec["unit_class"],
                "origin_node": spec["origin_node"],
                "target_node": spec["target_node"],
                "reachable": True,
                "eta_sec": spec["eta_sec"],
                "node_path": spec["node_path"],
                "edge_ids": spec["edge_ids"],
                "edge_statuses": spec["edge_statuses"],
                "hazardous_segments": spec["edge_statuses"].count("HAZARDOUS"),
                "geometry": {"type": "LineString", "coordinates": safe_coords},
                "masked_path": _coords_to_latlon(safe_coords),
                "ppo_proposed_path": _coords_to_latlon(ppo_coords),
                "ppo_edge_ids": spec["ppo_edge_ids"],
                "mask_altered": spec["edge_ids"] != spec["ppo_edge_ids"],
            }
        )

    plan_id = "PLAN-BASELINE-01"
    narrative = (
        "[BASELINE SEED] Plan PLAN-BASELINE-01 initialized with 0 blocked corridors. "
        "Total Fleet ETA: 810.0s (3/3 units routed via E-01, E-02, E-07, E-11, E-12; "
        "Bayou corridors E-03, E-10 remain hazardous at 0.34m–0.36m pending surge or satellite closure)."
    )
    return {
        "plan_id": plan_id,
        "route_id": plan_id,
        "unit_id": "MED-01",
        "incident_id": "INC-01",
        "created_utc": created_utc,
        "approval_status": "PENDING_APPROVAL",
        "status": "pending_approval",
        "is_counterfactual": 0,
        "eta_seconds": 810.0,
        "total_eta_sec": 810.0,
        "compute_ms": 0.0,
        "ppo_proposed_path": unit_routes[0]["ppo_proposed_path"],
        "masked_path": unit_routes[0]["masked_path"],
        "mask_altered": False,
        "mask_reason": "All 26 corridors passable at startup (0 blocked; E-03, E-10 hazardous).",
        "road_states": road_states,
        "uncertainty": 0.08,
        "shap_values": DEFAULT_ROUTING_SHAP,
        "counterfactuals": [
            {"alternative": "Direct Bayou Bridge E-10 (Unmasked PPO Proposal)", "delta_eta": -45.0},
            {"alternative": "East Levy Overpass E-11/E-12 Detour", "delta_eta": 0.0},
        ],
        "routes": unit_routes,
        "narrative_text": narrative,
        "explanation": narrative,
        "narrative_mode": "JINJA2_TEMPLATE",
    }


def seed_sector(db_path: str = DEFAULT_DB_PATH, num_buildings: int = 220) -> dict[str, int]:
    """Initialize SQLite WAL database and seed deterministic Houston-East baseline data."""
    init_db(db_path=db_path, reset=True)
    rng = random.Random(42)
    centroids = _generate_city_block_centroids(num_buildings, rng)
    now_utc = datetime.now(timezone.utc).isoformat()

    # 1. Seed Road Edges (26 edges across 13 intersections)
    road_counts = {"OPEN": 0, "HAZARDOUS": 0, "BLOCKED": 0}
    road_states: list[dict[str, Any]] = []
    for edge_id, u_node, v_node, flood_m in GRID_EDGES:
        lon1, lat1 = NODES[u_node]
        lon2, lat2 = NODES[v_node]
        curve_coords = EDGE_CURVES.get(edge_id) or [[lon1, lat1], [lon2, lat2]]
        geom = {"type": "LineString", "coordinates": curve_coords}

        if flood_m >= 0.50:
            closure = True
            capacity = 0.0
            restriction = "closed"
        elif flood_m >= 0.15:
            closure = False
            capacity = 0.5
            restriction = "high_clearance_only"
        else:
            closure = False
            capacity = 1.0
            restriction = "none"

        initial_status = derive_road_display_state(closure, flood_m, capacity, restriction).upper()

        road_counts[initial_status] += 1
        road_states.append(
            {
                "edge_id": edge_id,
                "state": initial_status.lower(),
                "closure_state": closure,
                "hazard_level": flood_m,
                "capacity": capacity,
                "vehicle_restriction": restriction,
            }
        )

        insert_road_edge(
            {
                "edge_id": edge_id,
                "u_node": u_node,
                "v_node": v_node,
                "geometry": geom,
                "length_m": 1200.0,
                "base_travel_sec": 90.0,
                "flood_depth_m": flood_m,
                "debris_score": 0.0,
                "closure_state": closure,
                "hazard_level": flood_m,
                "capacity": capacity,
                "vehicle_restriction": restriction,
                "edge_status": initial_status,
            },
            db_path=db_path,
        )

    # 2. Seed Buildings (~20% flagged LOW_CONFIDENCE < 0.65 directly; Grad-CAM only per Mismatch 1)
    low_conf_count = 0
    bld_0042_conf = 0.85
    for idx, (lon, lat) in enumerate(centroids, start=1):
        b_id = f"BLD-{idx:04d}"
        dist_to_bayou = math.hypot(lon - (-95.134), lat - 29.816)
        if dist_to_bayou < 0.026:
            dmg_idx = rng.choices([0, 1, 2, 3], weights=[0.15, 0.25, 0.35, 0.25])[0]
        else:
            dmg_idx = rng.choices([0, 1, 2, 3], weights=[0.55, 0.25, 0.13, 0.07])[0]
        dmg_label = CANONICAL_LABELS[dmg_idx]

        half_w = rng.uniform(0.00058, 0.00086)
        half_h = rng.uniform(0.00046, 0.00068)
        parcel_angle = rng.uniform(-0.45, 0.45)
        poly = _make_building_polygon(lon, lat, half_w=half_w, half_h=half_h, angle_rad=parcel_angle)

        if rng.random() < 0.22:
            confidence = round(rng.uniform(0.45, 0.64), 3)
            status_flag = "LOW_CONFIDENCE"
            low_conf_count += 1
        else:
            confidence = round(rng.uniform(0.72, 0.98), 3)
            status_flag = "CALIBRATED"
        if b_id == "BLD-0042":
            bld_0042_conf = confidence

        gradcam_poly = _make_building_polygon(lon, lat, half_w=half_w * 0.6, half_h=half_h * 0.6, angle_rad=parcel_angle)
        gradcam_attributions = [
            {"region": gradcam_poly, "weight": round(rng.uniform(0.75, 0.95), 2)}
        ]

        insert_building(
            {
                "building_id": b_id,
                "geometry": poly,
                "centroid_lat": lat,
                "centroid_lon": lon,
                "damage_class": dmg_label,
                "class_label": dmg_label,
                "confidence": confidence,
                "max_prob": confidence,
                "status_flag": status_flag,
                "gradcam_attribution": gradcam_attributions,
                "shap_top3": [],
                "embedding_ref": f"emb://dinov2/houston/{b_id}",
                "last_updated": now_utc,
                "data_source": "xBD_houston_east",
            },
            db_path=db_path,
        )

    # 3. Seed Incidents
    incidents = [
        {
            "incident_id": "INC-01",
            "location": {"lat": NODES[8][1], "lon": NODES[8][0]},
            "severity": "critical",
            "required_capability": "medical",
            "time_reported": now_utc,
            "status": "assigned",
        },
        {
            "incident_id": "INC-02",
            "location": {"lat": NODES[6][1], "lon": NODES[6][0]},
            "severity": "high",
            "required_capability": "transport",
            "time_reported": now_utc,
            "status": "assigned",
        },
        {
            "incident_id": "INC-03",
            "location": {"lat": NODES[7][1], "lon": NODES[7][0]},
            "severity": "critical",
            "required_capability": "rescue",
            "time_reported": now_utc,
            "status": "assigned",
        },
    ]
    for inc in incidents:
        insert_incident(inc, db_path=db_path)

    # 4. Seed baseline multi-unit Route plan directly
    plan = _build_baseline_plan(road_states=road_states, created_utc=now_utc)
    insert_route(plan, db_path=db_path)

    # 5. Seed baseline Explanation objects
    sample_exps = [
        {
            "explanation_id": "EXP-RT-01",
            "target_type": "route",
            "target_id": plan["plan_id"],
            "natural_language": (
                "Route MED-01 uses High-Ground Perimeter Hwy (E-01, E-02, E-11, E-12) "
                "while Buffalo Bayou segments E-03 and E-10 carry 0.34m–0.36m hazardous water (ETA 810.0s)."
            ),
            "attribution_method": "shap",
            "system_uncertainty": 0.08,
            "llm_used": False,
            "validation_passed": True,
            "timestamp": now_utc,
        },
        {
            "explanation_id": "EXP-BLD-0042",
            "target_type": "building",
            "target_id": "BLD-0042",
            "natural_language": "Baseline Grad-CAM region highlights north roof structure for damage inspection.",
            "attribution_method": "gradcam",
            "system_uncertainty": round(1.0 - bld_0042_conf, 4),
            "llm_used": False,
            "validation_passed": True,
            "timestamp": now_utc,
        },
    ]
    for exp in sample_exps:
        insert_explanation(exp, db_path=db_path)

    # 6. Seed initial feedback log entries (Portion 4 queue)
    log_feedback(
        {
            "target_id": "BLD-0005",
            "disaster_id": "HOUSTON_HARVEY_2017",
            "prediction": "minor",
            "uncertainty": 0.15,
            "responder_action": "confirm",
            "responder_id": "RESPONDER_FIELD_ALPHA",
            "time_taken_sec": 1.2,
        },
        db_path=db_path,
    )
    log_feedback(
        {
            "target_id": "BLD-0012",
            "disaster_id": "HOUSTON_HARVEY_2017",
            "prediction": "major",
            "uncertainty": 0.28,
            "responder_action": "correct",
            "corrected_label": "destroyed",
            "responder_id": "RESPONDER_FIELD_BRAVO",
            "time_taken_sec": 2.4,
        },
        db_path=db_path,
    )

    # 7. Append Genesis Audit Event
    append_audit_event(
        actor_role="SYSTEM",
        event_type="DB_SEEDED",
        payload={
            "sector": "xBD_houston_east",
            "buildings": num_buildings,
            "road_edges": len(GRID_EDGES),
            "plan_id": plan["plan_id"],
        },
        route_id=plan["plan_id"],
        db_path=db_path,
    )

    return {
        "buildings": num_buildings,
        "low_confidence_buildings": low_conf_count,
        "roads_open": road_counts["OPEN"],
        "roads_hazardous": road_counts["HAZARDOUS"],
        "roads_blocked": road_counts["BLOCKED"],
        "initial_plan_eta_sec": int(plan["total_eta_sec"]),
    }


if __name__ == "__main__":
    summary = seed_sector()
    print(f"[SEED COMPLETE] {json.dumps(summary, indent=2)}")

