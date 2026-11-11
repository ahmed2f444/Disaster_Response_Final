"""Road graph management and Section 4.2 hazard state adapter for Portion 2 (routing).

Supports two complementary graph sources:
1. Live Houston-East tactical network stored in SQLite via `db.py` (Zero-SQL):
   13 intersections (`n0..n12`) and 26 curved road segments (`E-01..E-26`).
2. Offline/OSM GraphML networks (`build_from_osm`, `synthetic_graph`, `load_graph`, `save_graph`).

Implements the Unified System Document Section 4.2 four-to-three road display state
mapping (`open`, `hazardous`, `blocked`) from `(closure_state, hazard_level, capacity,
vehicle_restriction)` as well as `update_road_hazards` for `main.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import networkx as nx

import db
import seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "osm" / "road_graph.graphml"
HOUSTON_GRAPH_PATH = PROJECT_ROOT / "data" / "osm" / "houston_east_graph.graphml"

N_NODES = 10
HAZARD_THRESHOLD = 0.4        # >= 0.4 on normalized [0, 1] GraphML scale -> hazardous
FLOOD_THRESHOLD = 0.8         # >= 0.8 on normalized [0, 1] GraphML scale -> flooded/blocked
DB_HAZARD_THRESHOLD = 0.15    # >= 0.15m on Houston-East DB scale -> hazardous
DB_FLOOD_THRESHOLD = 0.50     # >= 0.50m on Houston-East DB scale -> flooded/blocked

_SPEED = {
    "motorway": 90,
    "trunk": 70,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "unclassified": 35,
    "residential": 30,
    "living_street": 15,
    "service": 20,
}


def edge_id(a: str, b: str) -> str:
    """Canonical undirected pair identifier (`n0-n1`) sorted lexicographically."""
    x, y = sorted((a, b))
    return f"{x}-{y}"


def node_name(node_ref: int | str) -> str:
    """Normalize integer or string node identifier into `n<index>` form."""
    text = str(node_ref).strip()
    return text if text.startswith("n") else f"n{int(text)}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 coordinates."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _is_db_schema_attrs(attrs: dict[str, Any]) -> bool:
    return (
        attrs.get("_schema") == "db"
        or "closure_state" in attrs
        or "flood_depth_m" in attrs
        or "hazard_level" in attrs
    )


def road_state_label(attrs: dict[str, Any], unit_capability: str | None = None) -> str:
    """Map Section 4.2 underlying attributes into `open`, `hazardous`, or `blocked`.

    Supports both Section 4.2 DB attributes (`closure_state`, `hazard_level`/`flood_depth_m`,
    `capacity`, `vehicle_restriction`) and normalized `[0, 1]` GraphML attributes
    (`closed`, `hazard`, `capacity`, `vehicle_restriction`).
    """
    restriction = str(attrs.get("vehicle_restriction", "none") or "none").strip().lower()
    cap = float(attrs.get("capacity", 1.0))

    if restriction == "closed" or cap <= 0.0:
        return "blocked"
    if restriction == "no_heavy" and unit_capability == "transport":
        return "blocked"

    if _is_db_schema_attrs(attrs):
        closed = bool(attrs.get("closure_state", False)) or int(attrs.get("closed", 0)) == 1
        flood_m = float(attrs.get("flood_depth_m", 0.0))
        haz_lvl = float(attrs.get("hazard_level", max(flood_m, float(attrs.get("debris_score", 0.0)))))
        if closed or flood_m >= DB_FLOOD_THRESHOLD or haz_lvl >= DB_FLOOD_THRESHOLD:
            return "blocked"
        if haz_lvl >= DB_HAZARD_THRESHOLD or restriction not in ("", "none"):
            return "hazardous"
        return "open"

    closed = int(attrs.get("closed", 0)) == 1
    if closed:
        return "blocked"
    h = float(attrs.get("hazard", 0.0))
    if h >= FLOOD_THRESHOLD:
        return "blocked"
    if h >= HAZARD_THRESHOLD or restriction not in ("", "none"):
        return "hazardous"
    return "open"


def is_blocked(attrs: dict[str, Any], unit_capability: str | None = None) -> bool:
    """Return True if an edge is impassable under safety mask constraints."""
    return road_state_label(attrs, unit_capability=unit_capability) == "blocked"


def graph_edge_to_db_id(G: nx.Graph, u: str, v: str) -> str:
    """Return the database edge ID (e.g. `E-01`) for graph edge `(u, v)`."""
    d = G[u][v]
    return str(d.get("db_edge_id") or d.get("edge_id") or edge_id(u, v))


def db_id_to_pair(G: nx.Graph, db_eid: str) -> str | None:
    """Resolve a database edge ID (`E-01`) or pair ID (`n0-n1`) to canonical `pair_id`."""
    for u, v, d in G.edges(data=True):
        pid = edge_id(u, v)
        if db_eid in (d.get("db_edge_id"), d.get("edge_id"), pid):
            return pid
    return None


def _resolve_node_coords(edge_rows: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    """Build `{node_idx: (lon, lat)}` from DB edge geometries, falling back to `seed.NODES`."""
    coords: dict[int, tuple[float, float]] = {}
    for row in edge_rows:
        u_idx = int(row["u_node"])
        v_idx = int(row["v_node"])
        geom_pts = (row.get("geometry") or {}).get("coordinates") or []
        if u_idx not in coords:
            if len(geom_pts) >= 2:
                coords[u_idx] = (float(geom_pts[0][0]), float(geom_pts[0][1]))
            elif u_idx in seed.NODES:
                coords[u_idx] = seed.NODES[u_idx]
        if v_idx not in coords:
            if len(geom_pts) >= 2:
                coords[v_idx] = (float(geom_pts[-1][0]), float(geom_pts[-1][1]))
            elif v_idx in seed.NODES:
                coords[v_idx] = seed.NODES[v_idx]
    return coords


def _normalize_db_hazard(hazard_level: float) -> float:
    """Scale Houston-East hazard meters `[0, 0.625+]` into `[0.0, 1.0]` for PPO obs."""
    return round(min(1.0, max(0.0, float(hazard_level) / 0.625)), 4)


def build_graph_from_db(db_path: str = db.DEFAULT_DB_PATH) -> nx.Graph:
    """Build the NetworkX road graph from `db.get_road_edges` (Zero-SQL)."""
    rows = db.get_road_edges(db_path=db_path)
    if not rows:
        raise ValueError(f"No road edges found in database: {db_path}")

    node_coords = _resolve_node_coords(rows)
    G = nx.Graph(provenance="houston_east_db", note="Houston-East xBD road network from db.py")

    for n_idx in sorted(node_coords):
        lon, lat = node_coords[n_idx]
        G.add_node(f"n{n_idx}", node_index=n_idx, lat=float(lat), lon=float(lon), osmid=f"houston_n{n_idx}")

    for row in rows:
        u = f"n{int(row['u_node'])}"
        v = f"n{int(row['v_node'])}"
        pid = edge_id(u, v)
        db_eid = str(row["edge_id"])
        length_m = float(row.get("length_m", 1200.0))
        travel_s = float(row.get("base_travel_sec", 90.0))
        speed_kph = round((length_m / travel_s) * 3.6, 2) if travel_s > 0 else 48.0
        flood_m = float(row.get("flood_depth_m", 0.0))
        debris = float(row.get("debris_score", 0.0))
        haz_lvl = float(row.get("hazard_level", max(flood_m, debris)))
        closed = int(bool(row.get("closure_state", False)))
        capacity = float(row.get("capacity", 1.0))
        restriction = str(row.get("vehicle_restriction", "none") or "none")
        geom = row.get("geometry") or {
            "type": "LineString",
            "coordinates": [
                [G.nodes[u]["lon"], G.nodes[u]["lat"]],
                [G.nodes[v]["lon"], G.nodes[v]["lat"]],
            ],
        }
        G.add_edge(
            u,
            v,
            pair_id=pid,
            edge_id=pid,
            db_edge_id=db_eid,
            length_m=length_m,
            speed_kph=speed_kph,
            travel_time_s=travel_s,
            capacity=capacity,
            hazard=_normalize_db_hazard(haz_lvl),
            hazard_level=haz_lvl,
            flood_depth_m=flood_m,
            debris_score=debris,
            closed=closed,
            closure_state=closed,
            closure_reported=0 if closed else 1,
            vehicle_restriction=restriction,
            highway="primary" if (int("".join(c for c in db_eid if c.isdigit()) or "1") <= 12) else "secondary",
            osm_path=db_eid,
            geom_json=json.dumps(geom, separators=(",", ":")),
        )
    return G


def state_from_db(
    db_path: str = db.DEFAULT_DB_PATH,
    blocked_edges: list[str] | None = None,
    flood_delta: float = 0.0,
    reported_closures: set[str] | None = None,
    opened_edges: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build an in-memory road state dict keyed by `pair_id` (`n0-n1`) without mutating DB."""
    rows = db.get_road_edges(db_path=db_path)
    blocked_set = set(blocked_edges or [])
    opened_set = set(opened_edges or []) - blocked_set
    reported_set = set(reported_closures or ())
    state: dict[str, dict[str, Any]] = {}

    for row in rows:
        u = f"n{int(row['u_node'])}"
        v = f"n{int(row['v_node'])}"
        pid = edge_id(u, v)
        db_eid = str(row["edge_id"])

        base_flood = float(row.get("flood_depth_m", 0.0))
        eff_flood = max(0.0, round(base_flood + float(flood_delta), 3))
        debris = float(row.get("debris_score", 0.0))
        eff_hazard_lvl = round(max(eff_flood, debris), 3)

        manual_block = db_eid in blocked_set or pid in blocked_set
        manual_open = db_eid in opened_set or pid in opened_set
        base_closed = bool(row.get("closure_state", False))
        # Precedence: manual block > manual open > flood threshold / receding surge > DB closure
        if manual_block:
            eff_closed = True
            eff_flood = max(eff_flood, DB_FLOOD_THRESHOLD)
            eff_hazard_lvl = max(eff_hazard_lvl, DB_FLOOD_THRESHOLD)
        elif manual_open:
            eff_closed = False
            eff_flood = round(min(eff_flood, 0.05), 3)
            debris = 0.0
            eff_hazard_lvl = eff_flood
        elif eff_flood >= DB_FLOOD_THRESHOLD:
            eff_closed = True
        elif float(flood_delta) < 0.0 and str(row.get("vehicle_restriction", "")).lower() == "closed":
            eff_closed = False
        else:
            eff_closed = base_closed

        if eff_closed:
            eff_cap = 0.0
            eff_rest = "closed"
        elif manual_open:
            eff_cap = 1.0
            eff_rest = "none"
        elif eff_hazard_lvl >= DB_HAZARD_THRESHOLD:
            eff_cap = min(float(row.get("capacity", 0.5)) or 0.5, 0.5)
            prev_rest = str(row.get("vehicle_restriction", "none") or "none")
            eff_rest = "high_clearance_only" if prev_rest in ("none", "closed", "") else prev_rest
        else:
            eff_cap = float(row.get("capacity", 1.0))
            if eff_cap <= 0.0:
                eff_cap = 1.0
            prev_rest = str(row.get("vehicle_restriction", "none") or "none")
            eff_rest = "none" if prev_rest in ("closed", "high_clearance_only", "") else prev_rest

        state[pid] = {
            "_schema": "db",
            "db_edge_id": db_eid,
            "pair_id": pid,
            "hazard": _normalize_db_hazard(eff_hazard_lvl),
            "hazard_level": eff_hazard_lvl,
            "flood_depth_m": eff_flood,
            "debris_score": debris,
            "closed": int(eff_closed),
            "closure_state": bool(eff_closed),
            "capacity": eff_cap,
            "vehicle_restriction": eff_rest,
            "reported": 1 if (not eff_closed or db_eid in reported_set or pid in reported_set) else 0,
        }
    return state


def update_road_hazards(
    db_path: str = db.DEFAULT_DB_PATH,
    water_level_delta_m: float = 0.0,
    blocked_edges: list[str] | None = None,
    opened_edges: list[str] | None = None,
) -> dict[str, int]:
    """Update persisted `road_edges` telemetry via `db.py` helpers (Zero-SQL).

    Matches `main.py`'s Portion 2 hook: `update_road_hazards(db_path=..., water_level_delta_m=...)`.
    """
    rows = db.get_road_edges(db_path=db_path)
    blocked_set = set(blocked_edges or [])
    opened_set = set(opened_edges or []) - blocked_set
    counts = {"OPEN": 0, "HAZARDOUS": 0, "BLOCKED": 0}

    for row in rows:
        eid = str(row["edge_id"])
        flood_m = max(0.0, round(float(row.get("flood_depth_m", 0.0)) + float(water_level_delta_m), 3))
        debris = float(row.get("debris_score", 0.0))
        haz_lvl = round(max(flood_m, debris), 3)

        manual_block = eid in blocked_set
        manual_open = eid in opened_set
        was_closed = bool(row.get("closure_state", False))
        if manual_block or flood_m >= DB_FLOOD_THRESHOLD:
            closure = True
            capacity = 0.0
            restriction = "closed"
        elif manual_open:
            closure = False
            flood_m = round(min(flood_m, 0.05), 3)
            debris = 0.0
            haz_lvl = flood_m
            capacity = 1.0
            restriction = "none"
        elif float(water_level_delta_m) < 0.0 and was_closed and str(row.get("vehicle_restriction", "")).lower() == "closed":
            closure = False
            capacity = 0.5 if haz_lvl >= DB_HAZARD_THRESHOLD else 1.0
            restriction = "high_clearance_only" if haz_lvl >= DB_HAZARD_THRESHOLD else "none"
        elif was_closed:
            closure = True
            capacity = 0.0
            restriction = "closed"
        elif haz_lvl >= DB_HAZARD_THRESHOLD:
            closure = False
            capacity = 0.5
            prev_rest = str(row.get("vehicle_restriction", "none") or "none")
            restriction = "high_clearance_only" if prev_rest in ("none", "") else prev_rest
        else:
            closure = False
            capacity = 1.0
            prev_rest = str(row.get("vehicle_restriction", "none") or "none")
            restriction = "none" if prev_rest in ("closed", "high_clearance_only", "") else prev_rest

        status_upper = db.derive_road_display_state(
            closure_state=closure,
            hazard_level=haz_lvl,
            capacity=capacity,
            vehicle_restriction=restriction,
            hazard_threshold=DB_HAZARD_THRESHOLD,
        ).upper()
        counts[status_upper] = counts.get(status_upper, 0) + 1

        if float(water_level_delta_m) != 0.0 or blocked_edges is not None or opened_edges is not None:
            db.update_road_edge(
                edge_id=eid,
                flood_depth_m=flood_m,
                debris_score=debris,
                edge_status=status_upper,
                closure_state=closure,
                hazard_level=haz_lvl,
                capacity=capacity,
                vehicle_restriction=restriction,
                db_path=db_path,
            )

    return counts


def pick_arbitrary_blockage_edge(
    db_path: str = db.DEFAULT_DB_PATH,
    exclude: set[str] | None = None,
    seed_val: int | None = None,
) -> str | None:
    """Select a passable corridor for simulated arbitrary live-satellite blockage while preserving connectivity."""
    rows = db.get_road_edges(db_path=db_path)
    if not rows:
        return None
    excl = set(exclude or ())
    open_rows = [
        r for r in rows
        if str(r.get("edge_status", "")).upper() != "BLOCKED"
        and not bool(r.get("closure_state", False))
        and str(r["edge_id"]) not in excl
    ]
    if not open_rows:
        return None

    G = build_graph_from_db(db_path=db_path)
    passable_graph = nx.Graph()
    passable_graph.add_nodes_from(G.nodes)
    for r in rows:
        eid = str(r["edge_id"])
        if str(r.get("edge_status", "")).upper() != "BLOCKED" and not bool(r.get("closure_state", False)):
            passable_graph.add_edge(f"n{int(r['u_node'])}", f"n{int(r['v_node'])}", db_edge_id=eid)

    key_nodes = [n for n in ("n0", "n2", "n6", "n7", "n8") if n in passable_graph]
    safe_choices: list[str] = []
    for r in sorted(open_rows, key=lambda item: str(item["edge_id"])):
        u = f"n{int(r['u_node'])}"
        v = f"n{int(r['v_node'])}"
        if not passable_graph.has_edge(u, v):
            continue
        passable_graph.remove_edge(u, v)
        connected = all(nx.has_path(passable_graph, key_nodes[0], kn) for kn in key_nodes[1:]) if len(key_nodes) > 1 else True
        passable_graph.add_edge(u, v, db_edge_id=str(r["edge_id"]))
        if connected:
            safe_choices.append(str(r["edge_id"]))

    pool = safe_choices or sorted(str(r["edge_id"]) for r in open_rows)
    rng = random.Random(int(seed_val)) if seed_val is not None else random.SystemRandom()
    return rng.choice(pool)


def apply_satellite_damage_to_roads(
    db_path: str = db.DEFAULT_DB_PATH,
    assessed_buildings: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, int], bool]:
    """Project real satellite-detected building damage onto nearby Houston-East road corridors idempotently."""
    rows = db.get_road_edges(db_path=db_path)
    if not rows:
        return {"OPEN": 0, "HAZARDOUS": 0, "BLOCKED": 0}, False

    dmg_weights = {"destroyed": 1.0, "major": 0.70, "minor": 0.28, "no_damage": 0.0}
    if assessed_buildings is None:
        assessed_buildings = [
            b
            for b in db.get_buildings(db_path=db_path)
            if str(b.get("data_source", "")) == "resnet18_unet_pretrained"
        ]
    damaged: list[tuple[float, float, float]] = []
    for b in assessed_buildings:
        w = dmg_weights.get(str(b.get("damage_class", "no_damage")).lower(), 0.0)
        if w > 0.0:
            damaged.append((float(b.get("centroid_lat", 0.0)), float(b.get("centroid_lon", 0.0)), w))

    is_osm_graph = any(str(r.get("edge_id", "")).startswith("OSM-") for r in rows)
    if is_osm_graph:
        avg_len = sum(float(r.get("length_m", 300.0)) for r in rows) / max(1, len(rows))
        prox_radius_m = max(65.0, min(1800.0, avg_len * 0.35))
    else:
        prox_radius_m = 1800.0

    counts = {"OPEN": 0, "HAZARDOUS": 0, "BLOCKED": 0}
    changed = False
    for row in rows:
        eid = str(row["edge_id"])
        coords = (row.get("geometry") or {}).get("coordinates") or []
        if not coords:
            u_pt = seed.NODES.get(int(row["u_node"]))
            v_pt = seed.NODES.get(int(row["v_node"]))
            coords = [list(u_pt), list(v_pt)] if (u_pt and v_pt) else []

        score = 0.0
        if damaged and coords:
            hits: list[float] = []
            for b_lat, b_lon, weight in damaged:
                min_d = min(haversine_m(b_lat, b_lon, float(pt[1]), float(pt[0])) for pt in coords)
                if min_d < prox_radius_m:
                    hits.append(weight * (1.0 - min_d / prox_radius_m))
            if is_osm_graph:
                hits.sort(reverse=True)
                score = sum(hits[:3]) / 2.5
            else:
                score = sum(hits)

        sat_debris = round(min(0.75, score * 0.45), 3)
        old_debris = round(float(row.get("debris_score", 0.0)), 3)
        flood_m = round(float(row.get("flood_depth_m", 0.0)), 3)
        haz_lvl = round(max(flood_m, sat_debris), 3)

        was_closed = bool(row.get("closure_state", False))
        prev_rest = str(row.get("vehicle_restriction", "none") or "none").lower()
        # Preserve explicit manual/flood closures; only promote blockage when satellite debris reaches threshold
        if (was_closed and prev_rest == "closed") or flood_m >= DB_FLOOD_THRESHOLD or sat_debris >= DB_FLOOD_THRESHOLD:
            closure = True
            capacity = 0.0
            restriction = "closed"
        elif haz_lvl >= DB_HAZARD_THRESHOLD:
            closure = False
            capacity = 0.5
            restriction = "high_clearance_only" if prev_rest in ("none", "closed", "") else prev_rest
        else:
            closure = False
            capacity = 1.0
            restriction = "none" if prev_rest in ("closed", "high_clearance_only", "") else prev_rest

        status_upper = db.derive_road_display_state(
            closure_state=closure,
            hazard_level=haz_lvl,
            capacity=capacity,
            vehicle_restriction=restriction,
            hazard_threshold=DB_HAZARD_THRESHOLD,
        ).upper()
        counts[status_upper] = counts.get(status_upper, 0) + 1

        if (
            sat_debris != old_debris
            or abs(haz_lvl - round(float(row.get("hazard_level", 0.0)), 3)) > 1e-6
            or status_upper != str(row.get("edge_status", "")).upper()
            or closure != was_closed
        ):
            db.update_road_edge(
                edge_id=eid,
                flood_depth_m=flood_m,
                debris_score=sat_debris,
                edge_status=status_upper,
                closure_state=closure,
                hazard_level=haz_lvl,
                capacity=capacity,
                vehicle_restriction=restriction,
                db_path=db_path,
            )
            changed = True

    return counts, changed


def save_graph(G: nx.Graph, path: Path | str = DEFAULT_GRAPH_PATH) -> Path:
    """Persist graph to GraphML using primitive node/edge attributes."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = nx.Graph(**{k: str(v) for k, v in G.graph.items()})
    for n, d in G.nodes(data=True):
        clean.add_node(
            n,
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            osmid=str(d.get("osmid", n)),
        )
    for u, v, d in G.edges(data=True):
        edge_attrs: dict[str, Any] = {
            "length_m": float(d.get("length_m", 1200.0)),
            "speed_kph": float(d.get("speed_kph", 48.0)),
            "travel_time_s": float(d.get("travel_time_s", 90.0)),
            "capacity": float(d.get("capacity", 1.0)),
            "hazard": float(d.get("hazard", 0.0)),
            "closed": int(d.get("closed", 0)),
            "closure_reported": int(d.get("closure_reported", 1)),
            "vehicle_restriction": str(d.get("vehicle_restriction", "none")),
            "highway": str(d.get("highway", "primary")),
            "osm_path": str(d.get("osm_path", edge_id(u, v))),
            "edge_id": str(d.get("edge_id", edge_id(u, v))),
        }
        if "db_edge_id" in d:
            edge_attrs["db_edge_id"] = str(d["db_edge_id"])
        if "flood_depth_m" in d:
            edge_attrs["flood_depth_m"] = float(d["flood_depth_m"])
        if "hazard_level" in d:
            edge_attrs["hazard_level"] = float(d["hazard_level"])
        if "geom_json" in d:
            edge_attrs["geom_json"] = str(d["geom_json"])
        clean.add_edge(u, v, **edge_attrs)
    nx.write_graphml(clean, out_path)
    return out_path


def load_graph(path: Path | str = DEFAULT_GRAPH_PATH) -> nx.Graph:
    """Load a GraphML road network and restore numeric attribute types."""
    G = nx.read_graphml(Path(path))
    for n, d in G.nodes(data=True):
        d["lat"], d["lon"] = float(d["lat"]), float(d["lon"])
        if n.startswith("n") and n[1:].isdigit():
            d.setdefault("node_index", int(n[1:]))
    for u, v, d in G.edges(data=True):
        for k in ("length_m", "speed_kph", "travel_time_s", "capacity", "hazard"):
            d[k] = float(d.get(k, 0.0))
        for k in ("closed", "closure_reported"):
            d[k] = int(float(d.get(k, 0)))
        if "flood_depth_m" in d:
            d["flood_depth_m"] = float(d["flood_depth_m"])
        if "hazard_level" in d:
            d["hazard_level"] = float(d["hazard_level"])
        if "closure_state" not in d and "flood_depth_m" in d:
            d["closure_state"] = int(d["closed"])
        d.setdefault("vehicle_restriction", "none")
        pid = edge_id(u, v)
        d["pair_id"] = pid
        d["edge_id"] = pid
    return G


def export_houston_graphml(
    db_path: str = db.DEFAULT_DB_PATH,
    out_path: Path | str = HOUSTON_GRAPH_PATH,
) -> Path:
    """Export the seeded Houston-East SQLite road network to GraphML."""
    G = build_graph_from_db(db_path=db_path)
    return save_graph(G, out_path)


def _finish_edge_attrs(G: nx.Graph) -> None:
    for u, v, d in G.edges(data=True):
        pid = edge_id(u, v)
        d["pair_id"] = pid
        d["edge_id"] = pid
        d["travel_time_s"] = d["length_m"] / (d["speed_kph"] / 3.6)


def _mark_hazards(G: nx.Graph, rng: random.Random) -> None:
    eb = nx.edge_betweenness_centrality(G, weight="travel_time_s")
    ranked = sorted(eb, key=eb.get, reverse=True)
    a, b = ranked[0]
    G[a][b]["closed"], G[a][b]["closure_reported"] = 1, 0
    others = [e for e in ranked[1:] if G.degree(e[0]) > 2 and G.degree(e[1]) > 2]
    for u, v in others[:2]:
        G[u][v]["hazard"] = round(rng.uniform(0.45, 0.7), 2)


def synthetic_graph(seed_val: int = 7) -> nx.Graph:
    """10-node fallback graph for isolated RL experiments; not real OSM data."""
    rng = random.Random(seed_val)
    pts = [
        (0.000, 0.000), (0.004, 0.010), (0.002, 0.021), (0.012, -0.004),
        (0.014, 0.008), (0.011, 0.020), (0.024, 0.001), (0.025, 0.012),
        (0.022, 0.024), (0.032, 0.009),
    ]
    G = nx.Graph(
        provenance="synthetic_placeholder",
        note="NOT real OSM data - replace via routing.graph --bbox or build_graph_from_db",
    )
    for i, (la, lo) in enumerate(pts):
        G.add_node(f"n{i}", node_index=i, lat=la, lon=lo, osmid="synthetic")
    links = [
        (0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (3, 6),
        (4, 7), (5, 8), (6, 7), (7, 8), (6, 9), (7, 9), (8, 9), (1, 3), (4, 8),
    ]
    hw = ["primary", "secondary", "residential", "tertiary"]
    for a, b in links:
        na, nb = G.nodes[f"n{a}"], G.nodes[f"n{b}"]
        length = haversine_m(na["lat"], na["lon"], nb["lat"], nb["lon"]) * 1.15
        h = rng.choice(hw)
        lanes = {"primary": 2, "secondary": 2, "tertiary": 1, "residential": 1}[h]
        G.add_edge(
            f"n{a}",
            f"n{b}",
            length_m=length,
            speed_kph=_SPEED[h],
            capacity=lanes * 900.0,
            hazard=0.0,
            closed=0,
            closure_reported=1,
            vehicle_restriction="none",
            highway=h,
            osm_path="synthetic",
        )
    _finish_edge_attrs(G)
    _mark_hazards(G, rng)
    return G


def _num(x: Any, default: float) -> float:
    if isinstance(x, list):
        x = x[0]
    try:
        return float(str(x).split()[0])
    except (TypeError, ValueError):
        return default


def _hw(x: Any) -> str:
    return x[0] if isinstance(x, list) else (x or "unclassified")


def _subsample_coords(coords: list[list[float]], max_points: int = 20) -> list[list[float]]:
    """Keep endpoints and evenly sample intermediate LineString coordinates."""
    if len(coords) <= max_points:
        return coords
    idxs = [int(round(i * (len(coords) - 1) / (max_points - 1))) for i in range(max_points)]
    deduped: list[int] = []
    for idx in idxs:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    if deduped[-1] != len(coords) - 1:
        deduped[-1] = len(coords) - 1
    return [coords[i] for i in deduped]


def graph_bounds(G: nx.Graph) -> tuple[float, float, float, float]:
    """Return `(min_lon, min_lat, max_lon, max_lat)` across all nodes in `G`."""
    lons = [float(d["lon"]) for _, d in G.nodes(data=True)]
    lats = [float(d["lat"]) for _, d in G.nodes(data=True)]
    if not lons or not lats:
        raise ValueError("Graph contains no node coordinates")
    return (min(lons), min(lats), max(lons), max(lats))


def build_from_osm(
    source: Any,
    n_nodes: int = N_NODES,
    seed_val: int = 7,
    mark_hazards: bool = True,
) -> nx.Graph:
    """Reduce a drivable OpenStreetMap network to at most `n_nodes` intersections."""
    import osmnx as ox  # lazy import so offline edge deployments do not require osmnx
    import tempfile

    tmp_osm_path: Path | None = None
    try:
        if isinstance(source, (tuple, list)) and len(source) == 4:
            n, s, e, w = (float(x) for x in source)
            M = ox.graph_from_bbox((w, s, e, n), network_type="drive")
        elif isinstance(source, str) and "<osm" in source[:256]:
            with tempfile.NamedTemporaryFile("w", suffix=".osm", encoding="utf-8", delete=False) as tf:
                tf.write(source)
                tmp_osm_path = Path(tf.name)
            M = ox.graph_from_xml(str(tmp_osm_path), simplify=True)
        elif isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() in {".osm", ".xml"}:
            M = ox.graph_from_xml(str(source), simplify=True)
        else:
            M = ox.graph_from_place(str(source), network_type="drive")
    finally:
        if tmp_osm_path is not None and tmp_osm_path.exists():
            tmp_osm_path.unlink(missing_ok=True)

    if M.number_of_nodes() < 2:
        raise ValueError(f"OSM area has only {M.number_of_nodes()} drivable nodes; need at least 2")

    M_strong = ox.truncate.largest_component(M, strongly=True)
    if M_strong.number_of_nodes() >= min(4, n_nodes, M.number_of_nodes()):
        M = M_strong
    else:
        M = ox.truncate.largest_component(M, strongly=False)

    U = nx.Graph()
    for node, d in M.nodes(data=True):
        U.add_node(node, lat=float(d["y"]), lon=float(d["x"]))

    for u, v, d in M.edges(data=True):
        if u == v:
            continue
        length_m = float(d.get("length", 1.0) or 1.0)
        if U.has_edge(u, v) and U[u][v]["length"] <= length_m:
            continue
        geom_obj = d.get("geometry")
        if geom_obj is not None and hasattr(geom_obj, "coords"):
            raw_pts = [[round(float(x), 6), round(float(y), 6)] for x, y in geom_obj.coords]
        else:
            raw_pts = [
                [round(float(M.nodes[u]["x"]), 6), round(float(M.nodes[u]["y"]), 6)],
                [round(float(M.nodes[v]["x"]), 6), round(float(M.nodes[v]["y"]), 6)],
            ]
        u_lon, u_lat = float(M.nodes[u]["x"]), float(M.nodes[u]["y"])
        if raw_pts and haversine_m(u_lat, u_lon, raw_pts[-1][1], raw_pts[-1][0]) < haversine_m(
            u_lat, u_lon, raw_pts[0][1], raw_pts[0][0]
        ):
            raw_pts = list(reversed(raw_pts))
        U.add_edge(
            u,
            v,
            length=length_m,
            highway=_hw(d.get("highway")),
            maxspeed=d.get("maxspeed"),
            lanes=d.get("lanes"),
            u_ref=u,
            coords=raw_pts,
        )

    isolated = [n for n in list(U.nodes) if U.degree(n) == 0]
    U.remove_nodes_from(isolated)
    if U.number_of_nodes() < 2:
        raise ValueError(f"OSM network has only {U.number_of_nodes()} connected nodes; need at least 2")

    target_nodes = max(2, min(int(n_nodes), U.number_of_nodes()))
    cand = [x for x in U if U.degree(x) >= 3]
    if len(cand) < target_nodes:
        cand = list(U.nodes)

    def dist(a: Any, b: Any) -> float:
        return haversine_m(U.nodes[a]["lat"], U.nodes[a]["lon"], U.nodes[b]["lat"], U.nodes[b]["lon"])

    rng = random.Random(seed_val)
    lat_c = sum(U.nodes[x]["lat"] for x in cand) / len(cand)
    lon_c = sum(U.nodes[x]["lon"] for x in cand) / len(cand)
    first = min(cand, key=lambda x: haversine_m(U.nodes[x]["lat"], U.nodes[x]["lon"], lat_c, lon_c))
    chosen = [first]
    while len(chosen) < target_nodes:
        nxt = max(
            (x for x in cand if x not in chosen),
            key=lambda x: (min(dist(x, c) for c in chosen), rng.random()),
        )
        chosen.append(nxt)
    name = {osm: f"n{i}" for i, osm in enumerate(chosen)}
    chosen_set = set(chosen)

    paths: dict[tuple[Any, Any], tuple[float, list[Any]]] = {}
    for i, a in enumerate(chosen):
        lengths, pth = nx.single_source_dijkstra(U, a, weight="length")
        for b in chosen[i + 1 :]:
            if b not in pth:
                continue
            p = pth[b]
            if any(x in chosen_set for x in p[1:-1]):
                continue
            paths[(a, b)] = (lengths[b], p)

    G = nx.Graph(provenance="osm", note="reduced from OpenStreetMap drivable network")
    for i, osm in enumerate(chosen):
        G.add_node(name[osm], node_index=i, lat=U.nodes[osm]["lat"], lon=U.nodes[osm]["lon"], osmid=str(osm))

    def add(a: Any, b: Any) -> None:
        length, p = paths[(a, b)]
        segs = [U[p[i]][p[i + 1]] for i in range(len(p) - 1)]
        longest = max(segs, key=lambda s: s["length"])
        hw = longest["highway"]
        speeds = [_num(s["maxspeed"], _SPEED.get(s["highway"], 35)) for s in segs]
        t = sum(s["length"] / (sp / 3.6) for s, sp in zip(segs, speeds))
        speed = length / t * 3.6 if t > 0 else 35.0
        lanes = min(_num(s["lanes"], 1) for s in segs)
        stitched: list[list[float]] = []
        for seg_idx in range(len(p) - 1):
            x = p[seg_idx]
            y = p[seg_idx + 1]
            seg = U[x][y]
            seg_pts = list(seg["coords"]) if seg["u_ref"] == x else list(reversed(seg["coords"]))
            if not stitched:
                stitched.extend(seg_pts)
            else:
                stitched.extend(seg_pts[1:] if seg_pts and seg_pts[0] == stitched[-1] else seg_pts)
        coords = _subsample_coords(stitched, max_points=20)
        G.add_edge(
            name[a],
            name[b],
            length_m=length,
            speed_kph=speed,
            capacity=lanes * 900.0,
            hazard=0.0,
            closed=0,
            closure_reported=1,
            vehicle_restriction="none",
            highway=hw,
            osm_path="|".join(str(x) for x in p),
            geom_json={"type": "LineString", "coordinates": coords},
        )

    for a in chosen:
        mine = sorted(((v[0], k) for k, v in paths.items() if a in k))[:3]
        for _, k in mine:
            if not G.has_edge(name[k[0]], name[k[1]]):
                add(*k)
    while not nx.is_connected(G):
        comps = [set(c) for c in nx.connected_components(G)]
        best = None
        for (a, b), (length, _) in paths.items():
            ca = next(c for c in comps if name[a] in c)
            if name[b] not in ca and (best is None or length < best[0]):
                best = (length, (a, b))
        if best is None:
            raise ValueError("could not connect the reduced graph")
        add(*best[1])

    _finish_edge_attrs(G)
    if mark_hazards:
        _mark_hazards(G, rng)
    return G


def sync_osm_graph_to_db(
    source: Any,
    db_path: str = db.DEFAULT_DB_PATH,
    n_nodes: int = N_NODES,
    mark_hazards: bool = False,
) -> tuple[float, float, float, float]:
    """Build an OSM road graph and persist it to `road_edges`, returning `(min_lon, min_lat, max_lon, max_lat)`."""
    G = build_from_osm(source, n_nodes=n_nodes, mark_hazards=mark_hazards)
    edge_rows: list[dict[str, Any]] = []
    for idx, (u, v, d) in enumerate(sorted(G.edges(data=True), key=lambda item: edge_id(item[0], item[1])), start=1):
        u_idx = int(G.nodes[u]["node_index"])
        v_idx = int(G.nodes[v]["node_index"])
        u_pt = [round(float(G.nodes[u]["lon"]), 6), round(float(G.nodes[u]["lat"]), 6)]
        v_pt = [round(float(G.nodes[v]["lon"]), 6), round(float(G.nodes[v]["lat"]), 6)]
        raw_geom = d.get("geom_json") or {"type": "LineString", "coordinates": [u_pt, v_pt]}
        coords = list(raw_geom.get("coordinates") or [u_pt, v_pt])
        if len(coords) >= 2:
            d_first_u = haversine_m(u_pt[1], u_pt[0], float(coords[0][1]), float(coords[0][0]))
            d_last_u = haversine_m(u_pt[1], u_pt[0], float(coords[-1][1]), float(coords[-1][0]))
            if d_last_u < d_first_u:
                coords.reverse()
        closed = bool(int(d.get("closed", 0)))
        hazard = round(float(d.get("hazard", 0.0)), 3)
        if closed:
            cap = 0.0
            status = "BLOCKED"
        elif hazard >= DB_HAZARD_THRESHOLD:
            cap = 0.5
            status = "HAZARDOUS"
        else:
            cap = 1.0
            status = "OPEN"
        edge_rows.append(
            {
                "edge_id": f"OSM-{idx:02d}",
                "u_node": u_idx,
                "v_node": v_idx,
                "geometry": {"type": "LineString", "coordinates": coords},
                "length_m": round(float(d.get("length_m", 1200.0)), 2),
                "base_travel_sec": round(float(d.get("travel_time_s", 90.0)), 2),
                "flood_depth_m": hazard if hazard > 0.0 else 0.0,
                "debris_score": 0.0,
                "closure_state": closed,
                "hazard_level": hazard,
                "capacity": cap,
                "vehicle_restriction": str(d.get("vehicle_restriction", "none") or "none"),
                "edge_status": status,
            }
        )
    db.replace_road_edges(edge_rows, db_path=db_path)
    return graph_bounds(G)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or export the Portion 2 routing graph")
    ap.add_argument("--houston-db", nargs="?", const=db.DEFAULT_DB_PATH, help="export Houston-East SQLite graph to GraphML")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("NORTH", "SOUTH", "EAST", "WEST"))
    ap.add_argument("--place", help="OSM place name")
    ap.add_argument("--osm-file", help="offline .osm XML file")
    ap.add_argument("--synthetic", action="store_true", help="10-node placeholder graph")
    ap.add_argument("--out", default=str(HOUSTON_GRAPH_PATH))
    args = ap.parse_args()

    if args.houston_db:
        out = export_houston_graphml(db_path=args.houston_db, out_path=args.out)
        G = load_graph(out)
    elif args.synthetic:
        G = synthetic_graph()
        out = save_graph(G, args.out)
    elif args.bbox:
        G = build_from_osm(tuple(args.bbox))
        out = save_graph(G, args.out)
    elif args.place:
        G = build_from_osm(args.place)
        out = save_graph(G, args.out)
    elif args.osm_file:
        G = build_from_osm(args.osm_file)
        out = save_graph(G, args.out)
    else:
        ap.error("give --houston-db, --bbox, --place, --osm-file or --synthetic")
        return

    print(f"saved {out}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, provenance={G.graph.get('provenance')}")


if __name__ == "__main__":
    main()

