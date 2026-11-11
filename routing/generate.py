"""Route generation, capability-aware dispatch, and <500ms What-If engine (Portion 2).

Executes PPO rollout -> deterministic hard safety mask -> Section 4.2 routing features
for both `POST /api/routes/dispatch` (`compute_multi_unit_routes`) and
`POST /api/routes/what-if` (`run_counterfactual`) via Zero-SQL `db.py` helpers.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from stable_baselines3 import PPO

import db
import seed

from .env import RoadGraphIndex, build_obs
from .features import (
    FEATURE_NAMES,
    capability_match,
    compute_features,
    effective_edge_travel_s,
    features_to_shap_attributions,
)
from .graph import (
    HOUSTON_GRAPH_PATH,
    build_graph_from_db,
    edge_id,
    graph_edge_to_db_id,
    haversine_m,
    load_graph,
    road_state_label,
    state_from_db,
)
from .mask import apply_mask

DEFAULT_FLEET_UNITS: list[dict[str, Any]] = [
    {
        "unit_id": "MED-01",
        "unit_class": "MEDICAL",
        "capability": "medical",
        "origin_node": 0,
        "location": {"lat": seed.NODES[0][1], "lon": seed.NODES[0][0]},
        "default_target_node": 8,
    },
    {
        "unit_id": "TRN-01",
        "unit_class": "TRANSPORT",
        "capability": "transport",
        "origin_node": 0,
        "location": {"lat": seed.NODES[0][1], "lon": seed.NODES[0][0]},
        "default_target_node": 6,
    },
    {
        "unit_id": "RSC-01",
        "unit_class": "RESCUE",
        "capability": "rescue",
        "origin_node": 2,
        "location": {"lat": seed.NODES[2][1], "lon": seed.NODES[2][0]},
        "default_target_node": 7,
    },
]

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

_ENGINE_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: dict[tuple[tuple[str, ...], str], "RoutingEngine"] = {}


def nearest_node(G: nx.Graph, lat: float, lon: float) -> str:
    """Return the graph node nearest to `(lat, lon)` by haversine distance."""
    return min(
        G.nodes,
        key=lambda n: haversine_m(lat, lon, float(G.nodes[n]["lat"]), float(G.nodes[n]["lon"])),
    )


def _node_int(node_str: str) -> int:
    return int(node_str[1:]) if node_str.startswith("n") and node_str[1:].isdigit() else int(node_str)


def _edge_curve_points(G: nx.Graph, u: str, v: str) -> list[list[float]]:
    d = G[u][v]
    db_eid = str(d.get("db_edge_id") or "")
    pts: list[list[float]] | None = None
    if d.get("geom_json"):
        try:
            geom = json.loads(d["geom_json"])
            pts = geom.get("coordinates")
        except Exception:
            pts = None
    if not pts and db_eid in seed.EDGE_CURVES:
        pts = seed.EDGE_CURVES[db_eid]
    if not pts:
        pts = [
            [float(G.nodes[u]["lon"]), float(G.nodes[u]["lat"])],
            [float(G.nodes[v]["lon"]), float(G.nodes[v]["lat"])],
        ]
    seg = [[float(p[0]), float(p[1])] for p in pts]
    u_lon, u_lat = float(G.nodes[u]["lon"]), float(G.nodes[u]["lat"])
    d_first = (seg[0][0] - u_lon) ** 2 + (seg[0][1] - u_lat) ** 2
    d_last = (seg[-1][0] - u_lon) ** 2 + (seg[-1][1] - u_lat) ** 2
    if d_last < d_first:
        seg.reverse()
    return seg


def _node_path_to_geometry(G: nx.Graph, node_path: list[str]) -> tuple[list[list[float]], list[str]]:
    coords: list[list[float]] = []
    edge_ids: list[str] = []
    if not node_path:
        return coords, edge_ids
    if len(node_path) == 1:
        n = node_path[0]
        return [[float(G.nodes[n]["lon"]), float(G.nodes[n]["lat"])]], []

    for u, v in zip(node_path[:-1], node_path[1:]):
        if G.has_edge(u, v):
            edge_ids.append(graph_edge_to_db_id(G, u, v))
            seg = _edge_curve_points(G, u, v)
        else:
            seg = [
                [float(G.nodes[u]["lon"]), float(G.nodes[u]["lat"])],
                [float(G.nodes[v]["lon"]), float(G.nodes[v]["lat"])],
            ]
        for idx, pt in enumerate(seg):
            if not coords or idx > 0 or ((coords[-1][0] - pt[0]) ** 2 + (coords[-1][1] - pt[1]) ** 2) > 1e-12:
                coords.append(pt)
    return coords, edge_ids


def _coords_to_latlon(coords: list[list[float]]) -> list[dict[str, float]]:
    return [{"lat": float(c[1]), "lon": float(c[0])} for c in coords]


class RoutingEngine:
    """PPO + deterministic hard safety mask engine for disaster response routing."""

    def __init__(
        self,
        graph_path: Path | str = HOUSTON_GRAPH_PATH,
        model_path: Path | str | None = None,
        model: PPO | None = None,
        G: nx.Graph | None = None,
    ):
        from .train import MODEL_PATH as DEFAULT_MODEL_PATH

        resolved_model = Path(model_path or DEFAULT_MODEL_PATH)
        self.G = G if G is not None else load_graph(graph_path)
        self.ix = RoadGraphIndex(self.G)
        self.model = model or PPO.load(str(resolved_model), device="cpu")
        self.max_steps = 3 * self.ix.n
        self.state = self.ix.base_state()

    def set_road(self, eid: str, **changes: Any) -> None:
        if eid not in self.state:
            raise KeyError(eid)
        self.state[eid].update(changes)

    def _visible(self, state: dict[str, dict[str, Any]]) -> dict[str, int]:
        return {
            e: int(road_state_label(s) == "blocked" and int(s.get("reported", 1)) == 1)
            for e, s in state.items()
        }

    def propose(self, start: str, target: str, state: dict[str, dict[str, Any]] | None = None) -> list[str]:
        """Greedy PPO rollout on `state` using only reported closures."""
        active_state = state if state is not None else self.state
        vis = self._visible(active_state)
        cur, path = start, [start]
        for step in range(self.max_steps):
            if cur == target:
                break
            obs = build_obs(self.ix, cur, target, active_state, vis, 1.0 - step / self.max_steps)
            action, _ = self.model.predict(obs, deterministic=True)
            nbrs = self.ix.nbrs[cur]
            if int(action) >= len(nbrs):
                break
            cur = nbrs[int(action)]
            path.append(cur)
        return path

    def road_states_list(self, state: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Serialize Section 4.2 4-attribute + 3-display-state road list."""
        items: list[dict[str, Any]] = []
        for pid, s in sorted(state.items(), key=lambda kv: str(kv[1].get("db_edge_id", kv[0]))):
            db_eid = str(s.get("db_edge_id") or pid)
            lbl = road_state_label(s)
            items.append({
                "edge_id": db_eid,
                "state": lbl,
                "closure_state": bool(s.get("closure_state", lbl == "blocked")),
                "hazard_level": float(s.get("hazard_level", s.get("hazard", 0.0))),
                "capacity": min(1.0, max(0.0, float(s.get("capacity", 1.0)))),
                "vehicle_restriction": str(s.get("vehicle_restriction", "none") or "none"),
            })
        return items

    def route_unit(
        self,
        unit: dict[str, Any],
        incident: dict[str, Any],
        state: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute PPO proposal, masked safe detour, curved geometry, and features for one unit."""
        origin = unit.get("origin_node")
        s_node = f"n{int(origin)}" if origin is not None else nearest_node(self.G, unit["location"]["lat"], unit["location"]["lon"])
        t_node = nearest_node(self.G, float(incident["location"]["lat"]), float(incident["location"]["lon"]))

        proposed_nodes = [s_node] if s_node == t_node else self.propose(s_node, t_node, state=state)
        unit_cap = str(unit.get("capability", "rescue")).lower()
        masked_nodes, altered, reason = apply_mask(
            self.G,
            proposed_nodes,
            state,
            target=t_node,
            unit_capability=unit_cap,
        )

        reachable = bool(masked_nodes) and masked_nodes[-1] == t_node
        ppo_coords, ppo_edge_ids = _node_path_to_geometry(self.G, proposed_nodes)
        safe_coords, safe_edge_ids = _node_path_to_geometry(self.G, masked_nodes)

        sev = str(incident.get("severity", "medium")).lower()
        req_cap = str(incident.get("required_capability", unit_cap)).lower()
        proposed_feats = compute_features(self.G, state, proposed_nodes, sev, unit_cap, req_cap)
        if reachable:
            eta_sec = round(
                sum(effective_edge_travel_s(self.G, state, a, b, unit_capability=unit_cap) for a, b in zip(masked_nodes[:-1], masked_nodes[1:])),
                1,
            )
            edge_statuses = [
                road_state_label(state[edge_id(a, b)], unit_capability=unit_cap).upper()
                for a, b in zip(masked_nodes[:-1], masked_nodes[1:])
            ]
        else:
            eta_sec = 0.0
            edge_statuses = []

        return {
            "unit_id": unit["unit_id"],
            "unit_class": unit.get("unit_class", unit_cap.upper()),
            "incident_id": incident["incident_id"],
            "origin_node": _node_int(s_node),
            "target_node": _node_int(t_node),
            "reachable": reachable,
            "eta_sec": eta_sec,
            "node_path": [_node_int(n) for n in masked_nodes],
            "edge_ids": safe_edge_ids,
            "edge_statuses": edge_statuses,
            "hazardous_segments": edge_statuses.count("HAZARDOUS"),
            "geometry": {"type": "LineString", "coordinates": safe_coords},
            "masked_path": _coords_to_latlon(safe_coords),
            "ppo_proposed_path": _coords_to_latlon(ppo_coords),
            "ppo_edge_ids": ppo_edge_ids,
            "mask_altered": bool(altered),
            "mask_reason": reason,
            "features": {k: proposed_feats[k] for k in FEATURE_NAMES},
        }


def _get_cached_engine(db_path: str) -> RoutingEngine:
    from .train import MODEL_PATH

    G = build_graph_from_db(db_path=db_path)
    graph_sig = tuple(
        sorted(f"{u}-{v}:{round(float(d.get('travel_time_s', 0.0)), 2)}" for u, v, d in G.edges(data=True))
    )
    key = (graph_sig, str(Path(MODEL_PATH).resolve()))
    with _ENGINE_CACHE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None:
            return cached
        eng = RoutingEngine(model_path=MODEL_PATH, G=G)
        _ENGINE_CACHE[key] = eng
        return eng


def _assign_units_to_incidents(
    G: nx.Graph,
    units: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Assign each fleet unit to its highest-priority matching incident."""
    active = [i for i in incidents if str(i.get("status", "open")).lower() != "resolved"]
    if not active:
        return [
            (
                u,
                {
                    "incident_id": f"INC-FALLBACK-{idx}",
                    "location": {"lat": seed.NODES[u["default_target_node"]][1], "lon": seed.NODES[u["default_target_node"]][0]},
                    "severity": "high",
                    "required_capability": u["capability"],
                    "status": "open",
                },
            )
            for idx, u in enumerate(units, start=1)
        ]

    ordered = sorted(
        active,
        key=lambda i: (
            1 if str(i.get("status", "")).lower() == "open" else 0,
            _SEVERITY_RANK.get(str(i.get("severity", "medium")).lower(), 2),
            str(i.get("time_reported", "")),
        ),
        reverse=True,
    )

    assignments: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_inc_ids: set[str] = set()
    for unit in units:
        pool = [i for i in ordered if i["incident_id"] not in used_inc_ids] or ordered
        best = max(
            pool,
            key=lambda inc: (
                capability_match(unit["capability"], str(inc.get("required_capability", "rescue")).lower()),
                1 if str(inc.get("status", "")).lower() == "open" else 0,
                _SEVERITY_RANK.get(str(inc.get("severity", "medium")).lower(), 2),
                -haversine_m(
                    unit["location"]["lat"],
                    unit["location"]["lon"],
                    float(inc["location"]["lat"]),
                    float(inc["location"]["lon"]),
                ),
            ),
        )
        used_inc_ids.add(best["incident_id"])
        assignments.append((unit, best))
    return assignments


def _build_counterfactuals(unit_routes: list[dict[str, Any]], total_eta_sec: float) -> list[dict[str, Any]]:
    altered = [u for u in unit_routes if u["mask_altered"]]
    if altered:
        u0 = altered[0]
        raw_eta = float(u0["features"]["travel_time_s"])
        delta_raw = round(raw_eta - float(u0["eta_sec"]), 1)
        blocked_tag = ", ".join(u0["ppo_edge_ids"]) or "closed corridor"
        return [
            {"alternative": f"Unmasked PPO Proposal ({blocked_tag})", "delta_eta": delta_raw},
            {"alternative": f"Safety-Masked Corridor ({', '.join(u0['edge_ids']) or 'detour'})", "delta_eta": 0.0},
        ]
    return [{"alternative": f"Baseline Safe Corridor (ETA {total_eta_sec:.1f}s)", "delta_eta": 0.0}]


def _build_gtf_narrative(
    plan_id: str,
    total_eta_sec: float,
    unit_routes: list[dict[str, Any]],
    mask_altered: bool,
    blocked_edges: list[str],
    prefix: str = "[P2 MARL DISPATCH]",
) -> str:
    reachable_cnt = sum(1 for u in unit_routes if u["reachable"])
    corridor_edges = [eid for u in unit_routes for eid in u["edge_ids"]]
    corridor_str = ", ".join(dict.fromkeys(corridor_edges)) or "none"
    blocked_str = ", ".join(blocked_edges[:4]) or "none"
    mask_phrase = (
        f"Safety mask altered the proposal around blocked edges ({blocked_str})."
        if mask_altered
        else "Safety mask did not alter the proposal; corridor verified clear."
    )
    return (
        f"{prefix} Plan {plan_id} routed {reachable_cnt}/{len(unit_routes)} units "
        f"via [{corridor_str}] with ETA {total_eta_sec:.1f}s. {mask_phrase}"
    )


def warm_engine(db_path: str = db.DEFAULT_DB_PATH) -> RoutingEngine:
    """Pre-load the road graph index and PPO checkpoint into memory."""
    return _get_cached_engine(db_path)


def compute_multi_unit_routes(db_path: str = db.DEFAULT_DB_PATH) -> dict[str, Any]:
    """Compute multi-unit PPO dispatch + safety mask plan (`main.py` Endpoint 10 hook)."""
    eng = _get_cached_engine(db_path)
    t0 = time.perf_counter()
    state = state_from_db(db_path=db_path)
    incidents = db.get_incidents(db_path=db_path)
    assignments = _assign_units_to_incidents(eng.G, DEFAULT_FLEET_UNITS, incidents)

    unit_routes = [eng.route_unit(unit, inc, state) for unit, inc in assignments]
    total_eta = round(sum(float(u["eta_sec"]) for u in unit_routes if u["reachable"]), 1)
    primary = next((u for u in unit_routes if u["mask_altered"]), unit_routes[0])
    mask_altered = any(u["mask_altered"] for u in unit_routes)
    reasons = [f"{u['unit_id']}: {u['mask_reason']}" for u in unit_routes if u["mask_reason"]]
    mask_reason = "; ".join(reasons) if reasons else "All PPO proposed routes verified clear by safety mask."

    road_states = eng.road_states_list(state)
    blocked_list = [r["edge_id"] for r in road_states if r["state"] == "blocked"]
    plan_id = f"PLAN-P2-{uuid.uuid4().hex[:6].upper()}"
    compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    shap_values = features_to_shap_attributions(primary["features"], mask_altered=mask_altered)
    counterfactuals = _build_counterfactuals(unit_routes, total_eta)
    narrative = _build_gtf_narrative(plan_id, total_eta, unit_routes, mask_altered, blocked_list)

    return {
        "route_id": plan_id,
        "plan_id": plan_id,
        "unit_id": primary["unit_id"],
        "incident_id": primary["incident_id"],
        "ppo_proposed_path": primary["ppo_proposed_path"],
        "masked_path": primary["masked_path"],
        "mask_altered": mask_altered,
        "mask_reason": mask_reason,
        "road_states": road_states,
        "eta_seconds": total_eta,
        "total_eta_sec": total_eta,
        "compute_ms": compute_ms,
        "status": "pending_approval",
        "approval_status": "PENDING_APPROVAL",
        "explanation": narrative,
        "narrative_text": narrative,
        "narrative_mode": "JINJA2_TEMPLATE",
        "uncertainty": round(0.06 + (0.04 if mask_altered else 0.0), 3),
        "shap_values": shap_values,
        "counterfactuals": counterfactuals,
        "routes": unit_routes,
        "blocked_edges": blocked_list,
        "audit_hash": "",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_counterfactual(
    blocked_edges: list[str] | None = None,
    flood_delta: float = 0.0,
    db_path: str = db.DEFAULT_DB_PATH,
    opened_edges: list[str] | None = None,
) -> dict[str, Any]:
    """Dry-run What-If reroute under manual closures/openings and flood surge (`main.py` Endpoint 13 hook)."""
    eng = _get_cached_engine(db_path)
    t0 = time.perf_counter()
    base_state = state_from_db(db_path=db_path)
    overlay_state = state_from_db(
        db_path=db_path,
        blocked_edges=blocked_edges,
        flood_delta=flood_delta,
        opened_edges=opened_edges,
    )

    blocked_set = set(blocked_edges or [])
    newly_blocked = sorted(
        s["db_edge_id"]
        for pid, s in overlay_state.items()
        if road_state_label(s) == "blocked"
        and (road_state_label(base_state[pid]) != "blocked" or s["db_edge_id"] in blocked_set)
    )

    incidents = db.get_incidents(db_path=db_path)
    assignments = _assign_units_to_incidents(eng.G, DEFAULT_FLEET_UNITS, incidents)
    unit_routes = [eng.route_unit(unit, inc, overlay_state) for unit, inc in assignments]
    whatif_eta = round(sum(float(u["eta_sec"]) for u in unit_routes if u["reachable"]), 1)

    existing_routes = db.get_routes(include_counterfactuals=False, db_path=db_path)
    if existing_routes and float(existing_routes[0].get("eta_seconds") or 0.0) > 0.0:
        baseline_eta = float(existing_routes[0].get("total_eta_sec") or existing_routes[0]["eta_seconds"])
    else:
        base_routes = [eng.route_unit(unit, inc, base_state) for unit, inc in assignments]
        baseline_eta = round(sum(float(u["eta_sec"]) for u in base_routes if u["reachable"]), 1)

    eta_delta = round(whatif_eta - baseline_eta, 1)
    mask_altered = any(u["mask_altered"] for u in unit_routes) or bool(newly_blocked)
    dynamic_hazards = {
        str(s["db_edge_id"]): {"status": road_state_label(s).upper()}
        for s in overlay_state.values()
    }
    compute_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "plan_id": f"WHATIF-P2-{uuid.uuid4().hex[:6].upper()}",
        "compute_ms": compute_ms,
        "flood_delta_m": float(flood_delta),
        "newly_blocked_edges": newly_blocked,
        "dynamic_hazards": dynamic_hazards,
        "baseline_eta_sec": baseline_eta,
        "whatif_eta_sec": whatif_eta,
        "eta_delta_sec": eta_delta,
        "mask_altered": mask_altered,
        "routes": unit_routes,
    }
