"""Deterministic safety mask for Portion 2 routing (not learned).

Walks `ppo_proposed_path` edge by edge and replaces any blocked, flooded,
zero-capacity, or vehicle-restricted segment (or incomplete PPO rollout) with
the shortest passable detour by `travel_time_s`. Returns `(masked_path, changed, reason)`.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .graph import DB_FLOOD_THRESHOLD, FLOOD_THRESHOLD, edge_id, graph_edge_to_db_id, is_blocked, road_state_label


def _edge_attrs(G: nx.Graph, state: dict[str, dict[str, Any]] | None, a: str, b: str) -> dict[str, Any]:
    """Resolve current edge attributes by merging graph attributes with live `state`."""
    d = dict(G[a][b])
    if state is not None:
        pid = edge_id(a, b)
        db_eid = str(d.get("db_edge_id") or "")
        if pid in state:
            d.update(state[pid])
        elif db_eid and db_eid in state:
            d.update(state[db_eid])
    return d


def safe_view(
    G: nx.Graph,
    state: dict[str, dict[str, Any]] | None,
    unit_capability: str | None = None,
) -> nx.Graph:
    """Return a subgraph containing only passable (`open` or `hazardous`) road edges."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if not is_blocked(_edge_attrs(G, state, u, v), unit_capability=unit_capability):
            H.add_edge(u, v, **d)
    return H


def _shortest(H: nx.Graph, a: str, b: str) -> list[str] | None:
    try:
        return nx.shortest_path(H, a, b, weight="travel_time_s")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _drop_loops(path: list[str]) -> list[str]:
    out: list[str] = []
    for node in path:
        if node in out:
            out = out[: out.index(node) + 1]
        else:
            out.append(node)
    return out


def _describe_block_reason(G: nx.Graph, attrs: dict[str, Any], a: str, b: str, unit_capability: str | None) -> str:
    label_id = graph_edge_to_db_id(G, a, b)
    restriction = str(attrs.get("vehicle_restriction", "none") or "none").strip().lower()
    if bool(attrs.get("closure_state", False)) or int(attrs.get("closed", 0)) == 1 or restriction == "closed":
        return f"edge {label_id} is closed"
    if (
        float(attrs.get("flood_depth_m", 0.0)) >= DB_FLOOD_THRESHOLD
        or float(attrs.get("hazard_level", 0.0)) >= DB_FLOOD_THRESHOLD
        or float(attrs.get("hazard", 0.0)) >= FLOOD_THRESHOLD
    ):
        return f"edge {label_id} is flooded"
    if float(attrs.get("capacity", 1.0)) <= 0.0:
        return f"edge {label_id} has zero capacity"
    if restriction == "no_heavy" and unit_capability == "transport":
        return f"edge {label_id} restricts heavy transport"
    return f"edge {label_id} is {road_state_label(attrs, unit_capability=unit_capability)}"


def apply_mask(
    G: nx.Graph,
    path: list[str],
    state: dict[str, dict[str, Any]] | None = None,
    target: str | None = None,
    unit_capability: str | None = None,
) -> tuple[list[str], bool, str]:
    """Filter `path` against hard safety constraints and repair broken/incomplete paths."""
    if not path:
        return [], False, ""
    target = target or path[-1]
    H = safe_view(G, state, unit_capability=unit_capability)
    out, reasons, i = [path[0]], [], 0

    while i < len(path) - 1:
        a, b = path[i], path[i + 1]
        if not G.has_edge(a, b):
            seg = _shortest(H, out[-1], b)
            why = f"edge {a}->{b} does not exist in the road graph"
        else:
            attrs = _edge_attrs(G, state, a, b)
            if not is_blocked(attrs, unit_capability=unit_capability):
                out.append(b)
                i += 1
                continue
            seg = _shortest(H, out[-1], b)
            why = _describe_block_reason(G, attrs, a, b, unit_capability)

        if seg is None:
            seg = _shortest(H, out[-1], target)
            if seg is None:
                return [], True, f"{why}; no safe route to {target} exists"
            reasons.append(f"{why}; detour to {b} impossible, rerouted straight to {target}")
            out = _drop_loops(out + seg[1:])
            return out, True, "; ".join(reasons)

        detour_labels = [graph_edge_to_db_id(G, u, v) for u, v in zip(seg[:-1], seg[1:])]
        reasons.append(f"{why}; replaced by detour {'->'.join(detour_labels)}")
        out = _drop_loops(out + seg[1:])
        i += 1

    if out[-1] != target:
        seg = _shortest(H, out[-1], target)
        if seg is None:
            return [], True, f"PPO path ended at {out[-1]}; no safe route to {target}"
        reasons.append(f"PPO path ended at {out[-1]}, not at {target}; completed with safe shortest path")
        out = _drop_loops(out + seg[1:])

    changed = out != list(path)
    return out, changed, "; ".join(reasons) if changed else ""

