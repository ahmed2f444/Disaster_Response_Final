"""Gymnasium environment for Portion 2 MARL/PPO disaster routing.

State   : unit current node, incident node, local neighbor edge slots, and global
          road hazard + visible closure vectors.
Action  : Discrete(max_degree) selecting the k-th neighbor leaving the current node.
Reward  : +ARRIVE_REWARD on reaching target, minus travel time (minutes), hazard
          exposure, and CLOSED_PENALTY for entering a blocked road.

Trained with randomized edge closures/hazards and partial closure observability
(`visible_closure_p`), matching the safety-mask separation in Unified Doc Stage 5.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from .graph import edge_id, is_blocked

ARRIVE_REWARD = 20.0
CLOSED_PENALTY = 8.0
HAZARD_WEIGHT = 4.0
INVALID_PENALTY = 1.0
SLOT_FEATS = 5


def _node_sort_key(node_id: Any) -> tuple[int, int | str]:
    text = str(node_id)
    if text.startswith("n") and text[1:].isdigit():
        return (0, int(text[1:]))
    if text.isdigit():
        return (0, int(text))
    return (1, text)


class RoadGraphIndex:
    """Static, array-friendly view of the road graph shared by training and inference."""

    def __init__(self, G: nx.Graph):
        self.G = G
        self.nodes = sorted(G.nodes, key=_node_sort_key)
        self.n = len(self.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.edges = sorted(edge_id(u, v) for u, v in G.edges)
        self.eidx = {e: i for i, e in enumerate(self.edges)}
        self.nbrs = {n: sorted(G.neighbors(n), key=_node_sort_key) for n in self.nodes}
        self.max_deg = max(len(v) for v in self.nbrs.values())
        tt = dict(nx.all_pairs_dijkstra_path_length(G, weight="travel_time_s"))
        self.free_dist = np.array([[tt[a][b] for b in self.nodes] for a in self.nodes], dtype=np.float32)
        self.tt_max = max(float(d.get("travel_time_s", 1.0)) for _, _, d in G.edges(data=True)) or 1.0
        self.dist_max = float(self.free_dist.max()) or 1.0
        self.obs_dim = 2 * self.n + self.max_deg * SLOT_FEATS + 2 * len(self.edges) + 2

    def base_state(self) -> dict[str, dict[str, Any]]:
        """Build `{pair_id: normalized_rl_state}` from graph edge attributes."""
        out: dict[str, dict[str, Any]] = {}
        for u, v, d in self.G.edges(data=True):
            closed = int(d.get("closed", 0))
            cap = float(d.get("capacity", 1.0))
            out[edge_id(u, v)] = {
                "hazard": float(d.get("hazard", 0.0)),
                "closed": closed,
                "capacity": 0.0 if closed else (cap if cap > 0.0 else 1.0),
                "reported": int(d.get("closure_reported", 1)),
                "vehicle_restriction": str(d.get("vehicle_restriction", "none") or "none"),
            }
        return out


def state_blocked(s: dict[str, Any]) -> bool:
    """Return True if an edge state entry is impassable."""
    return is_blocked(s)


def build_obs(
    ix: RoadGraphIndex,
    cur: str,
    tgt: str,
    state: dict[str, dict[str, Any]],
    visible: dict[str, int],
    steps_left_frac: float,
) -> np.ndarray:
    """Construct the fixed-dimension float32 observation vector for PPO."""
    n = ix.n
    o = np.zeros(ix.obs_dim, dtype=np.float32)
    o[ix.idx[cur]] = 1.0
    o[n + ix.idx[tgt]] = 1.0
    t = ix.idx[tgt]
    base = 2 * n

    for k, nb in enumerate(ix.nbrs[cur]):
        e = edge_id(cur, nb)
        d = ix.G[cur][nb]
        j = base + k * SLOT_FEATS
        o[j] = 1.0
        o[j + 1] = float(d["travel_time_s"]) / ix.tt_max
        o[j + 2] = float(state[e].get("hazard", 0.0))
        o[j + 3] = float(visible[e])
        o[j + 4] = float(ix.free_dist[ix.idx[nb], t]) / ix.dist_max

    base += ix.max_deg * SLOT_FEATS
    for e, i in ix.eidx.items():
        o[base + 2 * i] = float(state[e].get("hazard", 0.0))
        o[base + 2 * i + 1] = float(visible[e])

    o[-2] = float(ix.free_dist[ix.idx[cur], t]) / ix.dist_max
    o[-1] = float(steps_left_frac)
    return o


class RoutingEnv(gym.Env):
    """Single-agent road-graph navigation environment for PPO policy training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        G: nx.Graph,
        max_steps: int | None = None,
        visible_closure_p: float = 0.6,
        seed: int | None = None,
        randomize: bool = True,
    ):
        super().__init__()
        self.ix = RoadGraphIndex(G)
        self.max_steps = max_steps or 3 * self.ix.n
        self.visible_closure_p = visible_closure_p
        self.randomize = randomize
        self.rng = random.Random(seed)
        self.action_space = spaces.Discrete(self.ix.max_deg)
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(self.ix.obs_dim,), dtype=np.float32)
        self._base = self.ix.base_state()
        self.cur: str | None = None
        self.tgt: str | None = None
        self.state: dict[str, dict[str, Any]] = {}
        self.visible: dict[str, int] = {}
        self.steps = 0
        self.time_s = 0.0

    def _sample_state(self) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        st = {e: dict(s) for e, s in self._base.items()}
        if self.randomize:
            for s in st.values():
                s["closed"] = int(self.rng.random() < 0.10)
                s["hazard"] = round(self.rng.uniform(0.2, 0.95), 2) if self.rng.random() < 0.25 else 0.0
                s["capacity"] = 0.0 if s["closed"] else 1.0
                s["vehicle_restriction"] = "none"

        vis: dict[str, int] = {}
        for e, s in st.items():
            blocked = state_blocked(s)
            if self.randomize:
                vis[e] = int(blocked and self.rng.random() < self.visible_closure_p)
            else:
                vis[e] = int(blocked and s.get("reported", 1) == 1)
        return st, vis

    def _safe_path_exists(self, a: str, b: str) -> bool:
        H = nx.Graph()
        H.add_nodes_from(self.ix.nodes)
        for u, v in self.ix.G.edges:
            if not state_blocked(self.state[edge_id(u, v)]):
                H.add_edge(u, v)
        return nx.has_path(H, a, b)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        options = options or {}
        for _ in range(50):
            if "state" in options:
                self.state = options["state"]
                self.visible = options["visible"]
            else:
                self.state, self.visible = self._sample_state()
            self.cur = options.get("start") or self.rng.choice(self.ix.nodes)
            self.tgt = options.get("target") or self.rng.choice(self.ix.nodes)
            if self.cur != self.tgt and (("state" in options) or self._safe_path_exists(self.cur, self.tgt)):
                break
        self.steps, self.time_s = 0, 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        assert self.cur is not None and self.tgt is not None
        return build_obs(
            self.ix,
            self.cur,
            self.tgt,
            self.state,
            self.visible,
            1.0 - self.steps / self.max_steps,
        )

    def step(self, action):
        assert self.cur is not None and self.tgt is not None
        self.steps += 1
        nbrs = self.ix.nbrs[self.cur]
        reward = 0.0
        info: dict[str, Any] = {"invalid": False, "blocked_edge": False}
        act_idx = int(action)

        if act_idx >= len(nbrs):
            reward -= INVALID_PENALTY
            info["invalid"] = True
        else:
            nxt = nbrs[act_idx]
            e = edge_id(self.cur, nxt)
            tt = float(self.ix.G[self.cur][nxt]["travel_time_s"])
            s = self.state[e]
            minutes = tt / 60.0
            reward -= minutes
            reward -= HAZARD_WEIGHT * float(s.get("hazard", 0.0)) * minutes
            if state_blocked(s):
                reward -= CLOSED_PENALTY
                info["blocked_edge"] = True
            self.time_s += tt
            self.cur = nxt

        terminated = self.cur == self.tgt
        if terminated:
            reward += ARRIVE_REWARD
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, info

