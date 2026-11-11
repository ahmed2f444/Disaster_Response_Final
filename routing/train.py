"""Train Portion 2 PPO routing policy (`stable-baselines3`) on the road graph.

Example:
    python -m routing.train --timesteps 250000 --minutes 5

Outputs:
    data/models/routing_ppo.zip        trained PPO weights
    routing/logs/training_reward.csv   episode reward log
    routing/logs/training_reward.png   reward curve plot
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .env import RoadGraphIndex, RoutingEnv, build_obs
from .graph import HOUSTON_GRAPH_PATH, PROJECT_ROOT, is_blocked, load_graph

LOG_DIR = Path(__file__).resolve().parent / "logs"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "routing_ppo.zip"


class RewardLogger(BaseCallback):
    """Appends one CSV row per finished episode and enforces optional wall-time budget."""

    def __init__(self, csv_path: Path, minutes: float | None):
        super().__init__()
        self.csv_path = csv_path
        self.minutes = minutes
        self.episode = 0
        self.t0 = time.time()
        self._f = None
        self._w = None

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["episode", "timesteps", "reward", "length", "wall_seconds"])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep and self._w is not None:
                self.episode += 1
                self._w.writerow([
                    self.episode,
                    self.num_timesteps,
                    round(ep["r"], 4),
                    int(ep["l"]),
                    round(time.time() - self.t0, 1),
                ])
        if self._f is not None and self.episode % 50 == 0:
            self._f.flush()
        if self.minutes and (time.time() - self.t0) > self.minutes * 60:
            return False
        return True

    def _on_training_end(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def plot_rewards(csv_path: Path, png_path: Path) -> dict[str, Any]:
    """Generate rolling-mean episode reward chart and return summary stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"episodes": 0, "first_mean": 0.0, "last_mean": 0.0}

    r = np.array([float(x["reward"]) for x in rows])
    w = max(1, min(len(r), max(10, len(r) // 50)))
    roll = np.convolve(r, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(r, alpha=0.25, label="episode reward")
    ax.plot(np.arange(w - 1, len(r)), roll, lw=2, label=f"rolling mean ({w})")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("PPO on the road graph: training reward")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    k = max(1, len(r) // 10)
    return {
        "episodes": len(r),
        "first_mean": round(float(r[:k].mean()), 4),
        "last_mean": round(float(r[-k:].mean()), 4),
    }


def evaluate_reachability(model: PPO, G) -> float:
    """Measure greedy PPO target reachability across all ordered node pairs."""
    ix = RoadGraphIndex(G)
    state = ix.base_state()
    vis = {e: int(is_blocked(s) and s.get("reported", 1) == 1) for e, s in state.items()}
    pairs = list(itertools.permutations(ix.nodes, 2))
    if not pairs:
        return 1.0

    reached = 0
    max_steps = 3 * ix.n
    for start, target in pairs:
        cur = start
        for step in range(max_steps):
            if cur == target:
                break
            obs = build_obs(ix, cur, target, state, vis, 1.0 - step / max_steps)
            action, _ = model.predict(obs, deterministic=True)
            nbrs = ix.nbrs[cur]
            if int(action) >= len(nbrs):
                break
            cur = nbrs[int(action)]
        if cur == target:
            reached += 1
    return reached / len(pairs)


def train(
    graph_path: Path | str = HOUSTON_GRAPH_PATH,
    minutes: float | None = 5.0,
    timesteps: int = 300_000,
    device: str = "cpu",
    seed: int = 0,
    model_path: Path | str = MODEL_PATH,
    log_dir: Path | str = LOG_DIR,
) -> dict[str, Any]:
    """Train PPO on `graph_path`, save checkpoint and reward logs, and return metrics."""
    G = load_graph(graph_path)
    env = Monitor(RoutingEnv(G, seed=seed))
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        seed=seed,
        verbose=0,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
    )
    log_path = Path(log_dir)
    csv_path = log_path / "training_reward.csv"
    model.learn(total_timesteps=timesteps, callback=RewardLogger(csv_path, minutes))

    target_model = Path(model_path)
    target_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(target_model)

    stats = plot_rewards(csv_path, log_path / "training_reward.png")
    stats["device"] = str(model.device)
    stats["reachability"] = round(evaluate_reachability(model, G), 4)
    stats["obs_dim"] = RoadGraphIndex(G).obs_dim
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PPO routing policy")
    ap.add_argument("--graph", default=str(HOUSTON_GRAPH_PATH))
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(train(args.graph, args.minutes, args.timesteps, args.device, args.seed))


if __name__ == "__main__":
    main()

