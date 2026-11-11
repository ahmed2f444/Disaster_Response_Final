"""Portion 4 — Dual-Buffer Non-FIFO Replay Memory (Stratified Reservoir + Recent).

Implements Portion 4 Plan Sec. 4:
- Long-term buffer stratified by (disaster_id, damage_class) using Algorithm R
  reservoir sampling with ZERO FIFO eviction across disasters.
- Small recent buffer for immediate incoming field corrections.
- Clipped prioritized sampling with annealed importance weights.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from db import (
    DEFAULT_DB_PATH,
    delete_replay_samples,
    get_replay_samples,
    normalize_damage_class,
    upsert_replay_sample,
)
from feedback import config


def _finite_clamped(raw: Any, default: float, lo: float, hi: float) -> float:
    if raw is None or isinstance(raw, bool):
        val = default
    else:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = default
        if not math.isfinite(val):
            val = default
    return float(max(lo, min(hi, val)))


class DualReplayBuffer:
    """Zero-SQL dual-buffer replay store backed by `db.py` `replay_buffer` table."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, seed: int = 42) -> None:
        self.db_path = db_path
        self.rng = np.random.default_rng(seed)
        self._stream_counts: dict[tuple[str, str], int] = {}

    def add_sample(
        self,
        target_id: str,
        disaster_id: str,
        damage_class: str | int,
        uncertainty: float = 0.5,
        priority: float = 1.0,
        embedding_ref: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert sample into recent buffer and stratified reservoir long-term buffer."""
        _, canonical_cls = normalize_damage_class(damage_class)
        clipped_prio = _finite_clamped(priority, 1.0, 0.01, config.REPLAY_PRIORITY_MAX)
        clean_unc = _finite_clamped(uncertainty, 0.5, 0.0, 1.0)
        now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        meta_payload = dict(metadata or {})

        # 1. Recent buffer update + recent-only cap maintenance
        recent_cap = max(0, int(config.REPLAY_RECENT_CAPACITY))
        if recent_cap > 0:
            recent_id = f"recent::{disaster_id}::{target_id}"
            upsert_replay_sample(
                {
                    "sample_id": recent_id,
                    "target_id": target_id,
                    "disaster_id": disaster_id,
                    "damage_class": canonical_cls,
                    "buffer_tier": "recent",
                    "priority": clipped_prio,
                    "uncertainty": clean_unc,
                    "embedding_ref": embedding_ref,
                    "sample_json": meta_payload,
                    "created_at": now_utc,
                },
                db_path=self.db_path,
            )
            recent_rows = get_replay_samples(
                buffer_tier="recent",
                limit=max(recent_cap + 500, 1000),
                db_path=self.db_path,
            )
            if len(recent_rows) > recent_cap:
                evict_ids = [r["sample_id"] for r in recent_rows[recent_cap:]]
                delete_replay_samples(evict_ids, db_path=self.db_path)

        # 2. Long-term stratified reservoir sampling (ZERO FIFO eviction across disasters)
        stratum = (str(disaster_id), canonical_cls)
        cap = max(1, int(config.REPLAY_LONG_TERM_PER_STRATUM))
        existing_lt = get_replay_samples(
            buffer_tier="long_term",
            disaster_id=disaster_id,
            damage_class=canonical_cls,
            limit=max(cap + 100, 500),
            db_path=self.db_path,
        )
        lt_id = f"lt::{disaster_id}::{canonical_cls}::{target_id}"
        already_in_lt = any(r["sample_id"] == lt_id for r in existing_lt)

        seen = max(self._stream_counts.get(stratum, len(existing_lt)), len(existing_lt))
        if not already_in_lt:
            seen += 1
        self._stream_counts[stratum] = seen

        if already_in_lt or len(existing_lt) < cap:
            upsert_replay_sample(
                {
                    "sample_id": lt_id,
                    "target_id": target_id,
                    "disaster_id": disaster_id,
                    "damage_class": canonical_cls,
                    "buffer_tier": "long_term",
                    "priority": clipped_prio,
                    "uncertainty": clean_unc,
                    "embedding_ref": embedding_ref,
                    "sample_json": meta_payload,
                    "created_at": now_utc,
                },
                db_path=self.db_path,
            )
        else:
            slot = int(self.rng.integers(0, seen))
            if slot < cap:
                victim_id = existing_lt[slot]["sample_id"]
                delete_replay_samples([victim_id], db_path=self.db_path)
                upsert_replay_sample(
                    {
                        "sample_id": lt_id,
                        "target_id": target_id,
                        "disaster_id": disaster_id,
                        "damage_class": canonical_cls,
                        "buffer_tier": "long_term",
                        "priority": clipped_prio,
                        "uncertainty": clean_unc,
                        "embedding_ref": embedding_ref,
                        "sample_json": meta_payload,
                        "created_at": now_utc,
                    },
                    db_path=self.db_path,
                )

        return {
            "target_id": target_id,
            "disaster_id": disaster_id,
            "damage_class": canonical_cls,
            "uncertainty": clean_unc,
            "priority": clipped_prio,
        }

    def get_long_term(
        self,
        disaster_id: str | None = None,
        damage_class: str | None = None,
    ) -> list[dict[str, Any]]:
        return get_replay_samples(
            buffer_tier="long_term",
            disaster_id=disaster_id,
            damage_class=damage_class,
            limit=2000,
            db_path=self.db_path,
        )

    def get_recent(self) -> list[dict[str, Any]]:
        return get_replay_samples(
            buffer_tier="recent",
            limit=max(1, int(config.REPLAY_RECENT_CAPACITY)),
            db_path=self.db_path,
        )

    def sample_batch(
        self,
        batch_size: int = 16,
        beta: float = 0.4,
        alpha: float = 0.6,
        prioritized: bool = True,
    ) -> list[dict[str, Any]]:
        """Sample across long-term and recent buffers with annealed importance weights."""
        if int(batch_size) <= 0:
            return []
        pool = self.get_long_term() + self.get_recent()
        if not pool:
            return []
        # Deduplicate by (disaster_id, target_id) while preserving highest priority
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for item in pool:
            key = (str(item.get("disaster_id", "")), str(item.get("target_id", "")))
            cur_p = _finite_clamped(item.get("priority"), 1.0, 0.01, config.REPLAY_PRIORITY_MAX)
            prev_p = _finite_clamped(dedup[key].get("priority"), 1.0, 0.01, config.REPLAY_PRIORITY_MAX) if key in dedup else -1.0
            if key not in dedup or cur_p > prev_p:
                clean_item = dict(item)
                clean_item["priority"] = cur_p
                clean_item["uncertainty"] = _finite_clamped(item.get("uncertainty"), 0.5, 0.0, 1.0)
                dedup[key] = clean_item
        unique_pool = list(dedup.values())

        n_pool = len(unique_pool)
        k = min(int(batch_size), n_pool)
        if not prioritized:
            chosen_idx = self.rng.choice(n_pool, size=k, replace=False)
            return [{**unique_pool[int(i)], "weight": 1.0} for i in chosen_idx]

        prios = np.array(
            [float(item["priority"]) for item in unique_pool],
            dtype=np.float64,
        )
        probs = np.power(prios, max(0.0, float(alpha)))
        prob_sum = float(probs.sum())
        if not math.isfinite(prob_sum) or prob_sum <= 0.0:
            probs = np.full(n_pool, 1.0 / n_pool, dtype=np.float64)
        else:
            probs /= prob_sum
        chosen_idx = self.rng.choice(n_pool, size=k, replace=False, p=probs)

        raw_weights = np.power(1.0 / np.maximum(n_pool * probs[chosen_idx], 1e-12), max(0.0, float(beta)))
        norm_weights = raw_weights / max(float(raw_weights.max()), 1e-12)

        batch: list[dict[str, Any]] = []
        for idx, w in zip(chosen_idx.tolist(), norm_weights.tolist()):
            item = dict(unique_pool[idx])
            item["weight"] = round(float(w), 4)
            batch.append(item)
        return batch

    def get_summary(self) -> dict[str, Any]:
        lt = self.get_long_term()
        rc = self.get_recent()
        disasters = sorted({r["disaster_id"] for r in lt})
        return {
            "long_term_count": len(lt),
            "recent_count": len(rc),
            "disasters_retained": disasters,
        }

