"""Portion 4 Configuration — Path, Drift Threshold, Active Learning & Safety Gate Settings.

All paths resolve deterministically from the repository root (`disaster response poc delieverable/`)
and can be overridden via environment variables for field/edge deployments.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(env_key: str, default_rel: str, fallback_env: str | None = None) -> str:
    raw = os.getenv(env_key) or (os.getenv(fallback_env) if fallback_env else None) or default_rel
    p = Path(raw)
    return str(p if p.is_absolute() else (ROOT_DIR / p))


# Shared SQLite WAL database (Unified System Document Section 14.3)
DB_PATH = _resolve_path("AEGIS_DB_PATH", "data/poc_mission.db", fallback_env="FEEDBACK_DB_PATH")

# Optional label whitelist (comma-separated); None accepts all canonical 4 damage tiers
_raw_labels = os.getenv("VALID_LABELS", "")
VALID_LABELS: set[str] | None = {lbl.strip() for lbl in _raw_labels.split(",") if lbl.strip()} or None

# Portion 1 activation/embedding export directory & baseline reference embeddings
EXPORT_DIR = _resolve_path("ACTIVATION_EXPORT_DIR", "data/exports")
REFERENCE_PATH = _resolve_path("REFERENCE_PATH", "data/exports/reference/reference.npy")
ADAPTER_DIR = _resolve_path("ADAPTER_MODEL_DIR", "data/models/adapters")

# Drift monitor parameters (Unified Doc Stage 9 & Portion 4 Plan Sec. 1)
CURRENT_WINDOW = int(os.getenv("CURRENT_WINDOW", "500"))
MAX_SAMPLES = int(os.getenv("MMD_MAX_SAMPLES", "2000"))
MIN_VECTORS = int(os.getenv("MMD_MIN_VECTORS", "30"))
PCA_TARGET_DIMS = int(os.getenv("DRIFT_PCA_DIMS", "64"))
SOFT_DRIFT_THRESHOLD = float(os.getenv("SOFT_DRIFT_THRESHOLD", "0.08"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))

# Active learning & noisy-label guardrails (Portion 4 Plan Sec. 2)
SELECTION_MIN_PER_CLASS = int(os.getenv("SELECTION_MIN_PER_CLASS", "5"))
SELECTION_MAX_PER_CLASS = int(os.getenv("SELECTION_MAX_PER_CLASS", "75"))
MAX_QUESTIONS_PER_RESPONDER_PER_HOUR = int(os.getenv("MAX_QUESTIONS_PER_RESPONDER_PER_HOUR", "30"))
MIN_NEIGHBOR_SEPARATION_DEG = float(os.getenv("MIN_NEIGHBOR_SEPARATION_DEG", "0.0012"))
SECOND_OPINION_TIER_JUMP = int(os.getenv("SECOND_OPINION_TIER_JUMP", "2"))

# Dual-buffer replay memory limits (Portion 4 Plan Sec. 4 — zero FIFO eviction on long-term)
REPLAY_LONG_TERM_PER_STRATUM = int(os.getenv("REPLAY_LONG_TERM_PER_STRATUM", "25"))
REPLAY_RECENT_CAPACITY = int(os.getenv("REPLAY_RECENT_CAPACITY", "100"))
REPLAY_PRIORITY_MAX = float(os.getenv("REPLAY_PRIORITY_MAX", "5.0"))

# Safety gate thresholds (Portion 4 Plan Sec. 5 & Stage 10)
MAX_FORGETTING_F1_DROP = float(os.getenv("MAX_FORGETTING_F1_DROP", "0.02"))
MAX_CALIBRATION_ECE_DELTA = float(os.getenv("MAX_CALIBRATION_ECE_DELTA", "0.05"))
MAX_TILE_INFERENCE_MS = float(os.getenv("MAX_TILE_INFERENCE_MS", "1500.0"))

