"""Portion 4 — Multi-Signal Embedding & Operational Drift Monitor.

Implements:
- Unbiased RBF-kernel MMD^2 (pure vectorized NumPy, zero scikit-learn ABI fragility)
- Sliced Wasserstein Distance (P7 / Portion 4 Plan Sec. 1)
- Truncated SVD/PCA dimensionality reduction for edge speed (P11)
- Strict `*_embedding.npy` filtering so 3D Grad-CAM tensors are never mixed in
- Multi-signal graceful degradation combining embedding drift, label-free
  class/uncertainty shifts, and delayed responder correction rate
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from db import DEFAULT_DB_PATH, get_buildings, get_feedback_stats, normalize_damage_class
from feedback import config

_CANONICAL_CLASSES = ("no_damage", "minor", "major", "destroyed")
# Nominal Houston-East prior distribution (aligned with seed.py baseline corridor)
_NOMINAL_CLASS_PRIOR = np.array([0.47, 0.24, 0.17, 0.12], dtype=np.float64)
_NOMINAL_UNCERTAINTY_CEILING = 0.28


class InsufficientData(Exception):
    """Raised when embedding vectors are missing or below minimum sample thresholds."""


def _coerce_2d_finite(arr: Any, name: str) -> np.ndarray:
    """Validate and convert an embedding batch into a finite 2D float64 array."""
    try:
        out = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InsufficientData(f"{name} must be a numeric matrix") from exc
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise InsufficientData(f"{name} must be a non-empty 2D matrix (got shape {out.shape})")
    if not np.all(np.isfinite(out)):
        raise InsufficientData(f"{name} contains non-finite values (NaN or Inf)")
    return out


def _validate_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Ensure two embedding matrices are finite 2D float64 arrays with matching feature dimensions."""
    x64 = _coerce_2d_finite(x, "current")
    y64 = _coerce_2d_finite(y, "reference")
    if x64.shape[1] != y64.shape[1]:
        raise InsufficientData(f"Dimension mismatch: {x64.shape[1]} vs {y64.shape[1]}")
    return x64, y64


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix exp(-gamma * ||x_i - y_j||^2) in vectorized NumPy."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    x_sq = np.sum(x64 * x64, axis=1, keepdims=True)
    y_sq = np.sum(y64 * y64, axis=1, keepdims=True).T
    sq_dists = np.maximum(0.0, x_sq + y_sq - 2.0 * (x64 @ y64.T))
    return np.exp(-float(gamma) * sq_dists)


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """RBF gamma = 1 / (2 * median_pairwise_sq_distance) on the pooled sample."""
    x64, y64 = _validate_pair(x, y)
    z = np.vstack([x64, y64])
    if len(z) > 1000:
        idx = np.random.default_rng(0).choice(len(z), 1000, replace=False)
        z = z[idx]
    if len(z) < 2:
        return 1.0
    sq_norm = np.sum(z * z, axis=1, keepdims=True)
    sq = np.maximum(0.0, sq_norm + sq_norm.T - 2.0 * (z @ z.T))
    tri = sq[np.triu_indices_from(sq, k=1)]
    med = float(np.median(tri)) if tri.size > 0 else 0.0
    return (1.0 / (2.0 * med)) if med > 0.0 else 1.0


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD^2 estimate with an RBF kernel, clipped at 0.0 for reporting."""
    x64, y64 = _validate_pair(x, y)
    m, n = len(x64), len(y64)
    if m < 2 or n < 2:
        raise InsufficientData(f"Need >= 2 samples per side for unbiased MMD^2 (got {m}, {n})")
    if gamma is None:
        gamma = median_heuristic_gamma(x64, y64)
    kxx = _rbf_kernel(x64, x64, gamma=gamma)
    kyy = _rbf_kernel(y64, y64, gamma=gamma)
    kxy = _rbf_kernel(x64, y64, gamma=gamma)
    term_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(max(term_x + term_y - 2.0 * kxy.mean(), 0.0))


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 64,
    seed: int = 0,
) -> float:
    """Compute 1D-projected Sliced Wasserstein Distance between two embedding sets."""
    x64, y64 = _validate_pair(x, y)
    dim = x64.shape[1]
    rng = np.random.default_rng(seed)
    dirs = rng.normal(0.0, 1.0, size=(dim, max(1, int(n_projections))))
    norms = np.linalg.norm(dirs, axis=0, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-12)

    proj_x = np.sort(x64 @ dirs, axis=0)
    proj_y = np.sort(y64 @ dirs, axis=0)
    if len(proj_x) != len(proj_y):
        q = np.linspace(0.0, 1.0, max(len(proj_x), len(proj_y)))
        qx = np.linspace(0.0, 1.0, len(proj_x))
        qy = np.linspace(0.0, 1.0, len(proj_y))
        proj_x = np.column_stack([np.interp(q, qx, proj_x[:, j]) for j in range(proj_x.shape[1])])
        proj_y = np.column_stack([np.interp(q, qy, proj_y[:, j]) for j in range(proj_y.shape[1])])
    return float(np.mean(np.abs(proj_x - proj_y)))


def reduce_dimensions(
    current: np.ndarray,
    reference: np.ndarray,
    target_dims: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Center on reference mean and project via SVD/PCA when feature dim exceeds target."""
    k = max(1, int(config.PCA_TARGET_DIMS if target_dims is None else target_dims))
    cur64, ref64 = _validate_pair(current, reference)
    if cur64.shape[1] <= k or ref64.shape[0] <= 2:
        return cur64.astype(np.float32), ref64.astype(np.float32)
    k_eff = min(k, ref64.shape[0] - 1, ref64.shape[1])
    mean = ref64.mean(axis=0, keepdims=True)
    ref_c = ref64 - mean
    cur_c = cur64 - mean
    try:
        _, _, vt = np.linalg.svd(ref_c, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise InsufficientData(f"PCA projection failed: {exc}") from exc
    basis = vt[:k_eff].T
    return (cur_c @ basis).astype(np.float32), (ref_c @ basis).astype(np.float32)


def _load_embedding_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None
    if not isinstance(arr, np.ndarray) or arr.dtype.kind not in {"f", "i", "u"}:
        return None
    if arr.ndim == 1 and arr.size > 0:
        arr2d = arr.reshape(1, -1).astype(np.float32)
    elif arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
        arr2d = arr.astype(np.float32)
    else:
        # Ignore 0D scalars or 3D/4D activation/gradient crops
        return None
    finite_rows = np.all(np.isfinite(arr2d), axis=1)
    cleaned = arr2d[finite_rows]
    return cleaned if len(cleaned) > 0 else None


def load_reference_embeddings(reference_path: str | Path | None = None) -> np.ndarray:
    """Load the validated baseline reference embedding matrix."""
    p = Path(reference_path or config.REFERENCE_PATH)
    if not p.is_file():
        raise InsufficientData(f"Reference embedding batch not found at {p}")
    arr = _load_embedding_array(p)
    if arr is None or arr.size == 0:
        raise InsufficientData(f"Reference file {p} does not contain a valid finite 1D/2D embedding matrix")
    return arr


def load_current_embeddings(
    export_dir: str | Path | None = None,
    window: int | None = None,
) -> np.ndarray:
    """Load newest *_embedding.npy vectors, explicitly excluding Grad-CAM activation/gradient tensors."""
    d = Path(export_dir or config.EXPORT_DIR)
    if not d.is_dir():
        raise InsufficientData(f"Export directory not found: {d}")
    files = sorted((f for f in d.glob("*_embedding.npy") if f.is_file()), key=lambda f: f.stat().st_mtime)
    if not files:
        raise InsufficientData(f"No *_embedding.npy exports found in {d}")

    raw_win = config.CURRENT_WINDOW if window is None else int(window)
    win = max(1, min(raw_win, 50000))
    chunks: list[np.ndarray] = []
    total = 0
    expected_dim: int | None = None
    for f in reversed(files):
        arr = _load_embedding_array(f)
        if arr is None or arr.size == 0:
            continue
        if expected_dim is None:
            expected_dim = int(arr.shape[1])
        elif int(arr.shape[1]) != expected_dim:
            continue
        chunks.append(arr)
        total += len(arr)
        if total >= win:
            break
    if not chunks:
        raise InsufficientData(f"No valid 1D/2D *_embedding.npy vectors found in {d}")
    return np.vstack(chunks[::-1])[-win:]


def _subsample(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(a) <= config.MAX_SAMPLES:
        return a
    return a[rng.choice(len(a), config.MAX_SAMPLES, replace=False)]


def _tile_outlier_distance(current: np.ndarray, reference: np.ndarray) -> float:
    """Compute max nearest-neighbor Euclidean distance from current tiles to reference batch."""
    c64, r64 = _validate_pair(current, reference)
    c_sq = np.sum(c64 * c64, axis=1, keepdims=True)
    r_sq = np.sum(r64 * r64, axis=1, keepdims=True).T
    sq = np.maximum(0.0, c_sq + r_sq - 2.0 * (c64 @ r64.T))
    min_dists = np.sqrt(np.min(sq, axis=1))
    return float(np.max(min_dists)) if min_dists.size else 0.0


def drift_status(
    export_dir: str | Path | None = None,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute embedding-level MMD^2, Sliced Wasserstein, and tile-outlier drift status."""
    cur = load_current_embeddings(export_dir=export_dir)
    ref = load_reference_embeddings(reference_path=reference_path)
    if cur.shape[1] != ref.shape[1]:
        raise InsufficientData(f"Dimension mismatch: current {cur.shape[1]} vs reference {ref.shape[1]}")
    n_cur, n_ref = len(cur), len(ref)
    if min(n_cur, n_ref) < config.MIN_VECTORS:
        raise InsufficientData(
            f"Need at least {config.MIN_VECTORS} embedding vectors per side (current={n_cur}, reference={n_ref})"
        )

    rng = np.random.default_rng(0)
    cur_sub = _subsample(cur, rng)
    ref_sub = _subsample(ref, rng)
    cur_red, ref_red = reduce_dimensions(cur_sub, ref_sub)

    mmd2 = mmd2_unbiased(cur_red, ref_red)
    sw_dist = sliced_wasserstein_distance(cur_red, ref_red)
    outlier_dist = _tile_outlier_distance(cur_red, ref_red)
    return {
        "distance": round(mmd2, 6),
        "mmd2": round(mmd2, 6),
        "sliced_wasserstein": round(sw_dist, 6),
        "tile_outlier_distance": round(outlier_dist, 6),
        "threshold": config.DRIFT_THRESHOLD,
        "soft_threshold": config.SOFT_DRIFT_THRESHOLD,
        "soft_warning": mmd2 >= config.SOFT_DRIFT_THRESHOLD,
        "drift_detected": mmd2 >= config.DRIFT_THRESHOLD,
        "n_current": n_cur,
        "n_reference": n_ref,
        "metric": "mmd2_unbiased_rbf+sliced_wasserstein",
    }


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute base-2 Jensen-Shannon divergence bounded in [0.0, 1.0]."""
    eps = 1e-12
    p_s = np.clip(p, eps, 1.0)
    q_s = np.clip(q, eps, 1.0)
    p_s /= p_s.sum()
    q_s /= q_s.sum()
    m = 0.5 * (p_s + q_s)
    kl_pm = float(np.sum(p_s * np.log2(p_s / m)))
    kl_qm = float(np.sum(q_s * np.log2(q_s / m)))
    return max(0.0, min(1.0, 0.5 * (kl_pm + kl_qm)))


def _compute_operational_signals(
    feedback_stats: dict[str, Any] | None,
    db_path: str | None,
) -> dict[str, Any]:
    """Compute label-free building shifts and shrunk responder correction-rate drift."""
    stats = feedback_stats if feedback_stats is not None else (get_feedback_stats(db_path=db_path) if db_path else {})
    total_fb = max(0, int(stats.get("total_count") or 0))
    raw_rate = stats.get("correction_rate")
    corr_rate_val = float(raw_rate) if (raw_rate is not None and math.isfinite(float(raw_rate))) else 0.0
    raw_corr = stats.get("corrected")
    corrected_fb = max(0, int(raw_corr)) if raw_corr is not None else int(round(corr_rate_val * total_fb))
    # Shrink small sample counts (e.g. 2 seeded feedback rows) so cold-start never false-alarms
    shrunk_corr_rate = corrected_fb / max(total_fb, 10)
    corr_signal = max(0.0, shrunk_corr_rate - 0.05)

    class_jsd = 0.0
    mean_uncertainty = 0.0
    excess_uncertainty = 0.0
    building_count = 0

    if db_path and Path(db_path).is_file():
        buildings = get_buildings(db_path=db_path)
        building_count = len(buildings)
        if building_count > 0:
            counts = {cls: 0 for cls in _CANONICAL_CLASSES}
            uncertainties: list[float] = []
            for b in buildings:
                raw_cls = b.get("damage_class", "no_damage")
                try:
                    _, cls = normalize_damage_class(raw_cls if raw_cls is not None else "no_damage")
                    counts[cls] += 1
                except ValueError:
                    pass
                raw_conf = b.get("confidence", 0.9)
                try:
                    conf = float(0.9 if raw_conf is None else raw_conf)
                    if not math.isfinite(conf):
                        conf = 0.9
                except (TypeError, ValueError):
                    conf = 0.9
                uncertainties.append(max(0.0, min(1.0, 1.0 - conf)))
            counted = sum(counts.values())
            if counted > 0:
                dist = np.array([counts[c] / counted for c in _CANONICAL_CLASSES], dtype=np.float64)
                class_jsd = _jensen_shannon_divergence(dist, _NOMINAL_CLASS_PRIOR)
            if uncertainties:
                mean_uncertainty = float(np.mean(uncertainties))
                excess_uncertainty = max(0.0, mean_uncertainty - _NOMINAL_UNCERTAINTY_CEILING)

    # Secondary score stays ~0.01-0.03 on seeded baseline and >=0.15 on genuine shifts
    secondary_score = min(1.0, 0.35 * corr_signal + 0.35 * class_jsd + 0.30 * excess_uncertainty)
    if total_fb >= 10 and (corrected_fb / max(total_fb, 1)) >= 0.45:
        secondary_score = max(secondary_score, 0.25 * (corrected_fb / total_fb))

    return {
        "building_count": building_count,
        "class_jsd": round(class_jsd, 4),
        "mean_uncertainty": round(mean_uncertainty, 4),
        "excess_uncertainty": round(excess_uncertainty, 4),
        "shrunk_correction_rate": round(shrunk_corr_rate, 4),
        "secondary_score": round(secondary_score, 4),
    }


def compute_drift_report(
    feedback_stats: dict[str, Any] | None = None,
    db_path: str | None = None,
    export_dir: str | Path | None = None,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return full multi-signal drift report, degrading gracefully if embedding batch is missing."""
    op = _compute_operational_signals(feedback_stats, db_path)
    emb_available = False
    emb_reason = ""
    emb_payload: dict[str, Any] = {}

    try:
        emb_payload = drift_status(export_dir=export_dir, reference_path=reference_path)
        emb_available = True
    except InsufficientData as exc:
        emb_reason = str(exc)

    emb_distance = float(emb_payload.get("mmd2", 0.0)) if emb_available else 0.0
    combined_distance = round(max(emb_distance, float(op["secondary_score"])), 4)

    return {
        "drift_distance": combined_distance,
        "soft_threshold": config.SOFT_DRIFT_THRESHOLD,
        "threshold": config.DRIFT_THRESHOLD,
        "soft_warning": combined_distance >= config.SOFT_DRIFT_THRESHOLD,
        "drift_detected": combined_distance >= config.DRIFT_THRESHOLD,
        "embedding_signal_available": emb_available,
        "embedding_reason": emb_reason,
        "embedding_metrics": emb_payload,
        "operational_metrics": op,
    }


def compute_drift_distance(
    feedback_stats: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> float:
    """Portion 5 plug-in entrypoint invoked by `main.py` without raising 502 on missing embeddings."""
    report = compute_drift_report(feedback_stats=feedback_stats, db_path=db_path)
    return float(report["drift_distance"])
