"""Portion 4 — Null-Distribution Reference Calibration Utility.

Computes empirical 99th-percentile MMD^2 threshold by repeatedly splitting a real
reference embedding batch (`data/exports/reference/reference.npy`) in half.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from feedback import drift


def calibrate_from_array(
    reference: np.ndarray,
    n_splits: int = 50,
    seed: int = 1,
) -> dict[str, float]:
    """Estimate null MMD^2 distribution across random reference half-splits."""
    try:
        ref = np.asarray(reference, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reference batch must be numeric") from exc
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if ref.ndim != 2 or ref.shape[1] == 0:
        raise ValueError(f"Reference batch must be a non-empty 2D embedding matrix (got shape {ref.shape})")
    if not np.all(np.isfinite(ref)):
        raise ValueError("Reference batch contains NaN or Inf values")
    half = len(ref) // 2
    if half < 2:
        raise ValueError("Reference batch must contain at least 4 embedding vectors for split calibration")

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(max(1, n_splits)):
        idx = rng.permutation(len(ref))
        vals.append(drift.mmd2_unbiased(ref[idx[:half]], ref[idx[half : 2 * half]]))

    p99 = float(np.percentile(vals, 99))
    return {
        "splits": float(len(vals)),
        "mean": round(float(np.mean(vals)), 6),
        "p99": round(p99, 6),
        "max": round(float(np.max(vals)), 6),
        "suggested_threshold": round(max(p99, 0.05), 6),
    }


def main() -> None:
    """CLI entrypoint: calibrate threshold from real stored reference embeddings."""
    try:
        ref = drift.load_reference_embeddings()
        summary = calibrate_from_array(ref)
    except (drift.InsufficientData, ValueError) as exc:
        print(f"Calibration skipped: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"null MMD^2 over {int(summary['splits'])} splits: "
        f"mean={summary['mean']:.5f} p99={summary['p99']:.5f} max={summary['max']:.5f}"
    )
    print(f"suggested DRIFT_THRESHOLD ~ {summary['suggested_threshold']:.5f}")


if __name__ == "__main__":
    main()

