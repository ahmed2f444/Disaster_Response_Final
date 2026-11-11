"""Task 2 TDD & QA Tests for Portion 4 (`feedback/drift.py` & `feedback/calibrate.py`)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestTask2DriftMonitor(unittest.TestCase):
    """Verify MMD^2, Sliced Wasserstein, PCA reduction, activation exclusion, and multi-signal drift."""

    def test_mmd2_and_sliced_wasserstein_detect_distribution_shift(self) -> None:
        from feedback import drift

        rng = np.random.default_rng(7)
        ref = rng.normal(0.0, 1.0, (120, 64)).astype(np.float32)
        same = rng.normal(0.0, 1.0, (120, 64)).astype(np.float32)
        shifted = rng.normal(0.9, 1.4, (120, 64)).astype(np.float32)

        mmd_same = drift.mmd2_unbiased(same, ref)
        mmd_shift = drift.mmd2_unbiased(shifted, ref)
        self.assertLess(mmd_same, 0.03)
        self.assertGreater(mmd_shift, 0.15)

        sw_same = drift.sliced_wasserstein_distance(same, ref, n_projections=64, seed=0)
        sw_shift = drift.sliced_wasserstein_distance(shifted, ref, n_projections=64, seed=0)
        self.assertLess(sw_same, 0.25)
        self.assertGreater(sw_shift, sw_same * 2.5)

    def test_pca_dimension_reduction_preserves_sample_count(self) -> None:
        from feedback import drift

        rng = np.random.default_rng(11)
        ref = rng.normal(0.0, 1.0, (80, 512)).astype(np.float32)
        cur = rng.normal(0.5, 1.0, (40, 512)).astype(np.float32)

        cur_red, ref_red = drift.reduce_dimensions(cur, ref, target_dims=32)
        self.assertEqual(cur_red.shape, (40, 32))
        self.assertEqual(ref_red.shape, (80, 32))

    def test_load_current_embeddings_ignores_3d_gradcam_activations_and_gradients(self) -> None:
        from feedback import config, drift

        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td) / "exports"
            ref_dir = exp_dir / "reference"
            ref_dir.mkdir(parents=True)

            # Real Portion 1 pattern: 3D Grad-CAM tensors alongside 1D (512,) tile embeddings
            np.save(exp_dir / "houston-east_T1_BLD-001_activations.npy", rng.normal(size=(64, 10, 12)).astype(np.float32))
            np.save(exp_dir / "houston-east_T1_BLD-001_gradients.npy", rng.normal(size=(64, 10, 12)).astype(np.float32))
            for idx in range(35):
                vec = rng.normal(0.0, 1.0, (512,)).astype(np.float32)
                vec /= np.linalg.norm(vec)
                np.save(exp_dir / f"tile_{idx:03d}_embedding.npy", vec)

            ref = rng.normal(0.0, 1.0, (40, 512)).astype(np.float32)
            ref /= np.linalg.norm(ref, axis=1, keepdims=True)
            ref_path = ref_dir / "reference.npy"
            np.save(ref_path, ref)

            orig_exp, orig_ref = config.EXPORT_DIR, config.REFERENCE_PATH
            config.EXPORT_DIR, config.REFERENCE_PATH = str(exp_dir), str(ref_path)
            try:
                loaded = drift.load_current_embeddings()
                self.assertEqual(loaded.shape, (35, 512))
                status = drift.drift_status()
                self.assertEqual(status["n_current"], 35)
                self.assertEqual(status["n_reference"], 40)
                self.assertIn("tile_outlier_distance", status)
            finally:
                config.EXPORT_DIR, config.REFERENCE_PATH = orig_exp, orig_ref

    def test_compute_drift_distance_graceful_when_reference_missing_and_detects_feedback_or_dist_shift(self) -> None:
        import db
        import seed
        from feedback import config, drift

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t2_drift.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            missing_ref = str(Path(td) / "no_ref" / "reference.npy")
            empty_exp = str(Path(td) / "no_exp")

            orig_exp, orig_ref = config.EXPORT_DIR, config.REFERENCE_PATH
            config.EXPORT_DIR, config.REFERENCE_PATH = empty_exp, missing_ref
            try:
                # Baseline seeded state should remain low / stable without raising InsufficientData
                report_base = drift.compute_drift_report(
                    feedback_stats={"total_count": 2, "correction_rate": 0.0},
                    db_path=db_file,
                )
                self.assertFalse(report_base["embedding_signal_available"])
                self.assertFalse(report_base["drift_detected"])
                self.assertLess(report_base["drift_distance"], config.DRIFT_THRESHOLD)

                # Elevated correction rate + destroyed spike + high uncertainty triggers drift
                for b in db.get_buildings(db_path=db_file):
                    db.update_building_label(b["building_id"], "destroyed", confidence=0.35, db_path=db_file)

                report_high = drift.compute_drift_report(
                    feedback_stats={"total_count": 20, "corrected": 12, "correction_rate": 0.60},
                    db_path=db_file,
                )
                self.assertTrue(report_high["soft_warning"])
                self.assertTrue(report_high["drift_detected"])
                self.assertGreaterEqual(report_high["drift_distance"], config.DRIFT_THRESHOLD)
                self.assertGreaterEqual(drift.compute_drift_distance({"total_count": 20, "correction_rate": 0.60}, db_path=db_file), config.DRIFT_THRESHOLD)
            finally:
                config.EXPORT_DIR, config.REFERENCE_PATH = orig_exp, orig_ref

    def test_calibrate_threshold_from_reference_array(self) -> None:
        from feedback import calibrate

        rng = np.random.default_rng(19)
        ref = rng.normal(0.0, 1.0, (80, 32)).astype(np.float32)
        summary = calibrate.calibrate_from_array(ref, n_splits=15, seed=1)
        self.assertGreaterEqual(summary["p99"], summary["mean"])
        self.assertGreaterEqual(summary["suggested_threshold"], 0.0)

    def test_qa_math_edge_cases_zero_variance_nan_and_dim_mismatch(self) -> None:
        from feedback import drift

        const_a = np.ones((20, 16), dtype=np.float32) * 0.5
        const_b = np.ones((20, 16), dtype=np.float32) * 0.5
        self.assertEqual(drift.mmd2_unbiased(const_a, const_b), 0.0)
        self.assertEqual(drift.sliced_wasserstein_distance(const_a, const_b, n_projections=32, seed=4), 0.0)

        cur_r, ref_r = drift.reduce_dimensions(const_a, const_b, target_dims=8)
        self.assertEqual(cur_r.shape, (20, 8))
        self.assertTrue(np.all(np.isfinite(cur_r)))
        self.assertTrue(np.all(np.isfinite(ref_r)))

        # Deterministic seed repeatability
        rng = np.random.default_rng(123)
        x = rng.normal(0.0, 1.0, (25, 32)).astype(np.float32)
        y = rng.normal(0.4, 1.0, (25, 32)).astype(np.float32)
        self.assertEqual(
            drift.sliced_wasserstein_distance(x, y, n_projections=48, seed=7),
            drift.sliced_wasserstein_distance(x, y, n_projections=48, seed=7),
        )

        # NaN / Inf / dim mismatch / N<2 / 3D must raise InsufficientData cleanly
        x_nan = x.copy()
        x_nan[0, 0] = np.nan
        with self.assertRaises(drift.InsufficientData):
            drift.mmd2_unbiased(x_nan, y)
        with self.assertRaises(drift.InsufficientData):
            drift.sliced_wasserstein_distance(x_nan, y)
        with self.assertRaises(drift.InsufficientData):
            drift.mmd2_unbiased(x, np.ones((25, 16), dtype=np.float32))
        with self.assertRaises(drift.InsufficientData):
            drift.mmd2_unbiased(x[:1], y)
        with self.assertRaises(drift.InsufficientData):
            drift.reduce_dimensions(np.ones((5, 4, 4), dtype=np.float32), y)

    def test_qa_corrupt_npy_nan_rows_dir_trap_and_window_clamping(self) -> None:
        from feedback import config, drift

        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td) / "exports"
            exp_dir.mkdir(parents=True)

            # Traps: directory ending with _embedding.npy, corrupt bytes, NaN row, wrong dim, 3D gradcam
            (exp_dir / "trap_dir_embedding.npy").mkdir()
            (exp_dir / "corrupt_embedding.npy").write_bytes(b"NOT_A_NUMPY_FILE")
            bad_nan = np.ones((512,), dtype=np.float32)
            bad_nan[3] = np.nan
            np.save(exp_dir / "nan_vec_embedding.npy", bad_nan)
            np.save(exp_dir / "wrong_dim_embedding.npy", np.ones((128,), dtype=np.float32))
            np.save(exp_dir / "gradcam_activations.npy", rng.normal(size=(32, 8, 8)).astype(np.float32))

            for i in range(12):
                v = rng.normal(0.0, 1.0, (512,)).astype(np.float32)
                np.save(exp_dir / f"valid_{i:02d}_embedding.npy", v)

            corrupt_ref = Path(td) / "corrupt_reference.npy"
            corrupt_ref.write_bytes(b"CORRUPTED_HEADER")

            orig_exp, orig_ref = config.EXPORT_DIR, config.REFERENCE_PATH
            config.EXPORT_DIR, config.REFERENCE_PATH = str(exp_dir), str(corrupt_ref)
            try:
                loaded_all = drift.load_current_embeddings(window=50)
                self.assertEqual(loaded_all.shape, (12, 512))
                self.assertTrue(np.all(np.isfinite(loaded_all)))
                # Window<=0 safely clamps to 1 instead of slicing entire history
                self.assertEqual(drift.load_current_embeddings(window=0).shape, (1, 512))
                self.assertEqual(drift.load_current_embeddings(window=-5).shape, (1, 512))

                # Corrupt reference must degrade gracefully without raising 502-causing exceptions
                rep = drift.compute_drift_report(feedback_stats={"total_count": 0, "correction_rate": 0.0})
                self.assertFalse(rep["embedding_signal_available"])
                self.assertGreater(len(rep["embedding_reason"]), 0)
            finally:
                config.EXPORT_DIR, config.REFERENCE_PATH = orig_exp, orig_ref

    def test_qa_operational_hyphen_labels_none_fields_nan_confidence_and_calibrate_errors(self) -> None:
        import json
        from unittest.mock import patch
        import db
        from feedback import calibrate, config, drift

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "op_qa.db")
            db.init_db(db_path=db_file, reset=True)
            db.insert_building(
                {
                    "building_id": "BLD-HYP-1",
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "centroid_lat": 29.75,
                    "centroid_lon": -95.25,
                    "damage_class": "major-damage",
                    "class_label": "major-damage",
                    "confidence": 0.85,
                },
                db_path=db_file,
            )

            orig_exp, orig_ref = config.EXPORT_DIR, config.REFERENCE_PATH
            config.EXPORT_DIR = str(Path(td) / "missing_exp")
            config.REFERENCE_PATH = str(Path(td) / "missing_ref.npy")
            try:
                rep = drift.compute_drift_report(
                    feedback_stats={"total_count": None, "corrected": None, "correction_rate": None},
                    db_path=db_file,
                )
                json.dumps(rep, allow_nan=False)
                self.assertEqual(rep["operational_metrics"]["building_count"], 1)

                # Defensive guard when upstream row contains hyphenated damage_class or None/NaN confidence
                malformed_rows = [
                    {"damage_class": "no-damage", "confidence": None},
                    {"damage_class": "major-damage", "confidence": float("nan")},
                    {"damage_class": "unknown-tier", "confidence": 0.4},
                ]
                with patch.object(drift, "get_buildings", return_value=malformed_rows):
                    rep_mal = drift.compute_drift_report(
                        feedback_stats={"total_count": None, "corrected": None, "correction_rate": None},
                        db_path=db_file,
                    )
                    json.dumps(rep_mal, allow_nan=False)
                    self.assertEqual(rep_mal["operational_metrics"]["building_count"], 3)

                # Soft vs hard threshold boundary verification on mocked stats
                rep_soft = drift.compute_drift_report(
                    feedback_stats={"total_count": 20, "corrected": 7, "correction_rate": 0.35},
                    db_path=None,
                )
                self.assertTrue(rep_soft["soft_warning"])
                self.assertFalse(rep_soft["drift_detected"])
                self.assertGreaterEqual(rep_soft["drift_distance"], config.SOFT_DRIFT_THRESHOLD)
                self.assertLess(rep_soft["drift_distance"], config.DRIFT_THRESHOLD)

                # Calibration guardrails & clean CLI exit when reference missing
                with self.assertRaises(ValueError):
                    calibrate.calibrate_from_array(np.ones((2, 16), dtype=np.float32))
                with self.assertRaises(ValueError):
                    calibrate.calibrate_from_array(np.ones((10, 4, 4), dtype=np.float32))
                bad_ref = np.ones((12, 8), dtype=np.float32)
                bad_ref[2, 1] = np.nan
                with self.assertRaises(ValueError):
                    calibrate.calibrate_from_array(bad_ref)
                with self.assertRaises(SystemExit) as cm:
                    calibrate.main()
                self.assertNotEqual(cm.exception.code, 0)
            finally:
                config.EXPORT_DIR, config.REFERENCE_PATH = orig_exp, orig_ref

        # Static Zero-SQL and zero stale-reference boundary check on drift.py & calibrate.py
        for fname in ("drift.py", "calibrate.py"):
            src = (ROOT_DIR / "feedback" / fname).read_text(encoding="utf-8")
            self.assertNotIn("abdelrahman", src.lower())
            for kw in ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE"):
                self.assertNotIn(kw, src.upper())

    def test_qa_http_endpoints_never_502_when_reference_missing_or_corrupt(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import seed
        from feedback import config

        rng = np.random.default_rng(77)
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "http_drift.db")
            seed.seed_sector(db_path=db_file, num_buildings=12)
            corrupt_ref = Path(td) / "corrupt_ref.npy"
            corrupt_ref.write_bytes(b"BAD_NPY")
            exp_dir = Path(td) / "exports"
            exp_dir.mkdir(parents=True)
            (exp_dir / "bad_embedding.npy").write_bytes(b"CORRUPT")

            client = TestClient(main.app)
            orig_exp, orig_ref = config.EXPORT_DIR, config.REFERENCE_PATH
            config.EXPORT_DIR, config.REFERENCE_PATH = str(exp_dir), str(corrupt_ref)
            try:
                r_state = client.get("/api/state", params={"db_path": db_file})
                self.assertEqual(r_state.status_code, 200)
                r_drift = client.get("/api/adaptation/drift", params={"db_path": db_file})
                self.assertEqual(r_drift.status_code, 200)
                body = r_drift.json()
                self.assertFalse(body["embedding_signal_available"])

                # Now provide valid >=30 shifted embeddings & valid reference -> embedding signal activates cleanly
                ref_mat = rng.normal(0.0, 1.0, (40, 64)).astype(np.float32)
                np.save(corrupt_ref, ref_mat)
                for idx in range(32):
                    vec = rng.normal(1.1, 1.3, (64,)).astype(np.float32)
                    np.save(exp_dir / f"tile_{idx:03d}_embedding.npy", vec)

                r_drift_ok = client.get("/api/adaptation/drift", params={"db_path": db_file})
                self.assertEqual(r_drift_ok.status_code, 200)
                ok_body = r_drift_ok.json()
                self.assertTrue(ok_body["embedding_signal_available"])
                self.assertTrue(ok_body["drift_detected"])
                self.assertGreaterEqual(ok_body["drift_distance"], config.DRIFT_THRESHOLD)
            finally:
                config.EXPORT_DIR, config.REFERENCE_PATH = orig_exp, orig_ref


if __name__ == "__main__":
    unittest.main()

