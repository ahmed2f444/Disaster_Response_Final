"""Task 1 TDD & QA Tests for Portion 4 (`feedback/` foundation & `db.py` persistence)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestTask1FeedbackFoundation(unittest.TestCase):
    """Verify Zero-SQL persistence, Section 4.4 normalization, and noisy-label guards."""

    def test_feedback_package_exists_with_zero_raw_sql(self) -> None:
        feedback_dir = ROOT_DIR / "feedback"
        self.assertTrue(feedback_dir.is_dir(), "Root feedback/ package must exist")
        for fname in ("__init__.py", "config.py", "feedback.py", "drift.py", "status.py"):
            fpath = feedback_dir / fname
            self.assertTrue(fpath.is_file(), f"Missing feedback module: {fname}")
            self.assertGreater(fpath.stat().st_size, 0, f"Module {fname} must be populated")

        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        for py_file in feedback_dir.rglob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Zero-SQL violation '{kw}' in {py_file.name}")

    def test_db_helpers_for_filtered_feedback_replay_and_model_registry(self) -> None:
        import db
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t1_db.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)

            res = db.log_feedback(
                {
                    "target_id": "BLD-0003",
                    "disaster_id": "HOUSTON_HARVEY_2017",
                    "prediction": "no_damage",
                    "uncertainty": 0.42,
                    "responder_action": "correct",
                    "corrected_label": "destroyed",
                    "responder_id": "RESP-A",
                    "severity_jump": 3,
                    "requires_second_opinion": True,
                    "reliability_weight": 0.85,
                },
                db_path=db_file,
            )
            self.assertIn("feedback_id", res)

            filtered = db.get_feedback_logs(limit=50, target_id="BLD-0003", db_path=db_file)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["corrected_label"], "destroyed")
            self.assertEqual(int(filtered[0]["severity_jump"]), 3)
            self.assertEqual(int(filtered[0]["requires_second_opinion"]), 1)

            # Model registry & active adapter persistence
            db.register_model_version(
                {
                    "model_version": "ResNet18-UNet-xBD-v1.0",
                    "adapter_version": "harvey-tier-a-v1",
                    "disaster_id": "HOUSTON_HARVEY_2017",
                    "weights_path": "data/models/adapters/HOUSTON_HARVEY_2017/harvey-tier-a-v1/adapter.pt",
                    "status": "shadow",
                    "metrics": {"f1_destroyed": 0.72},
                },
                db_path=db_file,
            )
            db.set_active_model_version("ResNet18-UNet-xBD-v1.0", "harvey-tier-a-v1", db_path=db_file)
            active = db.get_active_model_version(db_path=db_file)
            self.assertIsNotNone(active)
            self.assertEqual(active["adapter_version"], "harvey-tier-a-v1")
            self.assertEqual(active["status"], "active")

    def test_feedback_normalization_identical_rejection_and_second_opinion_flag(self) -> None:
        import seed
        from feedback import feedback as fb_mod

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t1_fb.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)

            # Identical predicted & corrected label on 'correct' action must raise ValueError
            with self.assertRaises(ValueError):
                fb_mod.record_responder_feedback(
                    {
                        "target_id": "BLD-0001",
                        "prediction": "minor-damage",
                        "uncertainty": 0.35,
                        "responder_action": "correct",
                        "corrected_label": "minor",
                        "responder_id": "RESP-01",
                    },
                    db_path=db_file,
                )

            # 2+ damage tier flip (no-damage -> major) triggers requires_second_opinion=True
            large_flip = fb_mod.record_responder_feedback(
                {
                    "target_id": "BLD-0002",
                    "prediction": "no-damage",
                    "uncertainty": 0.45,
                    "responder_action": "correct",
                    "corrected_label": "major-damage",
                    "responder_id": "RESP-01",
                },
                db_path=db_file,
            )
            self.assertEqual(large_flip["corrected_label"], "major")
            self.assertEqual(large_flip["severity_jump"], 2)
            self.assertTrue(large_flip["requires_second_opinion"])
            self.assertLess(large_flip["reliability_weight"], 1.0)

            # 1-tier flip (minor -> major) does NOT require second opinion
            small_flip = fb_mod.record_responder_feedback(
                {
                    "target_id": "BLD-0004",
                    "prediction": "minor",
                    "uncertainty": 0.40,
                    "responder_action": "correct",
                    "corrected_label": "major",
                    "responder_id": "RESP-01",
                },
                db_path=db_file,
            )
            self.assertEqual(small_flip["severity_jump"], 1)
            self.assertFalse(small_flip["requires_second_opinion"])

    def test_responder_hourly_workload_cap_flagged_without_blocking(self) -> None:
        import seed
        from feedback import config, feedback as fb_mod

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t1_cap.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)

            orig_cap = config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR
            config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR = 2
            try:
                r1 = fb_mod.record_responder_feedback(
                    {
                        "target_id": "BLD-0001",
                        "prediction": "minor",
                        "uncertainty": 0.3,
                        "responder_action": "confirm",
                        "responder_id": "RESP-BUSY",
                    },
                    db_path=db_file,
                )
                self.assertFalse(r1["rate_limited_warning"])

                r2 = fb_mod.record_responder_feedback(
                    {
                        "target_id": "BLD-0002",
                        "prediction": "minor",
                        "uncertainty": 0.3,
                        "responder_action": "skip",
                        "responder_id": "RESP-BUSY",
                    },
                    db_path=db_file,
                )
                self.assertFalse(r2["rate_limited_warning"])

                # 3rd submission in the hour still logs (never blocks ops) but emits rate_limited_warning
                r3 = fb_mod.record_responder_feedback(
                    {
                        "target_id": "BLD-0003",
                        "prediction": "major",
                        "uncertainty": 0.4,
                        "responder_action": "confirm",
                        "responder_id": "RESP-BUSY",
                    },
                    db_path=db_file,
                )
                self.assertTrue(r3["rate_limited_warning"])
            finally:
                config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR = orig_cap

    def test_qa_adversarial_validation_adapter_override_and_valid_labels(self) -> None:
        import db
        import seed
        from feedback import config, feedback as fb_mod

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t1_adv.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)

            # 1. Whitespace-only, boolean uncertainty, NaN, negative time, and contradicting confirm must raise ValueError
            bad_payloads = [
                {"target_id": "   ", "prediction": "minor", "responder_action": "confirm", "responder_id": "R1"},
                {"target_id": "BLD-1", "prediction": "minor", "responder_action": "confirm", "responder_id": "   "},
                {"target_id": "BLD-1", "prediction": "   ", "responder_action": "confirm", "responder_id": "R1"},
                {"target_id": "BLD-1", "prediction": "minor", "uncertainty": True, "responder_action": "confirm", "responder_id": "R1"},
                {"target_id": "BLD-1", "prediction": "minor", "uncertainty": float("nan"), "responder_action": "confirm", "responder_id": "R1"},
                {"target_id": "BLD-1", "prediction": "minor", "uncertainty": 1.2, "responder_action": "confirm", "responder_id": "R1"},
                {"target_id": "BLD-1", "prediction": "minor", "uncertainty": 0.2, "time_taken_sec": -0.5, "responder_action": "confirm", "responder_id": "R1"},
                {"target_id": "BLD-1", "prediction": "minor", "corrected_label": "destroyed", "responder_action": "confirm", "responder_id": "R1"},
            ]
            for bad in bad_payloads:
                with self.assertRaises(ValueError, msg=f"Expected ValueError for {bad!r}"):
                    fb_mod.record_responder_feedback(bad, db_path=db_file)

            # 2. Active adapter overrides default adapter_version='none'
            db.register_model_version(
                {"model_version": "ResNet18-UNet-xBD-v1.0", "adapter_version": "qa-active-v1", "status": "shadow"},
                db_path=db_file,
            )
            db.set_active_model_version("ResNet18-UNet-xBD-v1.0", "qa-active-v1", db_path=db_file)
            saved = fb_mod.record_responder_feedback(
                {
                    "target_id": "BLD-0005",
                    "prediction": "minor",
                    "uncertainty": 0.4,
                    "responder_action": "correct",
                    "corrected_label": "major",
                    "adapter_version": "none",
                    "responder_id": "R1",
                },
                db_path=db_file,
            )
            row = db.get_feedback_logs(limit=1, target_id="BLD-0005", db_path=db_file)[0]
            self.assertEqual(row["adapter_version"], "qa-active-v1")
            self.assertEqual(saved["severity_jump"], 1)

            # 3. VALID_LABELS whitelist enforcement
            orig_valid = config.VALID_LABELS
            config.VALID_LABELS = {"no_damage", "destroyed"}
            try:
                with self.assertRaises(ValueError):
                    fb_mod.record_responder_feedback(
                        {"target_id": "BLD-0006", "prediction": "minor", "responder_action": "confirm", "responder_id": "R1"},
                        db_path=db_file,
                    )
            finally:
                config.VALID_LABELS = orig_valid

    def test_qa_legacy_schema_migration_limit_clamping_utc_and_wal_concurrency(self) -> None:
        import concurrent.futures
        import sqlite3
        import db
        from feedback import feedback as fb_mod

        with tempfile.TemporaryDirectory() as td:
            legacy_db = str(Path(td) / "legacy_pre_t1.db")
            # Simulate pre-Task-1 SQLite DB with old 12-column feedback_log and no replay/registry tables
            conn = sqlite3.connect(legacy_db)
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE feedback_log (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id TEXT NOT NULL,
                    disaster_id TEXT NOT NULL DEFAULT 'HOUSTON_HARVEY_2017',
                    model_version TEXT NOT NULL DEFAULT 'ConvNeXt-Tiny-v1.0',
                    adapter_version TEXT NOT NULL DEFAULT 'none',
                    prediction TEXT NOT NULL,
                    uncertainty REAL NOT NULL,
                    responder_action TEXT NOT NULL,
                    corrected_label TEXT NOT NULL DEFAULT '',
                    responder_id TEXT NOT NULL,
                    time_taken_sec REAL NOT NULL DEFAULT 0.0,
                    timestamp TEXT NOT NULL
                );
                INSERT INTO feedback_log (target_id, prediction, uncertainty, responder_action, corrected_label, responder_id, timestamp)
                VALUES ('BLD-LEG-1', 'minor', 0.3, 'correct', 'destroyed', 'OLD_RESP', '2026-11-10T20:00:00Z');
                """
            )
            conn.close()

            # Migration runs lazily on read/write and preserves old rows
            migrated_rows = db.get_feedback_logs(limit=10, db_path=legacy_db)
            self.assertEqual(len(migrated_rows), 1)
            self.assertEqual(migrated_rows[0]["target_id"], "BLD-LEG-1")
            self.assertEqual(int(migrated_rows[0]["severity_jump"]), 0)

            # UTC normalization across +02:00 and Z timestamps
            db.log_feedback(
                {
                    "target_id": "BLD-TZ",
                    "prediction": "minor",
                    "uncertainty": 0.2,
                    "responder_action": "confirm",
                    "responder_id": "RESP_TZ",
                    "timestamp": "2026-11-10T23:30:00+02:00",  # 21:30 UTC
                },
                db_path=legacy_db,
            )
            since_2100 = db.get_feedback_logs(limit=10, since_utc="2026-11-10T21:00:00Z", db_path=legacy_db)
            self.assertEqual([r["target_id"] for r in since_2100], ["BLD-TZ"])

            # limit<=0 must clamp safely to 1 row instead of SQLite LIMIT -1 unbounded dump
            self.assertEqual(len(db.get_feedback_logs(limit=0, db_path=legacy_db)), 1)
            self.assertEqual(len(db.get_feedback_logs(limit=-5, db_path=legacy_db)), 1)

            # Concurrent multi-threaded WAL inserts (8 threads x 5 writes = 40 rows)
            def _worker(tid: int) -> None:
                for k in range(5):
                    fb_mod.record_responder_feedback(
                        {
                            "target_id": f"BLD-CONC-{tid}-{k}",
                            "prediction": "minor",
                            "uncertainty": 0.35,
                            "responder_action": "correct",
                            "corrected_label": "major",
                            "responder_id": f"RESP_T{tid}",
                        },
                        db_path=legacy_db,
                    )

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(_worker, i) for i in range(8)]
                for f in futs:
                    f.result()

            stats = db.get_feedback_stats(db_path=legacy_db)
            self.assertEqual(stats["total_count"], 42)  # 1 legacy + 1 TZ + 40 concurrent

    def test_qa_zero_abdelrahman_part_imports_in_feedback_package(self) -> None:
        for py_file in (ROOT_DIR / "feedback").rglob("*.py"):
            src = py_file.read_text(encoding="utf-8").lower()
            self.assertNotIn("abdelrahman", src, f"Stale reference in {py_file.name}")


if __name__ == "__main__":
    unittest.main()

