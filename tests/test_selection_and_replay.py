"""Task 3 TDD & QA Tests for Portion 4 (`feedback/selection.py` & `feedback/replay.py`)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestTask3SelectionAndReplay(unittest.TestCase):
    """Verify uncertainty+diversity label selection and non-FIFO dual-buffer replay."""

    def test_selection_ranks_uncertainty_with_diversity_and_neighbor_dedup(self) -> None:
        import db
        import seed
        from feedback import feedback as fb_mod, selection

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t3_sel.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)

            # Insert two near-duplicate low-confidence buildings right next to each other,
            # plus a spatially distant low-confidence building
            db.insert_building(
                {
                    "building_id": "BLD-DUP-A",
                    "geometry": {"type": "Polygon", "coordinates": [[[-95.14, 29.81], [-95.139, 29.81], [-95.139, 29.811], [-95.14, 29.81]]]},
                    "centroid_lat": 29.8100,
                    "centroid_lon": -95.1400,
                    "damage_class": "major",
                    "class_label": "major",
                    "confidence": 0.31,
                    "status_flag": "LOW_CONFIDENCE",
                    "last_updated": "2026-11-10T22:00:00Z",
                },
                db_path=db_file,
            )
            db.insert_building(
                {
                    "building_id": "BLD-DUP-B",
                    "geometry": {"type": "Polygon", "coordinates": [[[-95.1401, 29.8101], [-95.1391, 29.8101], [-95.1391, 29.8111], [-95.1401, 29.8101]]]},
                    "centroid_lat": 29.8101,
                    "centroid_lon": -95.1401,
                    "damage_class": "major",
                    "class_label": "major",
                    "confidence": 0.32,
                    "status_flag": "LOW_CONFIDENCE",
                    "last_updated": "2026-11-10T22:00:00Z",
                },
                db_path=db_file,
            )
            db.insert_building(
                {
                    "building_id": "BLD-DIST-C",
                    "geometry": {"type": "Polygon", "coordinates": [[[-95.09, 29.78], [-95.089, 29.78], [-95.089, 29.781], [-95.09, 29.78]]]},
                    "centroid_lat": 29.7800,
                    "centroid_lon": -95.0900,
                    "damage_class": "destroyed",
                    "class_label": "destroyed",
                    "confidence": 0.35,
                    "status_flag": "LOW_CONFIDENCE",
                    "last_updated": "2026-11-10T22:00:00Z",
                },
                db_path=db_file,
            )

            # Confirm BLD-DUP-A? No, select first and check BLD-DUP-B is deduplicated vs BLD-DUP-A
            queue = selection.select_labeling_candidates(budget=5, db_path=db_file)
            selected_ids = [c["building_id"] for c in queue["candidates"]]
            self.assertIn("BLD-DUP-A", selected_ids)
            self.assertIn("BLD-DIST-C", selected_ids)
            self.assertNotIn("BLD-DUP-B", selected_ids, "Adjacent duplicate neighbor must be deduplicated")

            # Confirming BLD-DUP-A removes it from unverified queue
            fb_mod.record_responder_feedback(
                {
                    "target_id": "BLD-DUP-A",
                    "prediction": "major",
                    "uncertainty": 0.69,
                    "responder_action": "confirm",
                    "responder_id": "RESP-01",
                },
                db_path=db_file,
            )
            queue_after = selection.select_labeling_candidates(budget=5, db_path=db_file)
            after_ids = [c["building_id"] for c in queue_after["candidates"]]
            self.assertNotIn("BLD-DUP-A", after_ids)

    def test_dual_buffer_replay_preserves_older_disasters_without_fifo_eviction(self) -> None:
        import seed
        from feedback import config, replay

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t3_replay.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)

            orig_stratum = config.REPLAY_LONG_TERM_PER_STRATUM
            orig_recent = config.REPLAY_RECENT_CAPACITY
            config.REPLAY_LONG_TERM_PER_STRATUM = 4
            config.REPLAY_RECENT_CAPACITY = 6
            try:
                buf = replay.DualReplayBuffer(db_path=db_file, seed=42)
                for i in range(4):
                    buf.add_sample(
                        target_id=f"OLD-BLD-{i}",
                        disaster_id="HOUSTON_HARVEY_2017",
                        damage_class="destroyed",
                        uncertainty=0.45,
                        priority=2.5,
                    )

                # Flood buffer with 25 new disaster samples (exceeding recent capacity=6 and stratum=4)
                for i in range(25):
                    buf.add_sample(
                        target_id=f"NEW-BLD-{i}",
                        disaster_id="GERMANY_FLOOD_2021",
                        damage_class="major",
                        uncertainty=0.55,
                        priority=12.0,  # exceeds REPLAY_PRIORITY_MAX -> must clip
                    )

                old_samples = buf.get_long_term(disaster_id="HOUSTON_HARVEY_2017")
                self.assertEqual(len(old_samples), 4, "Long-term stratified buffer must never FIFO-evict old disasters")

                recent_samples = buf.get_recent()
                self.assertLessEqual(len(recent_samples), 6)

                # Prioritized batch sampling returns clipped priorities and annealed importance weights
                batch = buf.sample_batch(batch_size=8, beta=0.4, prioritized=True)
                self.assertEqual(len(batch), 8)
                for item in batch:
                    self.assertLessEqual(item["priority"], config.REPLAY_PRIORITY_MAX)
                    self.assertTrue(0.0 < item["weight"] <= 1.0)
            finally:
                config.REPLAY_LONG_TERM_PER_STRATUM = orig_stratum
                config.REPLAY_RECENT_CAPACITY = orig_recent

    def test_qa_second_opinion_resolved_chronologically_in_desc_logs(self) -> None:
        import db
        import seed
        from feedback import feedback as fb_mod, selection

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t3_sec.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)
            target = db.get_buildings(db_path=db_file)[0]["building_id"]

            # Responder 1 submits a noisy >=2-tier jump -> flagged requires_second_opinion=1
            res1 = fb_mod.record_responder_feedback(
                {
                    "target_id": target,
                    "prediction": "no_damage",
                    "uncertainty": 0.65,
                    "responder_action": "correct",
                    "corrected_label": "major",
                    "responder_id": "RESP_ONE",
                },
                db_path=db_file,
            )
            self.assertTrue(res1["requires_second_opinion"])

            q1 = selection.select_labeling_candidates(budget=15, min_neighbor_deg=0.0, db_path=db_file)
            flagged_map = {c["building_id"]: c for c in q1["candidates"]}
            self.assertIn(target, flagged_map)
            self.assertTrue(flagged_map[target]["requires_second_opinion"])

            # Responder 2 confirms/resolves the flagged target -> must exit pending selection queue
            fb_mod.record_responder_feedback(
                {
                    "target_id": target,
                    "prediction": "major",
                    "uncertainty": 0.4,
                    "responder_action": "confirm",
                    "responder_id": "RESP_TWO",
                },
                db_path=db_file,
            )
            q2 = selection.select_labeling_candidates(budget=50, min_neighbor_deg=0.0, db_path=db_file)
            self.assertNotIn(target, [c["building_id"] for c in q2["candidates"]])

    def test_qa_selection_class_caps_empty_pool_malformed_rows_and_rate_warning(self) -> None:
        from unittest.mock import patch
        import db
        from feedback import config, feedback as fb_mod, selection

        with tempfile.TemporaryDirectory() as td:
            empty_db = str(Path(td) / "t3_empty.db")
            db.init_db(db_path=empty_db, reset=True)
            res_empty = selection.select_labeling_candidates(budget=0, db_path=empty_db)
            self.assertEqual(res_empty["candidates"], [])
            self.assertEqual(res_empty["pool_size"], 0)
            self.assertEqual(res_empty["budget"], 1)

            cap_db = str(Path(td) / "t3_caps.db")
            db.init_db(db_path=cap_db, reset=True)
            for i in range(5):
                db.insert_building(
                    {
                        "building_id": f"ND-{i}",
                        "geometry": {"type": "Polygon", "coordinates": []},
                        "centroid_lat": 29.7 + i * 0.02,
                        "centroid_lon": -95.2 - i * 0.02,
                        "damage_class": "no_damage",
                        "class_label": "no_damage",
                        "confidence": 0.25,
                    },
                    db_path=cap_db,
                )
            for i in range(2):
                db.insert_building(
                    {
                        "building_id": f"DST-{i}",
                        "geometry": {"type": "Polygon", "coordinates": []},
                        "centroid_lat": 29.85 + i * 0.02,
                        "centroid_lon": -95.1 - i * 0.02,
                        "damage_class": "destroyed",
                        "class_label": "destroyed",
                        "confidence": 0.30,
                    },
                    db_path=cap_db,
                )

            orig_max = config.SELECTION_MAX_PER_CLASS
            orig_hr = config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR
            config.SELECTION_MAX_PER_CLASS = 2
            config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR = 1
            try:
                fb_mod.record_responder_feedback(
                    {
                        "target_id": "ND-4",
                        "prediction": "no_damage",
                        "uncertainty": 0.5,
                        "responder_action": "confirm",
                        "responder_id": "OVERLOADED_RESP",
                    },
                    db_path=cap_db,
                )
                res_capped = selection.select_labeling_candidates(
                    budget=5, responder_id="OVERLOADED_RESP", db_path=cap_db
                )
                class_counts: dict[str, int] = {}
                for c in res_capped["candidates"]:
                    class_counts[c["damage_class"]] = class_counts.get(c["damage_class"], 0) + 1
                self.assertLessEqual(class_counts.get("no_damage", 0), 2)
                self.assertEqual(class_counts.get("destroyed", 0), 2)
                self.assertTrue(res_capped["responder_rate_warning"])
                self.assertGreaterEqual(res_capped["responder_hourly_count"], 1)

                # Malformed confidence/coords must not raise TypeError or NaN scores
                bad_rows = [
                    {
                        "building_id": "BAD-CONF",
                        "damage_class": "minor",
                        "confidence": None,
                        "centroid_lat": None,
                        "centroid_lon": float("nan"),
                    },
                    {
                        "building_id": "OK-CONF",
                        "damage_class": "major",
                        "confidence": 0.35,
                        "centroid_lat": 29.75,
                        "centroid_lon": -95.15,
                    },
                ]
                with patch.object(selection, "get_buildings", return_value=bad_rows):
                    res_mal = selection.select_labeling_candidates(budget=5, db_path=cap_db)
                    self.assertEqual(len(res_mal["candidates"]), 2)
            finally:
                config.SELECTION_MAX_PER_CLASS = orig_max
                config.MAX_QUESTIONS_PER_RESPONDER_PER_HOUR = orig_hr

    def test_qa_replay_edge_cases_multi_disaster_id_nan_safety_and_sql_bounds(self) -> None:
        import db
        from feedback import config, replay

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t3_qa_rep.db")
            db.init_db(db_path=db_file, reset=True)
            buf = replay.DualReplayBuffer(db_path=db_file, seed=7)

            # Empty buffer safely returns empty batch and zero summary counts
            self.assertEqual(buf.sample_batch(batch_size=8), [])
            self.assertEqual(buf.get_summary()["long_term_count"], 0)

            # Shared building ID across two different disasters must not collide in sample_batch
            buf.add_sample(
                target_id="BLD-SHARED-1",
                disaster_id="HOUSTON_HARVEY_2017",
                damage_class="destroyed",
                uncertainty=float("nan"),
                priority=float("nan"),
            )
            buf.add_sample(
                target_id="BLD-SHARED-1",
                disaster_id="TURKEY_QUAKE_2023",
                damage_class="major",
                uncertainty=1.8,
                priority=-3.0,
            )

            self.assertEqual(buf.sample_batch(batch_size=0), [])
            batch_both = buf.sample_batch(batch_size=4, prioritized=True)
            self.assertEqual(len(batch_both), 2)
            self.assertEqual({x["disaster_id"] for x in batch_both}, {"HOUSTON_HARVEY_2017", "TURKEY_QUAKE_2023"})
            for item in batch_both:
                self.assertTrue(0.0 <= item["uncertainty"] <= 1.0)
                self.assertTrue(0.01 <= item["priority"] <= config.REPLAY_PRIORITY_MAX)
                self.assertTrue(0.0 < item["weight"] <= 1.0)

            batch_unif = buf.sample_batch(batch_size=2, prioritized=False)
            self.assertEqual(len(batch_unif), 2)
            self.assertTrue(all(x["weight"] == 1.0 for x in batch_unif))

            # db.get_replay_samples limit<=0 must clamp safely instead of SQLite LIMIT -1
            self.assertEqual(len(db.get_replay_samples(limit=-1, db_path=db_file)), 1)

            # Updating existing long-term sample must update cleanly without exceeding stratum cap
            orig_stratum = config.REPLAY_LONG_TERM_PER_STRATUM
            orig_recent = config.REPLAY_RECENT_CAPACITY
            config.REPLAY_LONG_TERM_PER_STRATUM = 3
            config.REPLAY_RECENT_CAPACITY = 4
            try:
                buf.add_sample("BLD-SHARED-1", "HOUSTON_HARVEY_2017", "destroyed", uncertainty=0.91, priority=4.2)
                self.assertEqual(len(buf.get_long_term(disaster_id="HOUSTON_HARVEY_2017")), 1)
                for i in range(60):
                    buf.add_sample(
                        target_id=f"FLOOD-{i}",
                        disaster_id="GERMANY_FLOOD_2021",
                        damage_class=("no_damage", "minor", "major", "destroyed")[i % 4],
                        uncertainty=0.5,
                        priority=2.0,
                    )
                self.assertEqual(len(buf.get_long_term(disaster_id="HOUSTON_HARVEY_2017")), 1)
                self.assertLessEqual(len(buf.get_recent()), 4)
            finally:
                config.REPLAY_LONG_TERM_PER_STRATUM = orig_stratum
                config.REPLAY_RECENT_CAPACITY = orig_recent

        # Static Zero-SQL (AST string-literal check) and zero stale-reference boundary check on selection.py & replay.py
        import ast
        import re

        sql_re = re.compile(r"\b(SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE)\b", re.I | re.S)
        for fname in ("selection.py", "replay.py"):
            src = (ROOT_DIR / "feedback" / fname).read_text(encoding="utf-8")
            self.assertNotIn("abdelrahman", src.lower())
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertIsNone(sql_re.search(node.value), f"Raw SQL literal in {fname}: {node.value!r}")

    def test_qa_http_queue_endpoint_returns_active_learning_candidates(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t3_http_q.db")
            seed.seed_sector(db_path=db_file, num_buildings=14)
            client = TestClient(main.app)
            resp = client.get("/api/adaptation/queue", params={"budget": 4, "responder_id": "QA_AL_HTTP", "db_path": db_file})
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["budget"], 4)
            self.assertLessEqual(len(body["candidates"]), 4)
            self.assertIn("responder_rate_warning", body)
            self.assertGreater(body["pool_size"], 0)


if __name__ == "__main__":
    unittest.main()

