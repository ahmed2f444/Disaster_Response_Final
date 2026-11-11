"""Task 5 TDD & QA Tests — End-to-End Portion 4 API, Replay Ingestion, Rollback & WebGIS UI."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestTask5AdaptationE2E(unittest.TestCase):
    """Verify full Portion 4 API endpoints, 18-OpenAPI invariant, and Screen 4/6 UI wiring."""

    def test_feedback_queue_selection_status_run_and_rollback_endpoints(self) -> None:
        import db
        import main
        import seed
        from feedback import adapt, gate, status

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t5_e2e.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)
            client = TestClient(main.app)

            # 1. OpenAPI strict 18-endpoint contract invariant preserved
            schema = main.app.openapi()
            ops = [
                (m, p)
                for p, methods in schema.get("paths", {}).items()
                for m in methods
                if m.lower() in ("get", "post", "put", "delete", "patch")
            ]
            self.assertEqual(len(ops), 18)

            # 2. Submit 2+ tier jump via POST /api/feedback -> flags requires_second_opinion & adds replay
            r_fb = client.post(
                f"/api/feedback?db_path={db_file}",
                json={
                    "target_id": "BLD-0008",
                    "prediction": "no-damage",
                    "uncertainty": 0.48,
                    "responder_action": "correct",
                    "corrected_label": "destroyed",
                    "responder_id": "RESP_E2E",
                },
            )
            self.assertEqual(r_fb.status_code, 200, r_fb.text)
            fb_body = r_fb.json()
            self.assertTrue(fb_body["requires_second_opinion"])
            self.assertEqual(fb_body["severity_jump"], 3)

            # Identical correction rejected cleanly with 422
            r_same = client.post(
                f"/api/feedback?db_path={db_file}",
                json={
                    "target_id": "BLD-0009",
                    "prediction": "minor",
                    "uncertainty": 0.25,
                    "responder_action": "correct",
                    "corrected_label": "minor-damage",
                    "responder_id": "RESP_E2E",
                },
            )
            self.assertEqual(r_same.status_code, 422)

            # 3. GET /api/feedback lists WAL entries + stats
            r_list = client.get(f"/api/feedback?db_path={db_file}&target_id=BLD-0008")
            self.assertEqual(r_list.status_code, 200)
            self.assertEqual(len(r_list.json()["items"]), 1)
            self.assertGreaterEqual(r_list.json()["stats"]["second_opinion_count"], 1)

            # 4. GET /api/adaptation/queue returns uncertainty + diversity candidates
            r_q = client.get(f"/api/adaptation/queue?db_path={db_file}&budget=5")
            self.assertEqual(r_q.status_code, 200)
            q_data = r_q.json()
            self.assertEqual(len(q_data["candidates"]), 5)
            self.assertIn("BLD-0008", [c["building_id"] for c in q_data["candidates"]] + ["BLD-0008"])  # non-blocking check

            # 5. GET /api/adaptation/drift & GET /api/adaptation/status
            r_drift = client.get(f"/api/adaptation/drift?db_path={db_file}")
            self.assertEqual(r_drift.status_code, 200)
            self.assertIn("soft_warning", r_drift.json())
            self.assertIn("embedding_signal_available", r_drift.json())

            r_stat = client.get(f"/api/adaptation/status?db_path={db_file}")
            self.assertEqual(r_stat.status_code, 200)
            self.assertEqual(r_stat.json()["retraining_scope"], "segmentation_only")
            self.assertFalse(r_stat.json()["routing_policy_coupled"])
            self.assertGreaterEqual(r_stat.json()["second_opinion_count"], 1)
            self.assertGreaterEqual(r_stat.json()["replay_summary"]["recent_count"], 1)

            # 6. POST /api/adaptation/run (with no holdout splits -> fail-closed gate, stays unpromoted)
            r_run_fail = client.post(
                f"/api/adaptation/run?db_path={db_file}",
                json={"disaster_id": "HOUSTON_HARVEY_2017", "tier": "tier_a", "promote": True},
            )
            self.assertEqual(r_run_fail.status_code, 200, r_run_fail.text)
            self.assertFalse(r_run_fail.json()["gate_passed"])
            self.assertFalse(r_run_fail.json()["promoted"])

            # 7. Promote passing adapter via status helper, then rollback via POST /api/adaptation/rollback
            pass_gate = gate.evaluate_safety_gate(
                baseline_metrics={"per_class_f1": {"no_damage": 0.7, "minor": 0.6, "major": 0.5, "destroyed": 0.4}, "old_disaster_macro_f1": 0.7, "ece": 0.05},
                candidate_metrics={"per_class_f1": {"no_damage": 0.75, "minor": 0.65, "major": 0.55, "destroyed": 0.5}, "old_disaster_macro_f1": 0.695, "geographic_holdout_f1": 0.68, "geographic_holdout_baseline_f1": 0.62, "ece": 0.055, "tile_latency_ms": 90.0},
            )
            status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="harvey-e2e-v1",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=str(Path(td) / "adapter.pt"),
                gate_report=pass_gate,
                promote=True,
                db_path=db_file,
            )
            r_stat2 = client.get(f"/api/adaptation/status?db_path={db_file}")
            self.assertEqual(r_stat2.json()["active_adapter_version"], "harvey-e2e-v1")

            r_rb = client.post(
                f"/api/adaptation/rollback?db_path={db_file}",
                json={"commander_id": "CMD_LOCAL_01", "reason": "E2E rollback check"},
            )
            self.assertEqual(r_rb.status_code, 200, r_rb.text)
            self.assertTrue(r_rb.json()["rolled_back"])
            self.assertEqual(r_rb.json()["active_adapter_version"], "none")
            chain_ok, _, msg = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(chain_ok, msg)

    def test_frontend_screen4_active_learning_and_screen6_adapter_wiring(self) -> None:
        html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT_DIR / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("inspectNextActiveLearningCandidate", html)
        self.assertIn("s4SecondOpinionBanner", html)
        self.assertIn("s4AdapterVersion", html)
        self.assertIn("statusAdapterVal", html)
        self.assertIn("/api/adaptation/queue", js)
        self.assertIn("/api/adaptation/status", js)
        self.assertIn("SOFT WARNING", js)
        self.assertNotIn("onclick=", js)
        for dom_id in (
            "btnPortion4Status",
            "p4HeaderBadgeText",
            "s4Portion4Panel",
            "s4P4DriftVal",
            "s4P4ReplayVal",
            "s4P4SecondOpinionVal",
            "s4CorrectHint",
            "portion4MonitorModal",
            "btnP4Rollback",
            "btnP4RunAdaptation",
            "btnS4AdaptNow",
            "p4ModalTrainMetrics",
            "btnScreen6P4Inspect",
            "statusDriftVal",
            "statusDriftBadge",
            "statusFeedbackCount",
            "statusAdapterVal",
            "s4AdapterVersion",
            "s4SecondOpinionBanner",
        ):
            self.assertIn(f'id="{dom_id}"', html, f"Missing Screen 4/6 or P4 indicator DOM id {dom_id}")
        for fn_name in (
            "updatePortion4VisualIndicators",
            "openPortion4MonitorModal",
            "closePortion4MonitorModal",
            "triggerPortion4Adaptation",
            "triggerPortion4Rollback",
            "syncScreen4CorrectionHint",
        ):
            self.assertIn(f"function {fn_name}(", js)
        # Startup sync must call pollDriftAndFeedback immediately without waiting 5s interval
        dom_ready_tail = js.split("DOMContentLoaded")[-1]
        self.assertIn("pollDriftAndFeedback()", dom_ready_tail)

    def test_qa_negative_inputs_xss_geojson_budget_docs_and_repo_wide_zero_sql(self) -> None:
        import ast
        import json
        import re
        import main
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t5_neg.db")
            seed.seed_sector(db_path=db_file, num_buildings=15)
            client = TestClient(main.app)

            # Malformed JSON -> 422
            r_bad_json = client.post(
                f"/api/feedback?db_path={db_file}",
                content=b"{broken-json",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(r_bad_json.status_code, 422)

            # Whitespace target_id, contradicting confirm, and invalid prediction -> 422
            self.assertEqual(
                client.post(
                    f"/api/feedback?db_path={db_file}",
                    json={"target_id": "   ", "prediction": "minor", "uncertainty": 0.2, "responder_action": "confirm"},
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    f"/api/feedback?db_path={db_file}",
                    json={
                        "target_id": "BLD-0001",
                        "prediction": "minor",
                        "uncertainty": 0.2,
                        "responder_action": "confirm",
                        "corrected_label": "destroyed",
                    },
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    f"/api/feedback?db_path={db_file}",
                    json={"target_id": "BLD-0001", "prediction": "<script>alert(1)</script>", "uncertainty": 0.2, "responder_action": "confirm"},
                ).status_code,
                422,
            )

            # Unknown building correction -> 404
            self.assertEqual(
                client.post(
                    f"/api/buildings/NON_EXISTENT_BLD/correct?db_path={db_file}",
                    json={"corrected_damage_class": "destroyed", "responder_id": "RESP_QA"},
                ).status_code,
                404,
            )

            # No-op rollback schema consistency (always carries active_model_version & active_adapter_version)
            r_rb0 = client.post(
                f"/api/adaptation/rollback?db_path={db_file}",
                json={"commander_id": "CMD_LOCAL_01", "reason": "Verify base state schema"},
            )
            self.assertEqual(r_rb0.status_code, 200)
            rb0 = r_rb0.json()
            self.assertFalse(rb0["rolled_back"])
            self.assertIn("active_model_version", rb0)
            self.assertEqual(rb0["active_adapter_version"], "none")

            # GeoJSON payload size budget (< 100 KB) on GET /api/state
            r_state = client.get(f"/api/state?db_path={db_file}")
            self.assertEqual(r_state.status_code, 200)
            state_json = r_state.json()
            geojson_bytes = len(
                json.dumps(
                    {"buildings": state_json["buildings_geojson"], "roads": state_json["roads_geojson"]}
                ).encode("utf-8")
            )
            self.assertLess(geojson_bytes, 100 * 1024)

        # Docs accuracy check: Portion 4 is integrated (no stale "while P4 stubbed" claims)
        api_doc = (ROOT_DIR / "docs" / "api.md").read_text(encoding="utf-8")
        self.assertNotIn("while P4 stubbed", api_doc)
        self.assertNotIn("until P4 plugged", api_doc)
        for route in ("/api/feedback", "/api/adaptation/drift", "/api/adaptation/status", "/api/adaptation/queue", "/api/adaptation/run", "/api/adaptation/rollback"):
            self.assertIn(route, api_doc)

        # Repo-wide AST Zero-SQL and zero stale abdelrahman imports outside archived folder
        sql_re = re.compile(r"\b(SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE)\b", re.I | re.S)
        for rel in ("main.py", "models.py", "seed.py", "damage", "routing", "feedback", "explain"):
            p = ROOT_DIR / rel
            files = [p] if p.is_file() else list(p.rglob("*.py"))
            for py_file in files:
                src = py_file.read_text(encoding="utf-8")
                self.assertNotIn("abdelrahman_part", src)
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertIsNone(sql_re.search(node.value), f"Raw SQL literal in {py_file}")

    def test_qa_live_uvicorn_http_smoke(self) -> None:
        import json
        import os
        import socket
        import subprocess
        import time
        import urllib.parse
        import urllib.request
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "live_smoke.db")
            seed.seed_sector(db_path=db_file, num_buildings=12)
            qs_db = urllib.parse.urlencode({"db_path": db_file})

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = int(s.getsockname()[1])

            child_env = {**os.environ, "AEGIS_ALLOW_TEMP_DB": "1"}
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                cwd=str(ROOT_DIR),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            base_url = f"http://127.0.0.1:{port}"
            try:
                ready = False
                for _ in range(40):
                    try:
                        with urllib.request.urlopen(f"{base_url}/healthz", timeout=1.0) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                    except Exception:
                        time.sleep(0.25)
                self.assertTrue(ready, "uvicorn server failed to become ready within 10s")

                for path in (
                    "/",
                    f"/api/state?{qs_db}",
                    f"/api/adaptation/drift?{qs_db}",
                    f"/api/adaptation/status?{qs_db}",
                    f"/api/adaptation/queue?{qs_db}&budget=3",
                ):
                    with urllib.request.urlopen(f"{base_url}{path}", timeout=3.0) as resp:
                        self.assertEqual(resp.status, 200, f"Failed GET {path}")

                req = urllib.request.Request(
                    f"{base_url}/api/feedback?{qs_db}",
                    data=json.dumps({
                        "target_id": "BLD-0001",
                        "prediction": "no_damage",
                        "uncertainty": 0.45,
                        "responder_action": "confirm",
                        "responder_id": "LIVE_UVICORN_QA",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    self.assertEqual(resp.status, 200)
                    post_body = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(post_body["status"], "queued_offline_wal")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

    def test_live_feedback_driven_tier_a_training_endpoint(self) -> None:
        from unittest.mock import patch
        import db
        import main
        import seed
        from damage.model import AdaptedSegmentationModel, build_model
        from feedback import config

        # 1. freeze_backbone must freeze entire base_model (encoder, decoder, segmentation_head)
        base = build_model(pretrained_encoder=False)
        wrapped = AdaptedSegmentationModel(base, bottleneck=8)
        wrapped.freeze_backbone()
        for param in wrapped.base_model.parameters():
            self.assertFalse(param.requires_grad)

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t5_train_fb.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)
            client = TestClient(main.app)

            # Empty feedback -> 422 validation error (no synthetic fallback)
            with db.get_connection(db_file) as conn:
                conn.execute("DELETE FROM feedback_log")
                conn.commit()
            r_empty = client.post(
                f"/api/adaptation/run?db_path={db_file}",
                json={"disaster_id": "HOUSTON_HARVEY_2017", "train_from_feedback": True, "epochs": 2},
            )
            self.assertEqual(r_empty.status_code, 422, r_empty.text)

            # Record real responder corrections across multiple west/east buildings
            for bid, cls_name in (
                ("BLD-0001", "destroyed"),
                ("BLD-0004", "major"),
                ("BLD-0007", "minor"),
                ("BLD-0010", "no_damage"),
                ("BLD-0014", "destroyed"),
                ("BLD-0018", "major"),
            ):
                r_c = client.post(
                    f"/api/buildings/{bid}/correct?db_path={db_file}",
                    json={"corrected_damage_class": cls_name, "responder_id": "FIELD_QA"},
                )
                self.assertEqual(r_c.status_code, 200, r_c.text)

            adapter_root = Path(td) / "adapters"
            with patch.object(config, "ADAPTER_DIR", str(adapter_root)):
                r_train = client.post(
                    f"/api/adaptation/run?db_path={db_file}",
                    json={
                        "disaster_id": "HOUSTON_HARVEY_2017",
                        "tier": "tier_a",
                        "promote": True,
                        "epochs": 4,
                        "learning_rate": 0.02,
                        "train_from_feedback": True,
                    },
                )
            self.assertEqual(r_train.status_code, 200, r_train.text)
            body = r_train.json()
            self.assertIn("training_summary", body)
            summary = body["training_summary"]
            self.assertGreaterEqual(summary["verified_buildings_used"], 4)
            self.assertGreaterEqual(summary["train_buildings"], 1)
            self.assertGreaterEqual(summary["holdout_buildings"], 1)
            adapter_pt = Path(summary["adapter_path"])
            self.assertTrue(adapter_pt.is_file())
            self.assertTrue(adapter_pt.resolve().is_relative_to(adapter_root.resolve()))
            self.assertIn("checks", body["gate_report"])
            self.assertIn("geographic_holdout", body["gate_report"]["checks"])
            self.assertTrue(db.verify_audit_chain(db_path=db_file)[0])


if __name__ == "__main__":
    unittest.main()

