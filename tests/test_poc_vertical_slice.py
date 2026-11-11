"""Portion 5 (Ahmed) — System Integration & Contract Verification Suite.

Tests strict compliance with Unified_System_Document.pdf, ahmed.pdf, and disaster_platform:
- 18 System HTTP Contract Endpoints in OpenAPI with Pydantic V2 response schemas
- 4 Shared Objects (Incident, Building, Route, Explanation) + Sec. 4.2 RoadEdgeState attributes
- Zero-SQL helper access layer (only db.py contains SQL)
- Active Portion 1 (damage/), Portion 2 (routing/), Portion 3 (explain/), & Portion 4 (feedback/) integration + 0 B data/script stubs (scripts/download_*)
- Live P2 compute endpoints (/dispatch, /what-if -> 200 OK, <500ms) with HTTP 501 fallback guards when plugins are absent
- Section 4.5 GeoJSON payload budget (< 100 KB, < 10,000 features)
- Section 7 Portion 5 Explanation Validation & ETA Gate (validation_passed=False + template fallback)
- SHA-256 Merkle chain formula: SHA-256(prev_hash + route_id + action + actor + timestamp)
- Split static/ 6-in-1 Web UI serving & live API wiring
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import db
import main
import models
import seed


class TestPortion5Backbone(unittest.TestCase):
    def test_database_helpers_road_attrs_and_hashing(self) -> None:
        """Verify zero-SQL db.py helpers, Section 4.2 road attributes, and SHA-256 formula."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = str(Path(tmp_dir) / "helper_test.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)

            blds = db.get_buildings(db_path=db_file)
            self.assertEqual(len(blds), 25)
            b = db.get_building("BLD-0001", db_path=db_file)
            self.assertIsNotNone(b)
            self.assertEqual(b["building_id"], "BLD-0001")

            updated = db.update_building_label("BLD-0001", 3, "destroyed", confidence=0.99, db_path=db_file)
            self.assertTrue(updated)
            b_after = db.get_building("BLD-0001", db_path=db_file)
            self.assertEqual(b_after["damage_class"], "destroyed")
            self.assertEqual(b_after["damage_tier"], 3)
            self.assertEqual(b_after["class_label"], "destroyed")

            # Verify Section 4.2 road edge attributes stored & retrievable via db.py
            edges = db.get_road_edges(db_path=db_file)
            self.assertEqual(len(edges), 26)
            for edge in edges:
                self.assertIn("closure_state", edge)
                self.assertIn("hazard_level", edge)
                self.assertIn("capacity", edge)
                self.assertIn("vehicle_restriction", edge)

            edge_model = models.RoadEdgeState(
                edge_id=edges[0]["edge_id"],
                state=edges[0]["edge_status"].lower(),
                closure_state=edges[0]["closure_state"],
                hazard_level=edges[0]["hazard_level"],
                capacity=edges[0]["capacity"],
                vehicle_restriction=edges[0]["vehicle_restriction"],
            )
            self.assertEqual(edge_model.edge_id, edges[0]["edge_id"])

            db.insert_incident(
                {
                    "incident_id": "INC-TEST-99",
                    "location": {"lat": 29.81, "lon": -95.15},
                    "severity": "high",
                    "required_capability": "rescue",
                },
                db_path=db_file,
            )
            inc = db.get_incident("INC-TEST-99", db_path=db_file)
            self.assertIsNotNone(inc)
            self.assertEqual(inc["severity"], "high")

            db.insert_route(
                {
                    "route_id": "RT-TEST-01",
                    "unit_id": "MED-01",
                    "incident_id": "INC-TEST-99",
                    "ppo_proposed_path": [{"lat": 29.85, "lon": -95.18}, {"lat": 29.81, "lon": -95.14}],
                    "masked_path": [{"lat": 29.85, "lon": -95.18}, {"lat": 29.85, "lon": -95.14}, {"lat": 29.81, "lon": -95.14}],
                    "mask_altered": True,
                    "mask_reason": "Bridge submerged",
                    "eta_seconds": 180.0,
                    "status": "pending_approval",
                },
                db_path=db_file,
            )
            rt = db.get_route("RT-TEST-01", db_path=db_file)
            self.assertIsNotNone(rt)
            self.assertTrue(rt["mask_altered"])

            # Verify exact SHA-256 formula: SHA-256(prev_hash + route_id + action + actor + timestamp)
            prev = "0" * 64
            r_id = "RT-TEST-01"
            action = "APPROVED"
            actor = "COMMANDER:CMD_01"
            ts = "2026-11-10T12:00:00Z"
            expected_hash = hashlib.sha256(f"{prev}{r_id}{action}{actor}{ts}".encode("utf-8")).hexdigest()
            self.assertEqual(db.compute_audit_hash(prev, r_id, action, actor, ts), expected_hash)

            stats = db.get_feedback_stats(db_path=db_file)
            self.assertIn("total_count", stats)

            # Verify SHA-256 tamper detection catches mutated audit rows
            with db.get_connection(db_file) as conn:
                conn.execute("UPDATE audit_log SET event_type = 'TAMPERED' WHERE seq_id = 1")
                conn.commit()
            tamper_ok, tamper_seq, _ = db.verify_audit_chain(db_path=db_file)
            self.assertFalse(tamper_ok)
            self.assertEqual(tamper_seq, 1)

    def test_p5_endpoints_validation_gate_and_501_shells(self) -> None:
        """Verify all 18 HTTP endpoints, live P1/P2 routing compute (<500ms), <100KB GeoJSON, Validation Gate & SHA-256."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = str(Path(tmp_dir) / "test_poc_mission.db")
            summary = seed.seed_sector(db_path=db_file, num_buildings=220)

            self.assertEqual(summary["buildings"], 220)
            self.assertGreater(summary["low_confidence_buildings"], 0)
            self.assertEqual(summary["roads_blocked"], 0)
            self.assertEqual(summary["roads_hazardous"], 7)
            self.assertEqual(
                summary["roads_open"] + summary["roads_hazardous"] + summary["roads_blocked"],
                26,
            )

            routes_list = db.get_routes(db_path=db_file)
            self.assertGreater(len(routes_list), 0)
            plan_id = str(routes_list[0]["plan_id"])

            approval = main.approve_and_dispatch(plan_id=plan_id, commander_id="CMD_TEST", db_path=db_file)
            self.assertEqual(approval["approval_status"], "APPROVED")

            chain_valid, entry_count, msg = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(chain_valid, msg)
            self.assertGreaterEqual(entry_count, 2)

            client = TestClient(main.app)

            # [1] GET /healthz
            r1 = client.get("/healthz")
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.json()["status"], "healthy")
            self.assertEqual(r1.json()["active_endpoints"], 18)

            # [2] GET /api/state (including Sec. 4.5 GeoJSON < 100 KB and < 10,000 features)
            r2 = client.get(f"/api/state?db_path={db_file}")
            self.assertEqual(r2.status_code, 200)
            st = r2.json()
            self.assertEqual(len(st["buildings_geojson"]["features"]), 220)
            self.assertEqual(len(st["roads_geojson"]["features"]), 26)
            self.assertGreaterEqual(len(st["incidents"]), 1)
            self.assertTrue(st["audit_status"]["valid"])
            valid_dmg = {"no_damage", "minor", "major", "destroyed"}
            self.assertTrue(
                all(f["properties"]["damage_class"] in valid_dmg for f in st["buildings_geojson"]["features"])
            )

            geojson_bytes = len(
                json.dumps(
                    {"buildings": st["buildings_geojson"], "roads": st["roads_geojson"]},
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            total_features = len(st["buildings_geojson"]["features"]) + len(st["roads_geojson"]["features"])
            self.assertLess(geojson_bytes, 100 * 1024, f"GeoJSON exceeded 100 KB: {geojson_bytes}")
            self.assertLess(total_features, 10_000)

            # [3] POST /api/assessment/upload
            r3 = client.post(f"/api/assessment/upload?db_path={db_file}", json={"sector_id": "houston-east"})
            self.assertEqual(r3.status_code, 200)
            self.assertEqual(r3.json()["status"], "pending")
            task_id = r3.json()["task_id"]
            self.assertIn("TASK_ASSESS", task_id)

            # [4] GET /api/assessment/status/{task_id} (pending on metadata-only upload)
            r4 = client.get(f"/api/assessment/status/{task_id}?db_path={db_file}")
            self.assertEqual(r4.status_code, 200)
            self.assertEqual(r4.json()["status"], "pending")
            self.assertEqual(r4.json()["progress_pct"], 0.0)

            # [5] GET /api/buildings
            r5 = client.get(f"/api/buildings?db_path={db_file}")
            self.assertEqual(r5.status_code, 200)
            self.assertEqual(r5.json()["total_count"], 220)
            self.assertTrue(all(b["damage_class"] in valid_dmg for b in r5.json()["buildings"]))

            # [6] GET /api/buildings/{building_id}
            r6 = client.get(f"/api/buildings/BLD-0001?db_path={db_file}")
            self.assertEqual(r6.status_code, 200)
            self.assertEqual(r6.json()["building"]["building_id"], "BLD-0001")
            self.assertIn(r6.json()["building"]["damage_class"], {"no_damage", "minor", "major", "destroyed"})
            self.assertIn("gradcam_overlay_url", r6.json())

            # [7] POST /api/buildings/{building_id}/correct
            r7 = client.post(
                f"/api/buildings/BLD-0001/correct?db_path={db_file}",
                json={"corrected_damage_class": "major", "responder_id": "RESP_01"},
            )
            self.assertEqual(r7.status_code, 200)
            self.assertEqual(r7.json()["updated_class"], "major")
            self.assertTrue(r7.json()["feedback_logged"])

            # [8] POST /api/incidents & [9] GET /api/incidents
            r8 = client.post(
                f"/api/incidents?db_path={db_file}",
                json={"lat": 29.80, "lon": -95.12, "severity": "critical", "required_capability": "medical"},
            )
            self.assertEqual(r8.status_code, 200)
            r9 = client.get(f"/api/incidents?db_path={db_file}")
            self.assertEqual(r9.status_code, 200)
            self.assertGreaterEqual(r9.json()["total_count"], 4)

            # [10] POST /api/routes/dispatch (live Portion 2 MARL dispatch)
            r10 = client.post(f"/api/routes/dispatch?db_path={db_file}", json={"water_level_delta_m": 0.0})
            self.assertEqual(r10.status_code, 200)
            self.assertLess(r10.json()["compute_ms"], 500.0)

            # [11] GET /api/routes & [12] GET /api/routes/{route_id}
            r11 = client.get(f"/api/routes?db_path={db_file}")
            self.assertEqual(r11.status_code, 200)
            self.assertGreaterEqual(r11.json()["total_routes"], 1)

            r12 = client.get(f"/api/routes/{plan_id}?db_path={db_file}")
            self.assertEqual(r12.status_code, 200)
            self.assertIn("ppo_proposed_path", r12.json())
            self.assertIn("masked_path", r12.json())

            # [13] POST /api/routes/what-if (live Portion 2 <500ms counterfactual + P5 narrative fallback)
            r13 = client.post(
                f"/api/routes/what-if?db_path={db_file}",
                json={"blocked_edges": ["E-01"], "flood_delta": 0.25},
            )
            self.assertEqual(r13.status_code, 200)
            self.assertIn("E-01", r13.json()["newly_blocked_edges"])
            self.assertLess(r13.json()["compute_ms"], 500.0)

            # [14] GET /api/explanations/route/{route_id} + Sec. 7 Validation & ETA Gate
            r14_ok = client.get(f"/api/explanations/route/{plan_id}?db_path={db_file}")
            self.assertEqual(r14_ok.status_code, 200)
            self.assertTrue(r14_ok.json()["validation_passed"])
            self.assertIn("shap_values", r14_ok.json())

            db.insert_explanation(
                {
                    "explanation_id": "EXP-RT-INVALID",
                    "target_type": "route",
                    "target_id": plan_id,
                    "natural_language": "Hallucinated ETA 120.0s across E-01.",
                    "attribution_method": "shap",
                    "system_uncertainty": 0.08,
                    "llm_used": True,
                    "validation_passed": True,
                },
                db_path=db_file,
            )
            r14_bad = client.get(f"/api/explanations/route/{plan_id}?db_path={db_file}")
            self.assertEqual(r14_bad.status_code, 200)
            self.assertFalse(r14_bad.json()["validation_passed"])
            self.assertIn("[TEMPLATE FALLBACK]", r14_bad.json()["natural_language"])

            # [15] GET /api/explanations/building/{building_id} + Sec. 7 Validation Gate
            r15_ok = client.get(f"/api/explanations/building/BLD-0001?db_path={db_file}")
            self.assertEqual(r15_ok.status_code, 200)
            self.assertIn("gradcam_regions", r15_ok.json())

            bld_now = db.get_building("BLD-0001", db_path=db_file)
            # Comparative prose mentioning other classes negated must pass gate
            other_cls = [c for c in ("no_damage", "minor", "major", "destroyed") if c != bld_now["class_label"]]
            db.insert_explanation(
                {
                    "explanation_id": "EXP-BLD-COMPARATIVE",
                    "target_type": "building",
                    "target_id": "BLD-0001",
                    "natural_language": (
                        f"Classified as {bld_now['class_label']} (not {other_cls[0]} or {other_cls[1]}). "
                        f"Confidence: {bld_now['confidence'] * 100:.1f}%."
                    ),
                    "attribution_method": "gradcam",
                    "system_uncertainty": round(1.0 - bld_now["confidence"], 4),
                    "llm_used": False,
                    "validation_passed": True,
                },
                db_path=db_file,
            )
            r15_cmp = client.get(f"/api/explanations/building/BLD-0001?db_path={db_file}")
            self.assertEqual(r15_cmp.status_code, 200)
            self.assertTrue(r15_cmp.json()["validation_passed"])

            wrong_lbl = "no_damage" if bld_now["class_label"] != "no_damage" else "destroyed"
            db.insert_explanation(
                {
                    "explanation_id": "EXP-BLD-INVALID",
                    "target_type": "building",
                    "target_id": "BLD-0001",
                    "natural_language": f"Fabricated {wrong_lbl} with 15.0% confidence.",
                    "attribution_method": "gradcam",
                    "system_uncertainty": 0.05,
                    "llm_used": True,
                    "validation_passed": True,
                },
                db_path=db_file,
            )
            r15_bad = client.get(f"/api/explanations/building/BLD-0001?db_path={db_file}")
            self.assertEqual(r15_bad.status_code, 200)
            self.assertFalse(r15_bad.json()["validation_passed"])
            self.assertIn("[TEMPLATE FALLBACK]", r15_bad.json()["natural_language"])

            # [16] & [17] Missing route returns 404; valid route Approve & Reject Gates verify SHA-256
            self.assertEqual(
                client.post(f"/api/routes/NONEXISTENT-99/approve?db_path={db_file}", json={"commander_id": "CMD_X"}).status_code,
                404,
            )
            self.assertEqual(
                client.post(f"/api/routes/NONEXISTENT-99/reject?db_path={db_file}", json={"commander_id": "CMD_X"}).status_code,
                404,
            )
            # plan_id was already approved by approve_and_dispatch() above -> re-approval hits 409 gate lock
            self.assertEqual(
                client.post(
                    f"/api/routes/{plan_id}/approve?db_path={db_file}",
                    json={"commander_id": "CMD_ALPHA", "notes": "Duplicate approval must 409"},
                ).status_code,
                409,
            )

            # Insert fresh pending_approval routes for Approve (200->409) & Reject (200) SHA-256 chain checks
            db.insert_route(
                {
                    "route_id": "RT-APPROVE-TEST",
                    "plan_id": "RT-APPROVE-TEST",
                    "unit_id": "MED-01",
                    "incident_id": "INC-01",
                    "eta_seconds": 720.0,
                    "status": "pending_approval",
                },
                db_path=db_file,
            )
            r16 = client.post(
                f"/api/routes/RT-APPROVE-TEST/approve?db_path={db_file}",
                json={"commander_id": "CMD_ALPHA", "notes": "Approved for tactical deployment"},
            )
            self.assertEqual(r16.status_code, 200)
            app_resp = r16.json()
            self.assertEqual(app_resp["action"], "APPROVED")
            self.assertEqual(
                app_resp["audit_hash"],
                hashlib.sha256(
                    f"{app_resp['prev_hash']}{app_resp['route_id']}{app_resp['action']}{app_resp['actor']}{app_resp['timestamp']}".encode("utf-8")
                ).hexdigest(),
            )

            # Re-deciding an already-approved route must return HTTP 409 Conflict (gate lock)
            r_conflict = client.post(
                f"/api/routes/RT-APPROVE-TEST/reject?db_path={db_file}",
                json={"commander_id": "CMD_BRAVO", "notes": "Should fail on locked plan"},
            )
            self.assertEqual(r_conflict.status_code, 409)

            # Reject a fresh pending_approval route and verify SHA-256 continuity
            db.insert_route(
                {
                    "route_id": "RT-REJECT-TEST",
                    "plan_id": "RT-REJECT-TEST",
                    "unit_id": "MED-01",
                    "incident_id": "INC-01",
                    "eta_seconds": 640.0,
                    "status": "pending_approval",
                },
                db_path=db_file,
            )
            r17 = client.post(
                f"/api/routes/RT-REJECT-TEST/reject?db_path={db_file}",
                json={"commander_id": "CMD_BRAVO", "notes": "Rejected due to rising bayou water levels"},
            )
            self.assertEqual(r17.status_code, 200)
            rej_resp = r17.json()
            self.assertEqual(rej_resp["action"], "REJECTED")
            self.assertEqual(rej_resp["prev_hash"], app_resp["audit_hash"])
            self.assertEqual(
                rej_resp["audit_hash"],
                hashlib.sha256(
                    f"{rej_resp['prev_hash']}{rej_resp['route_id']}{rej_resp['action']}{rej_resp['actor']}{rej_resp['timestamp']}".encode("utf-8")
                ).hexdigest(),
            )

            # [18] POST /api/feedback
            r18 = client.post(
                f"/api/feedback?db_path={db_file}",
                json={
                    "target_id": "BLD-0010",
                    "prediction": "minor",
                    "uncertainty": 0.22,
                    "responder_action": "confirm",
                    "responder_id": "FIELD_UNIT_7",
                },
            )
            self.assertEqual(r18.status_code, 200)
            self.assertEqual(r18.json()["status"], "queued_offline_wal")

            # Diagnostic endpoints: live Portion 4 multi-signal drift (<0.15 on baseline) & mock delegation check
            r_drift = client.get(f"/api/adaptation/drift?db_path={db_file}")
            self.assertEqual(r_drift.status_code, 200)
            self.assertLess(r_drift.json()["drift_distance"], 0.15)
            self.assertFalse(r_drift.json()["drift_detected"])
            self.assertIn("Tier A", r_drift.json()["last_adaptation_tier"])
            with mock.patch.object(
                main,
                "_load_plugin_callable",
                side_effect=lambda mod, fn: (lambda stats: 0.42) if (mod, fn) == ("feedback.drift", "compute_drift_distance") else None,
            ):
                r_drift_plugin = client.get(f"/api/adaptation/drift?db_path={db_file}")
                self.assertEqual(r_drift_plugin.status_code, 200)
                self.assertEqual(r_drift_plugin.json()["drift_distance"], 0.42)

            r_audit = client.get(f"/api/audit-log?db_path={db_file}")
            self.assertEqual(r_audit.status_code, 200)
            self.assertTrue(r_audit.json()["valid"])

    def test_qa_unified_system_and_ahmed_pdf_compliance(self) -> None:
        """QA Audit: Zero raw SQL outside db.py, 0 B teammate stubs, 18 OpenAPI contracts, split static/ UI."""
        # 1. Zero raw SQL in root P5 files, teammate dirs, or scripts (only db.py may write SQL)
        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        checked_files = [ROOT_DIR / "main.py", ROOT_DIR / "models.py", ROOT_DIR / "seed.py"]
        for folder in ("damage", "routing", "explain", "feedback", "scripts"):
            folder_path = ROOT_DIR / folder
            if folder_path.exists():
                checked_files.extend(folder_path.rglob("*.py"))

        for py_file in checked_files:
            if not py_file.exists() or py_file.name == "db.py":
                continue
            source_text = py_file.read_text(encoding="utf-8")
            for kw in sql_keywords:
                self.assertNotIn(
                    kw,
                    source_text,
                    f"QA Violation: Raw SQL '{kw}' found in {py_file.relative_to(ROOT_DIR)}",
                )

        # 2. Verify Portion 1 damage/, Portion 2 routing/, Portion 3 explain/, & Portion 4 feedback/ modules are integrated (>0 B) and remaining 6 stubs stay 0 B
        integrated_modules = [
            "damage/__init__.py", "damage/assess.py", "damage/dataset.py",
            "damage/exports.py", "damage/model.py", "damage/train.py",
            "routing/__init__.py", "routing/env.py", "routing/features.py",
            "routing/generate.py", "routing/graph.py", "routing/mask.py", "routing/train.py",
            "explain/__init__.py", "explain/generate.py", "explain/gradcam.py",
            "explain/llm.py", "explain/shap_explainer.py", "explain/validate.py",
            "feedback/__init__.py", "feedback/adapt.py", "feedback/calibrate.py", "feedback/config.py",
            "feedback/drift.py", "feedback/feedback.py", "feedback/gate.py", "feedback/replay.py",
            "feedback/selection.py", "feedback/status.py",
        ]
        for rel in integrated_modules:
            mod_path = ROOT_DIR / rel
            self.assertTrue(mod_path.exists(), f"Missing integrated module: {rel}")
            self.assertGreater(mod_path.stat().st_size, 0, f"Integrated module {rel} must be populated")

        expected_stubs = [
            "scripts/download_osm.py", "scripts/download_xbd.py",
            "data/exports/.gitkeep", "data/models/.gitkeep", "data/osm/.gitkeep", "data/xbd/.gitkeep",
        ]
        for rel in expected_stubs:
            stub_path = ROOT_DIR / rel
            self.assertTrue(stub_path.exists(), f"Missing disaster_platform stub: {rel}")
            self.assertEqual(stub_path.stat().st_size, 0, f"Unintegrated teammate stub {rel} must stay 0 B")

        # 3. Verify OpenAPI exposes exact 18 HTTP contract endpoints with response schemas
        schema = main.app.openapi()
        paths = schema.get("paths", {})
        operations = [
            (method.upper(), path, spec)
            for path, methods in paths.items()
            for method, spec in methods.items()
            if method.lower() in ("get", "post", "put", "delete", "patch")
        ]
        self.assertEqual(
            len(operations),
            18,
            f"Expected 18 HTTP contract endpoints, got {len(operations)}: {[(m, p) for m, p, _ in operations]}",
        )
        for method, path, spec in operations:
            resp_200 = spec.get("responses", {}).get("200", {})
            schema_ref = resp_200.get("content", {}).get("application/json", {}).get("schema", {})
            self.assertTrue(bool(schema_ref), f"Endpoint {method} {path} missing Pydantic response_model schema")

        # 4. Verify split static/ UI serves all 6 screens + offline assets + live API wiring
        client = TestClient(main.app)
        ui_resp = client.get("/")
        self.assertEqual(ui_resp.status_code, 200)
        html = ui_resp.text
        self.assertIn('<link rel="stylesheet" href="/static/leaflet.css">', html)
        self.assertIn('<link rel="stylesheet" href="/static/style.css">', html)
        self.assertIn('<script src="/static/leaflet.js"></script>', html)
        self.assertIn('<script src="/static/app.js"></script>', html)
        self.assertIn("leafletMap", html)
        self.assertNotIn("three.min.js", html)
        self.assertNotIn("OrbitControls.js", html)
        for screen_id in (
            "screenUploadView",
            "mapShell",
            "screenRouteView",
            "screenBuildingView",
            "screenApprovalModal",
            "statusDriftVal",
            "statusFeedbackCount",
        ):
            self.assertIn(screen_id, html, f"Missing screen element {screen_id}")

        css_resp = client.get("/static/style.css")
        js_resp = client.get("/static/app.js")
        self.assertEqual(css_resp.status_code, 200)
        self.assertEqual(js_resp.status_code, 200)
        js_text = js_resp.text
        for live_url in (
            "/api/assessment/upload",
            "/api/routes/what-if",
            "/api/explanations/route/",
            "/api/buildings/",
            "/api/explanations/building/",
            "/api/feedback",
            "/api/audit-log",
        ):
            self.assertIn(live_url, js_text, f"Missing live endpoint call {live_url} in static/app.js")
        self.assertIn("res.status === 501", js_text)
        self.assertIn("L.canvas(", js_text)
        self.assertIn("L.circleMarker(", js_text)
        self.assertIn("performance.now()", js_text)
        self.assertNotIn("window.location.port !== '8000'", js_text)

    def test_final_qa_findings_1_2_4_5(self) -> None:
        """Verify fixes for Final QA Findings 1, 2, 4, and 5 without mutating preserved model data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = str(Path(tmp_dir) / "qa_findings.db")
            seed.seed_sector(db_path=db_file, num_buildings=30)
            client = TestClient(main.app)

            # Finding 1: POST /api/adaptation/run with train_from_feedback=False and caller-supplied metrics
            # must NOT promote a phantom adapter with empty weights_path to active in model_registry
            spoof_resp = client.post(
                f"/api/adaptation/run?db_path={db_file}",
                json={
                    "disaster_id": "HOUSTON_HARVEY_2017",
                    "tier": "tier_a",
                    "adapter_version": "phantom-unverified-v99",
                    "promote": True,
                    "train_from_feedback": False,
                    "baseline_metrics": {
                        "per_class_f1": {"no_damage": 0.7, "minor": 0.6, "major": 0.6, "destroyed": 0.5},
                        "old_disaster_macro_f1": 0.72,
                        "ece": 0.06,
                    },
                    "candidate_metrics": {
                        "per_class_f1": {"no_damage": 0.8, "minor": 0.7, "major": 0.7, "destroyed": 0.65},
                        "old_disaster_macro_f1": 0.71,
                        "geographic_holdout_f1": 0.74,
                        "geographic_holdout_baseline_f1": 0.70,
                        "train_region_ids": ["r1"],
                        "holdout_region_ids": ["r2"],
                        "ece": 0.05,
                        "tile_latency_ms": 40.0,
                    },
                },
            )
            self.assertEqual(spoof_resp.status_code, 200)
            spoof_body = spoof_resp.json()
            self.assertFalse(spoof_body["promoted"], "Phantom adapter without weights file on disk must not be promoted")
            self.assertIsNone(db.get_active_model_version(db_path=db_file))

            # Finding 2: GET /api/routes/{id} and GET /api/explanations/route/{id} must return identical 7-feature SHAP values
            disp_resp = client.post(f"/api/routes/dispatch?db_path={db_file}", json={"blocked_edges": ["E-03"]})
            self.assertEqual(disp_resp.status_code, 200)
            route_id = disp_resp.json()["route_id"]
            route_shap = {s["feature"]: s["value"] for s in client.get(f"/api/routes/{route_id}?db_path={db_file}").json()["shap_values"]}
            exp_shap = {s["feature"]: s["value"] for s in client.get(f"/api/explanations/route/{route_id}?db_path={db_file}").json()["shap_values"]}
            self.assertEqual(route_shap, exp_shap)

            # Finding 4: Mutating payload_json in audit_log must be detected by db.verify_audit_chain()
            with db.get_connection(db_file) as conn:
                conn.execute("UPDATE audit_log SET payload_json = '{\"tampered\":true}' WHERE seq_id = 1")
                conn.commit()
            chain_ok, failed_seq, chain_msg = db.verify_audit_chain(db_path=db_file)
            self.assertFalse(chain_ok, f"verify_audit_chain failed to detect payload_json tampering: {chain_msg}")
            self.assertEqual(failed_seq, 1)

            # Finding 5: Screen 4 #s4DamageBadge must format no_damage as 'NO DAMAGE' and destroyed as 'DESTROYED' (not 'NO_DAMAGE DAMAGE' / 'DESTROYED DAMAGE')
            js_text = (ROOT_DIR / "static" / "app.js").read_text(encoding="utf-8")
            self.assertIn("DAMAGE_BADGE_TEXT", js_text)
            self.assertNotIn('lbl.toUpperCase() + " DAMAGE"', js_text)


if __name__ == "__main__":
    unittest.main()
