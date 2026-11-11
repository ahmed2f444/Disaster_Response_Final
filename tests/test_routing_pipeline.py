"""End-to-End Integration & TDD Tests for Portion 2 (`routing/`) & Portion 5 WebGIS Backbone."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import db
import main
import seed


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TestPortion2RoutingPipeline(unittest.TestCase):
    """Verify full Portion 2 routing dispatch, dry-run what-if, restrictions, audit chain, and live server."""

    def test_end_to_end_dispatch_whatif_explain_approve_and_geojson_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "phase4_e2e.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            with TestClient(main.app) as client:
                # 1. Create a new OPEN incident near Node 12 (-95.106, 29.798)
                inc_res = client.post(
                    f"/api/incidents?db_path={db_file}",
                    json={
                        "lat": 29.798,
                        "lon": -95.106,
                        "severity": "critical",
                        "required_capability": "rescue",
                        "description": "Phase 5 live dispatch pipeline test incident",
                    },
                )
                self.assertEqual(inc_res.status_code, 200, inc_res.text)
                inc_data = inc_res.json()
                inc_id = inc_data["incident_id"]
                self.assertEqual(inc_data["status"], "open")

                # 2. Compute live Portion 2 MARL dispatch (persist + audit + assign incident)
                disp_res = client.post(
                    f"/api/routes/dispatch?db_path={db_file}",
                    json={"water_level_delta_m": 0.20, "blocked_edges": ["E-01"]},
                )
                self.assertEqual(disp_res.status_code, 200, disp_res.text)
                disp = disp_res.json()
                route_id = disp["route_id"]
                self.assertTrue(route_id.startswith("PLAN-P2-"))
                self.assertIn("E-01", disp["blocked_edges"])
                self.assertLess(disp["compute_ms"], 500.0)
                self.assertGreater(disp["eta_seconds"], 0.0)

                assigned_ids = {str(sub.get("incident_id")) for sub in disp["routes"]}
                self.assertIn(inc_id, assigned_ids)
                inc_list = client.get(f"/api/incidents?db_path={db_file}").json()["incidents"]
                inc_row = next(item for item in inc_list if item["incident_id"] == inc_id)
                self.assertEqual(inc_row["status"], "assigned")

                # 3. Dry-run What-If counterfactual with incremental flood surge (+0.15m) + edge block
                before_e03 = next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-03")
                wif_res = client.post(
                    f"/api/routes/what-if?db_path={db_file}",
                    json={"blocked_edges": ["E-03"], "flood_delta": 0.15},
                )
                self.assertEqual(wif_res.status_code, 200, wif_res.text)
                wif = wif_res.json()
                self.assertLess(wif["compute_ms"], 500.0)
                self.assertIn("E-03", wif["newly_blocked_edges"])
                self.assertIn("E-03", wif["dynamic_hazards"])
                self.assertEqual(wif["dynamic_hazards"]["E-03"]["status"], "BLOCKED")
                self.assertTrue(
                    any(v["status"] == "HAZARDOUS" for v in wif["dynamic_hazards"].values()),
                    "Expected at least one HAZARDOUS road in dynamic_hazards after flood surge",
                )
                after_e03 = next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-03")
                self.assertEqual(before_e03["edge_status"], after_e03["edge_status"])

                # 4. Portion 5 Explanation + Validation Gate check on generated plan
                exp_res = client.get(f"/api/explanations/route/{route_id}?db_path={db_file}")
                self.assertEqual(exp_res.status_code, 200, exp_res.text)
                exp = exp_res.json()
                self.assertTrue(exp["validation_passed"])
                self.assertEqual(len(exp["shap_values"]), 7)

                # 5. Approve gate -> 200, duplicate -> 409, verify SHA-256 chain & ROUTE_DISPATCHED
                app_res = client.post(
                    f"/api/routes/{route_id}/approve?db_path={db_file}",
                    json={"commander_id": "CMD_QA_01", "notes": "Pipeline integration QA"},
                )
                self.assertEqual(app_res.status_code, 200, app_res.text)
                self.assertEqual(app_res.json()["action"], "APPROVED")

                dup_res = client.post(
                    f"/api/routes/{route_id}/approve?db_path={db_file}",
                    json={"commander_id": "CMD_QA_01", "notes": "duplicate"},
                )
                self.assertEqual(dup_res.status_code, 409)

                chain_valid, _, chain_msg = db.verify_audit_chain(db_path=db_file)
                self.assertTrue(chain_valid, chain_msg)
                audit_res = client.get(f"/api/audit-log?db_path={db_file}")
                self.assertEqual(audit_res.status_code, 200)
                self.assertTrue(audit_res.json()["valid"])
                event_types = {e["event_type"] for e in audit_res.json()["entries"]}
                self.assertIn("ROUTE_DISPATCHED", event_types)
                self.assertIn("APPROVED", event_types)

                # 6. Section 4.5 GeoJSON payload budget (< 100 KB, < 10,000 features) & OpenAPI contract
                state_res = client.get(f"/api/state?db_path={db_file}")
                self.assertEqual(state_res.status_code, 200)
                self.assertLess(len(state_res.content), 102400)
                state_json = state_res.json()
                total_features = (
                    len(state_json["roads_geojson"]["features"])
                    + len(state_json["buildings_geojson"]["features"])
                )
                self.assertLess(total_features, 10000)

                openapi = client.get("/openapi.json").json()
                ops = sum(
                    1
                    for path_item in openapi["paths"].values()
                    for m in path_item.keys()
                    if m.lower() in {"get", "post", "put", "delete", "patch"}
                )
                self.assertEqual(ops, 18)

    def test_transport_no_heavy_restriction_and_negative_surge_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "restriction_recovery.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            # Restrict primary arterial E-01 (N0-N1) as no_heavy -> blocks heavy TRN-01, remains passable (HAZARDOUS) for MED-01/RSC-01
            db.update_road_edge(
                "E-01",
                flood_depth_m=0.10,
                debris_score=0.0,
                edge_status="HAZARDOUS",
                closure_state=False,
                hazard_level=0.10,
                capacity=0.5,
                vehicle_restriction="no_heavy",
                db_path=db_file,
            )

            with TestClient(main.app) as client:
                res = client.post(
                    f"/api/routes/what-if?db_path={db_file}",
                    json={"blocked_edges": [], "flood_delta": 0.0},
                )
                self.assertEqual(res.status_code, 200, res.text)
                routes_by_unit = {r["unit_id"]: r for r in res.json()["routes"]}
                trn_route = routes_by_unit["TRN-01"]
                self.assertTrue(trn_route["reachable"])
                self.assertNotIn("E-01", trn_route["edge_ids"])

                # Flood-surge +0.50m closes corridors, then -0.45m receding surge reopens flood-closed corridors
                r_flood = client.post(
                    f"/api/routes/dispatch?db_path={db_file}",
                    json={"water_level_delta_m": 0.50, "blocked_edges": []},
                )
                self.assertEqual(r_flood.status_code, 200)
                self.assertGreaterEqual(len(r_flood.json()["blocked_edges"]), 20)

                r_recede = client.post(
                    f"/api/routes/dispatch?db_path={db_file}",
                    json={"water_level_delta_m": -0.45, "blocked_edges": []},
                )
                self.assertEqual(r_recede.status_code, 200)
                self.assertLess(len(r_recede.json()["blocked_edges"]), 5)

    def test_zero_default_blocked_and_bidirectional_open_close_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "bidir_toggle.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            with TestClient(main.app) as client:
                state0 = client.get(f"/api/state?db_path={db_file}").json()
                self.assertEqual(state0["road_counts"]["BLOCKED"], 0)

                # Close E-18 (N10-N2) persistently via MARL dispatch
                disp_close = client.post(
                    f"/api/routes/dispatch?db_path={db_file}",
                    json={"water_level_delta_m": 0.0, "blocked_edges": ["E-18"], "opened_edges": []},
                )
                self.assertEqual(disp_close.status_code, 200, disp_close.text)
                self.assertIn("E-18", disp_close.json()["blocked_edges"])
                self.assertLess(disp_close.json()["compute_ms"], 500.0)
                e18_closed = next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-18")
                self.assertEqual(e18_closed["edge_status"], "BLOCKED")

                # Dry-run What-If reopens E-18 without mutating persisted DB
                wif_open = client.post(
                    f"/api/routes/what-if?db_path={db_file}",
                    json={"blocked_edges": [], "opened_edges": ["E-18"], "flood_delta": 0.0},
                )
                self.assertEqual(wif_open.status_code, 200, wif_open.text)
                self.assertEqual(wif_open.json()["dynamic_hazards"]["E-18"]["status"], "OPEN")
                self.assertLess(wif_open.json()["compute_ms"], 500.0)
                self.assertEqual(
                    next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-18")["edge_status"],
                    "BLOCKED",
                )

                # Persistently reopen E-18 AND verify blocked_edges overrides opened_edges precedence
                disp_open = client.post(
                    f"/api/routes/dispatch?db_path={db_file}",
                    json={"water_level_delta_m": 0.0, "blocked_edges": ["E-03"], "opened_edges": ["E-18", "E-03"]},
                )
                self.assertEqual(disp_open.status_code, 200, disp_open.text)
                self.assertNotIn("E-18", disp_open.json()["blocked_edges"])
                self.assertIn("E-03", disp_open.json()["blocked_edges"])
                e18_open = next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-18")
                self.assertEqual(e18_open["edge_status"], "OPEN")
                self.assertFalse(e18_open["closure_state"])

    def test_arbitrary_satellite_blockage_simulation_triggers_persistent_marl_reroute_and_gtf_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "arb_block.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            with TestClient(main.app) as client:
                disp = client.post(
                    f"/api/routes/dispatch?db_path={db_file}",
                    json={"water_level_delta_m": 0.0, "arbitrary_blockage": True, "arbitrary_seed": 7},
                )
                self.assertEqual(disp.status_code, 200, disp.text)
                body = disp.json()
                self.assertLess(body["compute_ms"], 500.0)
                self.assertEqual(len(body["blocked_edges"]), 1)
                picked_eid = body["blocked_edges"][0]

                exp_res = client.get(f"/api/explanations/route/{body['route_id']}?db_path={db_file}")
                self.assertEqual(exp_res.status_code, 200, exp_res.text)
                self.assertTrue(exp_res.json()["validation_passed"])
                self.assertEqual(len(exp_res.json()["shap_values"]), 7)

                audit = client.get(f"/api/audit-log?db_path={db_file}").json()
                self.assertTrue(audit["valid"])
                self.assertEqual(audit["entries"][0]["payload"].get("arbitrary_blocked_edge"), picked_eid)

    def test_ui_and_docs_reflect_portion2_live_integration(self) -> None:
        html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT_DIR / "static" / "app.js").read_text(encoding="utf-8")
        api_md = (ROOT_DIR / "docs" / "api.md").read_text(encoding="utf-8")
        readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

        self.assertIn('id="btnRecomputeDispatch"', html)
        self.assertIn('id="btnSimulateSatBlockage"', html)
        self.assertIn('id="btnLoadDatasetSample"', html)
        self.assertIn("MAP + MARL ROUTING LIVE", html)
        self.assertNotIn("WHAT-IF 501 STUB", html)
        self.assertIn("async function recomputeDispatch(", js)
        self.assertIn("function simulateLiveSatelliteBlockage()", js)
        self.assertIn("async function loadDatasetSampleTiles()", js)
        self.assertIn("selectedOpenedEdges", js)
        self.assertIn("opened_edges", js)
        self.assertIn('/api/routes/dispatch', js)
        self.assertIn("**Active (Portion 2):**", api_md)
        self.assertIn("opened_edges", api_md)
        self.assertIn("**Integrated**", readme)
        self.assertTrue((ROOT_DIR / "static" / "samples" / "test_fixture_T0_pre_disaster.png").is_file())
        self.assertTrue((ROOT_DIR / "static" / "samples" / "test_fixture_T1_post_disaster.png").is_file())

    def test_live_uvicorn_server_serves_webgis_and_sub_500ms_routes(self) -> None:
        import base64

        port = _free_local_port()
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "live_srv.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            q_db = urllib.request.quote(db_file, safe="")

            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
                cwd=str(ROOT_DIR),
                env={**os.environ, "AEGIS_ALLOW_TEMP_DB": "1"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                base_url = f"http://127.0.0.1:{port}"
                ready = False
                deadline = time.monotonic() + 25.0
                while time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen(f"{base_url}/healthz", timeout=1.5) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                    except Exception:
                        time.sleep(0.25)
                self.assertTrue(ready, "Live Uvicorn server failed to become healthy on /healthz within 25s")

                with urllib.request.urlopen(f"{base_url}/", timeout=5.0) as resp:
                    self.assertEqual(resp.status, 200)
                    index_body = resp.read().decode("utf-8")
                    self.assertIn("btnRecomputeDispatch", index_body)
                    self.assertIn("btnSimulateSatBlockage", index_body)
                    self.assertIn("btnLoadDatasetSample", index_body)

                with urllib.request.urlopen(f"{base_url}/static/samples/test_fixture_T0_pre_disaster.png", timeout=5.0) as r_t0:
                    self.assertEqual(r_t0.status, 200)
                    t0_b64 = "data:image/png;base64," + base64.b64encode(r_t0.read()).decode("ascii")
                with urllib.request.urlopen(f"{base_url}/static/samples/test_fixture_T1_post_disaster.png", timeout=5.0) as r_t1:
                    self.assertEqual(r_t1.status, 200)
                    t1_b64 = "data:image/png;base64," + base64.b64encode(r_t1.read()).decode("ascii")

                with urllib.request.urlopen(f"{base_url}/api/state?db_path={q_db}", timeout=5.0) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(json.loads(resp.read().decode("utf-8"))["road_counts"]["BLOCKED"], 0)

                req_up = urllib.request.Request(
                    f"{base_url}/api/assessment/upload?db_path={q_db}",
                    data=json.dumps({"pre_disaster_image": t0_b64, "post_disaster_image": t1_b64, "sector_id": "houston-east"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_up, timeout=20.0) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(json.loads(resp.read().decode("utf-8"))["status"], "completed")

                for endpoint, payload in (
                    (f"{base_url}/api/routes/what-if?db_path={q_db}", {"blocked_edges": ["E-01"], "flood_delta": 0.25}),
                    (f"{base_url}/api/routes/dispatch?db_path={q_db}", {"water_level_delta_m": 0.0, "blocked_edges": ["E-18"]}),
                    (f"{base_url}/api/routes/dispatch?db_path={q_db}", {"water_level_delta_m": 0.0, "opened_edges": ["E-18"]}),
                    (f"{base_url}/api/routes/dispatch?db_path={q_db}", {"water_level_delta_m": 0.0, "arbitrary_blockage": True, "arbitrary_seed": 3}),
                ):
                    req = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        self.assertEqual(resp.status, 200)
                        body = json.loads(resp.read().decode("utf-8"))
                        self.assertLess(body["compute_ms"], 500.0)

                req_rst = urllib.request.Request(f"{base_url}/api/reset-sector?db_path={q_db}", data=b"", method="POST")
                with urllib.request.urlopen(req_rst, timeout=5.0) as resp:
                    self.assertEqual(resp.status, 200)
                    rst_body = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(rst_body["state"]["road_counts"]["BLOCKED"], 0)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

    def test_zero_sql_and_no_legacy_artifacts_or_refs(self) -> None:
        legacy_folder = "negm " + "part"
        legacy_store = "routing." + "store"
        legacy_api = "routing." + "api"
        self.assertFalse((ROOT_DIR / legacy_folder).exists(), f"Dumped source folder still exists: {legacy_folder}")

        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        code_files = [
            ROOT_DIR / "main.py",
            ROOT_DIR / "models.py",
            ROOT_DIR / "seed.py",
            *(ROOT_DIR / "routing").glob("*.py"),
            *(ROOT_DIR / "damage").glob("*.py"),
        ]
        for py_file in code_files:
            src = py_file.read_text(encoding="utf-8")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Raw SQL '{kw}' found in {py_file.name}")

        audited_files = [
            *code_files,
            ROOT_DIR / "db.py",
            ROOT_DIR / "README.md",
            ROOT_DIR / "docs" / "api.md",
            ROOT_DIR / "static" / "index.html",
            ROOT_DIR / "static" / "app.js",
            *(ROOT_DIR / "tests").glob("*.py"),
        ]
        for path in audited_files:
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn(legacy_folder, text, f"Stale reference '{legacy_folder}' in {path.name}")
            self.assertNotIn(legacy_store, text, f"Stale reference '{legacy_store}' in {path.name}")
            self.assertNotIn(legacy_api, text, f"Stale reference '{legacy_api}' in {path.name}")


if __name__ == "__main__":
    unittest.main()

