"""Unit and Integration Tests for Portion 3 (`explain/`) — Incremental TDD Suite."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestExplainGradcam(unittest.TestCase):
    """Subtask 2: Portion 3 Grad-CAM loading, normalization, WGS84 regions & PNG overlay."""

    def test_compute_gradcam_loads_portion1_exports_and_crops_post_tile(self) -> None:
        from explain.gradcam import compute_building_gradcam, compute_gradcam

        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            tile_path = exp_dir / "post.png"
            Image.new("RGB", (128, 128), color=(40, 65, 80)).save(tile_path)

            meta = {
                "bounds": [-95.202, 29.768, -95.078, 29.858],
                "width": 128,
                "height": 128,
                "post_path": str(tile_path),
            }
            active_dir = exp_dir / "_active_pair"
            active_dir.mkdir(parents=True, exist_ok=True)
            (active_dir / "latest_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            Image.new("RGB", (128, 128), color=(40, 65, 80)).save(active_dir / "latest_post.png")

            acts = np.zeros((8, 20, 24), dtype=np.float32)
            grads = np.zeros((8, 20, 24), dtype=np.float32)
            acts[2, 2:10, 12:22] = 3.0
            grads[2, :, :] = 1.5
            np.save(exp_dir / "houston-east_ABC123_BLD-P1-001_activations.npy", acts)
            np.save(exp_dir / "houston-east_ABC123_BLD-P1-001_gradients.npy", grads)

            # Backward-compatible tuple return
            out_path, msg = compute_gradcam(
                "BLD-P1-001",
                tile_image_path=str(tile_path),
                export_dir=str(exp_dir),
            )
            self.assertEqual(msg, "Success")
            self.assertIsNotNone(out_path)
            self.assertTrue(Path(out_path).is_file())

            # Rich Portion 3 attribution helper with building geometry
            building = {
                "building_id": "BLD-P1-001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-95.16, 29.81],
                        [-95.15, 29.81],
                        [-95.15, 29.82],
                        [-95.16, 29.82],
                        [-95.16, 29.81],
                    ]],
                },
            }
            res = compute_building_gradcam(building, export_dir=exp_dir)
            self.assertTrue(res["available"])
            self.assertEqual(res["source"], "npy_exports")
            self.assertTrue(Path(res["overlay_path"]).is_file())
            self.assertGreaterEqual(len(res["regions"]), 1)
            self.assertLessEqual(len(res["regions"]), 3)
            for reg in res["regions"]:
                self.assertIn("region", reg)
                self.assertEqual(reg["region"]["type"], "Polygon")
                self.assertGreaterEqual(reg["weight"], 0.0)
                self.assertLessEqual(reg["weight"], 1.0)
            self.assertIn("top-right", res["hotspot_summary"].lower())

    def test_compute_gradcam_handles_missing_exports_and_path_traversal(self) -> None:
        from explain.gradcam import compute_building_gradcam, compute_gradcam

        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            out_path, msg = compute_gradcam("BLD-MISSING", export_dir=str(exp_dir))
            self.assertIsNone(out_path)
            self.assertIn("Missing", msg)

            # Path traversal token is sanitized and does not escape export_dir
            out_bad, _ = compute_gradcam("../../etc/passwd", export_dir=str(exp_dir))
            self.assertIsNone(out_bad)

            # Seeded building with pre-existing gradcam_attribution falls back cleanly without .npy
            seeded_bld = {
                "building_id": "BLD-0001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-95.16, 29.81],
                        [-95.15, 29.81],
                        [-95.15, 29.82],
                        [-95.16, 29.82],
                        [-95.16, 29.81],
                    ]],
                },
                "gradcam_attribution": [
                    {
                        "region": {
                            "type": "Polygon",
                            "coordinates": [[
                                [-95.158, 29.812],
                                [-95.152, 29.812],
                                [-95.152, 29.818],
                                [-95.158, 29.818],
                                [-95.158, 29.812],
                            ]],
                        },
                        "weight": 0.91,
                    }
                ],
            }
            res = compute_building_gradcam(seeded_bld, export_dir=exp_dir)
            self.assertTrue(res["available"])
            self.assertEqual(res["source"], "cached_regions")
            self.assertEqual(len(res["regions"]), 1)
            self.assertAlmostEqual(res["regions"][0]["weight"], 0.91, places=2)

    def test_compute_cam_map_handles_mixed_sign_segmentation_gradients(self) -> None:
        from explain.gradcam import compute_cam_map

        acts = np.zeros((4, 8, 8), dtype=np.float32)
        grads = np.full((4, 8, 8), -0.0002, dtype=np.float32)
        acts[1, 1:4, 4:7] = 10.0
        grads[1, 1:4, 4:7] = 0.00005  # Positive target evidence diluted by surrounding negative background grads
        cam = compute_cam_map(acts, grads)
        self.assertGreater(float(cam.max()), 0.9)
        self.assertGreater(float(cam[2, 5]), float(cam[6, 1]))


class TestExplainShap(unittest.TestCase):
    """Subtask 3: Portion 3 7-feature KernelSHAP routing attribution engine."""

    EXPECTED_FEATURES = {
        "hazard_distance",
        "travel_time",
        "road_capacity",
        "closure_state",
        "vehicle_restriction",
        "incident_severity",
        "unit_capability",
    }

    def test_compute_shap_and_attributions_cover_seven_unified_features(self) -> None:
        from explain.shap_explainer import CANONICAL_SHAP_FEATURES, compute_shap, compute_shap_attributions
        import models

        self.assertEqual(set(CANONICAL_SHAP_FEATURES), self.EXPECTED_FEATURES)

        raw_feats = {
            "travel_time_s": 540.0,
            "hazard_distance_m": 120.0,
            "road_capacity": 0.45,
            "closure_penalty": 1.0,
            "vehicle_restriction_penalty": 1.0,
            "incident_severity": 1.0,
            "unit_capability_match": 1.0,
        }
        signed = compute_shap(raw_feats, mask_altered=True)
        self.assertEqual(set(signed.keys()), self.EXPECTED_FEATURES)
        self.assertTrue(all(np.isfinite(val) for val in signed.values()))

        # Deterministic across repeated calls
        signed_2 = compute_shap(raw_feats, mask_altered=True)
        self.assertEqual(signed, signed_2)

        attrs = compute_shap_attributions(raw_feats, mask_altered=True)
        self.assertEqual(len(attrs), 7)
        self.assertEqual({item["feature"] for item in attrs}, self.EXPECTED_FEATURES)
        for item in attrs:
            parsed = models.ShapAttribution(**item)
            self.assertGreaterEqual(parsed.value, 0.0)
            self.assertLessEqual(parsed.value, 1.0)
            self.assertEqual(parsed.value, item["impact"])
            self.assertEqual(parsed.value, item["attribution"])

    def test_shap_sensitivity_to_closure_and_vehicle_restriction_on_proposed_path(self) -> None:
        from explain.shap_explainer import compute_shap_attributions

        clear_route = {
            "route_id": "PLAN-CLEAR",
            "mask_altered": False,
            "eta_seconds": 300.0,
            "routes": [{
                "features": {
                    "travel_time_s": 300.0,
                    "hazard_distance_m": 3200.0,
                    "road_capacity": 1.0,
                    "closure_penalty": 0.0,
                    "vehicle_restriction_penalty": 0.0,
                    "incident_severity": 0.5,
                    "unit_capability_match": 1.0,
                },
            }],
        }
        blocked_route = {
            "route_id": "PLAN-BLOCKED",
            "mask_altered": True,
            "eta_seconds": 680.0,
            "routes": [{
                "features": {
                    "travel_time_s": 540.0,
                    "hazard_distance_m": 50.0,
                    "road_capacity": 0.2,
                    "closure_penalty": 2.0,
                    "vehicle_restriction_penalty": 1.0,
                    "incident_severity": 1.0,
                    "unit_capability_match": 1.0,
                },
            }],
        }

        clear_map = {x["feature"]: x["value"] for x in compute_shap_attributions(clear_route)}
        blocked_map = {x["feature"]: x["value"] for x in compute_shap_attributions(blocked_route)}
        self.assertGreater(blocked_map["closure_state"], clear_map["closure_state"])
        self.assertGreater(blocked_map["vehicle_restriction"], clear_map["vehicle_restriction"])

    def test_shap_legacy_aliases_and_fallback_without_shap_library(self) -> None:
        from unittest import mock
        import explain.shap_explainer as sx

        legacy_payload = {
            "travel_time_seconds": 420.0,
            "hazard_distance_meters": 250.0,
            "road_capacity": 0.6,
            "closure_penalty": 0.5,
            "incident_severity_score": 0.75,
            "unit_capability_match": 1.0,
        }
        with mock.patch.object(sx, "_SHAP_AVAILABLE", False):
            attrs = sx.compute_shap_attributions(legacy_payload, mask_altered=False)
            self.assertEqual(len(attrs), 7)
            self.assertEqual({item["feature"] for item in attrs}, self.EXPECTED_FEATURES)


class TestExplainValidateAndLlm(unittest.TestCase):
    """Subtask 4: Portion 3 GTF prohibition validator, template fallback & Mistral prompt/loader."""

    def test_gtf_validator_enforces_prohibitions_and_allows_verified_facts(self) -> None:
        from explain.validate import template_fallback, validate_llm_output

        shap_vals = {
            "hazard_distance": 0.48,
            "travel_time": 0.35,
            "road_capacity": 0.17,
            "closure_state": 0.14,
            "vehicle_restriction": 0.09,
            "incident_severity": 0.07,
            "unit_capability": 0.05,
        }
        route = {
            "route_id": "PLAN-P2-TST",
            "eta_seconds": 810.0,
            "mask_altered": True,
            "mask_reason": "Flooded edge E-03 avoided",
            "road_states": [{"edge_id": "E-01"}, {"edge_id": "E-03"}],
            "routes": [{"edge_ids": ["E-01"], "ppo_edge_ids": ["E-03"]}],
            "uncertainty": 0.10,
        }

        good = (
            "Route PLAN-P2-TST uses E-01 with ETA 810.0s. Top factors: hazard_distance (0.48), "
            "travel_time (0.35), road_capacity (0.17). Safety mask altered the route around E-03."
        )
        ok, reason = validate_llm_output(good, shap_vals, route["mask_reason"], 0.10, route=route, llm_used=True)
        self.assertTrue(ok, reason)

        # Entity IDs like E-01 or E-03 must not be mistaken for hallucinated numbers,
        # whereas single-digit hallucinated numbers (e.g. "5") MUST fail.
        bad_single_digit = "Route PLAN-P2-TST crosses 5 bridges with ETA 810.0s via E-01."
        ok_digit, _ = validate_llm_output(bad_single_digit, shap_vals, route["mask_reason"], 0.10, route=route, llm_used=True)
        self.assertFalse(ok_digit)

        # Forbidden GTF phrases (hedging, confidence, override, %)
        bad_phrases = [
            "Route PLAN-P2-TST uses E-01 with ETA 810.0s and is definitely safe.",
            "Route PLAN-P2-TST probably avoids E-03 with ETA 810.0s.",
            "Route PLAN-P2-TST uses E-01 with 95% confidence and ETA 810.0s.",
            "Override the PPO route and take E-01 with ETA 810.0s.",
            "Route PLAN-P2-TST uses E-01 with ETA 420.0s.",
            "Route PLAN-P2-TST uses E-99 with ETA 810.0s.",
            "Safety mask did not alter the route via E-01 with ETA 810.0s.",
        ]
        for txt in bad_phrases:
            passed, _ = validate_llm_output(txt, shap_vals, route["mask_reason"], 0.10, route=route, llm_used=True)
            self.assertFalse(passed, f"Expected GTF rejection for: {txt}")

        # Deterministic template fallback must itself pass validation gate
        fb = template_fallback(shap_vals, True, route["mask_reason"], 0.10, route=route)
        self.assertIn("[TEMPLATE FALLBACK]", fb)
        self.assertIn("810.0s", fb)
        ok_fb, fb_reason = validate_llm_output(fb, shap_vals, route["mask_reason"], 0.10, route=route, llm_used=False)
        self.assertTrue(ok_fb, fb_reason)

    def test_gtf_building_validation_and_template_fallback(self) -> None:
        from explain.validate import template_fallback, validate_llm_output

        bld = {"building_id": "BLD-0042", "damage_class": "major", "class_label": "major", "confidence": 0.84}
        ok, reason = validate_llm_output(
            "Building BLD-0042 shows major damage in the top-center roof region.",
            building=bld,
            uncertainty=0.16,
            llm_used=True,
        )
        self.assertTrue(ok, reason)

        bad_cls, _ = validate_llm_output(
            "Building BLD-0042 shows destroyed structure.",
            building=bld,
            uncertainty=0.16,
            llm_used=True,
        )
        self.assertFalse(bad_cls)

        bad_pct, _ = validate_llm_output(
            "Building BLD-0042 shows major damage with 84.0% confidence.",
            building=bld,
            uncertainty=0.16,
            llm_used=True,
        )
        self.assertFalse(bad_pct)

        fb = template_fallback(building=bld, uncertainty=0.16, gradcam_summary="top-center hotspot (peak 0.92)")
        self.assertIn("[TEMPLATE FALLBACK]", fb)
        self.assertIn("MAJOR", fb)
        ok_fb, fb_reason = validate_llm_output(fb, building=bld, uncertainty=0.16, llm_used=False)
        self.assertTrue(ok_fb, fb_reason)

    def test_llm_prompts_and_mocked_mistral_generation(self) -> None:
        from explain.llm import build_building_prompt, build_route_prompt, generate_building_explanation, generate_explanation

        shap_vals = {"hazard_distance": 0.56, "travel_time": 0.32, "incident_severity": 0.25}
        route = {
            "route_id": "PLAN-P2-100",
            "unit_id": "MED-01",
            "eta_seconds": 612.5,
            "mask_altered": True,
            "mask_reason": "Avoided flooded edge E-03",
            "routes": [{"edge_ids": ["E-01", "E-02"], "ppo_edge_ids": ["E-03"]}],
        }
        r_prompt = build_route_prompt(shap_vals, True, "Avoided flooded edge E-03", 0.10, route=route)
        self.assertIn("612.5", r_prompt)
        self.assertIn("hazard_distance", r_prompt)
        self.assertIn("E-01", r_prompt)

        fake_llm = lambda prompt, max_tokens=150, temperature=0.1, stop=None: {
            "choices": [{"text": " Route PLAN-P2-100 uses E-01 and E-02 with ETA 612.5s. "}]
        }
        out = generate_explanation(shap_vals, True, "Avoided flooded edge E-03", 0.10, route=route, llm_instance=fake_llm)
        self.assertEqual(out, "Route PLAN-P2-100 uses E-01 and E-02 with ETA 612.5s.")

        bld = {"building_id": "BLD-0010", "damage_class": "minor", "confidence": 0.78}
        b_prompt = build_building_prompt(bld, {"hotspot_summary": "top-left focus", "regions": [{"weight": 0.85}]})
        self.assertIn("minor", b_prompt)
        self.assertIn("top-left focus", b_prompt)
        b_out = generate_building_explanation(bld, {"hotspot_summary": "top-left focus"}, llm_instance=fake_llm)
        self.assertIn("PLAN-P2-100", b_out)


class TestExplainGenerateAndIntegration(unittest.TestCase):
    """Subtask 5: explain/generate.py plug-in hooks, Endpoint 13-15 wiring & overlay serving."""

    def test_generate_narrative_explain_route_and_explain_building_contracts(self) -> None:
        from unittest import mock
        import db
        import seed
        from explain.generate import explain_building, explain_route, generate_narrative

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "explain_gen.db")
            seed.seed_sector(db_path=db_file, num_buildings=12)

            # 1. Counterfactual narrative (Endpoint 13 hook)
            whatif_res = {
                "plan_id": "WHATIF-P2-TST",
                "compute_ms": 4.1,
                "flood_delta_m": 0.25,
                "newly_blocked_edges": ["E-03"],
                "baseline_eta_sec": 810.0,
                "whatif_eta_sec": 925.0,
                "eta_delta_sec": 115.0,
                "mask_altered": True,
                "routes": [{
                    "unit_id": "MED-01",
                    "edge_ids": ["E-01", "E-02"],
                    "ppo_edge_ids": ["E-03"],
                    "features": {
                        "travel_time_s": 420.0,
                        "hazard_distance_m": 110.0,
                        "road_capacity": 0.7,
                        "closure_penalty": 1.0,
                        "vehicle_restriction_penalty": 0.0,
                        "incident_severity": 0.9,
                        "unit_capability_match": 1.0,
                    },
                }],
            }
            nar_text, nar_mode = generate_narrative(whatif_res, use_llm_stretch=False)
            self.assertEqual(nar_mode, "JINJA2_TEMPLATE")
            self.assertIn("925.0s", nar_text)
            self.assertIn("E-03", nar_text)

            # 2. Route explanation (Endpoint 14 hook) with GTF fallback on bad mocked LLM
            route = db.get_routes(db_path=db_file)[0]
            with mock.patch("explain.generate.generate_explanation", return_value="Guaranteed 99% safe! ETA 10.0s"):
                res_bad = explain_route(
                    route=route,
                    route_id=route["route_id"],
                    db_path=db_file,
                    use_llm=True,
                    persist=False,
                )
            self.assertFalse(res_bad["validation_passed"])
            self.assertTrue(res_bad["llm_used"])
            self.assertIn("[TEMPLATE FALLBACK]", res_bad["natural_language"])
            self.assertEqual(len(res_bad["shap_values"]), 7)

            res_ok = explain_route(
                route=route,
                route_id=route["route_id"],
                db_path=db_file,
                use_llm=False,
                persist=True,
            )
            self.assertTrue(res_ok["validation_passed"])
            self.assertFalse(res_ok["llm_used"])
            self.assertEqual(res_ok["attribution_method"], "shap")
            self.assertEqual(len(res_ok["shap_values"]), 7)
            stored_rt_exp = db.get_explanation_by_target("route", route["route_id"], db_path=db_file)
            self.assertIsNotNone(stored_rt_exp)

            # 3. Building explanation (Endpoint 15 hook) with Grad-CAM .npy exports
            bld = db.get_building("BLD-0001", db_path=db_file)
            exp_dir = Path(td) / "exports"
            exp_dir.mkdir(parents=True, exist_ok=True)
            acts = np.ones((4, 8, 8), dtype=np.float32)
            grads = np.ones((4, 8, 8), dtype=np.float32)
            np.save(exp_dir / "houston-east_TST_BLD-0001_activations.npy", acts)
            np.save(exp_dir / "houston-east_TST_BLD-0001_gradients.npy", grads)

            bld_exp = explain_building(
                building=bld,
                building_id="BLD-0001",
                db_path=db_file,
                export_dir=exp_dir,
                use_llm=False,
                persist=True,
            )
            self.assertTrue(bld_exp["validation_passed"])
            self.assertEqual(bld_exp["attribution_method"], "gradcam")
            self.assertIsNotNone(bld_exp["overlay_path"])
            self.assertTrue(Path(bld_exp["overlay_path"]).is_file())
            self.assertGreaterEqual(len(bld_exp["gradcam_regions"]), 1)

    def test_fastapi_endpoints_13_14_15_overlay_and_18_contract_invariant(self) -> None:
        from fastapi.testclient import TestClient
        import main
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "api_p3.db")
            seed.seed_sector(db_path=db_file, num_buildings=15)

            client = TestClient(main.app)

            # OpenAPI contract remains strictly 18 endpoints
            openapi = client.get("/openapi.json").json()
            op_count = sum(
                1
                for path_item in openapi["paths"].values()
                for m in path_item.keys()
                if m.lower() in {"get", "post", "put", "delete", "patch"}
            )
            self.assertEqual(op_count, 18)

            # Endpoint 10 dispatch still < 500 ms
            r_disp = client.post(f"/api/routes/dispatch?db_path={db_file}", json={"water_level_delta_m": 0.0})
            self.assertEqual(r_disp.status_code, 200)
            self.assertLess(r_disp.json()["compute_ms"], 500.0)
            route_id = r_disp.json()["route_id"]

            # Endpoint 14 returns KernelSHAP 7 features & valid P3 explanation
            r_exp_rt = client.get(f"/api/explanations/route/{route_id}?db_path={db_file}")
            self.assertEqual(r_exp_rt.status_code, 200)
            rt_data = r_exp_rt.json()
            self.assertTrue(rt_data["validation_passed"])
            self.assertEqual(len(rt_data["shap_values"]), 7)
            self.assertNotIn("Awaiting Portion 3", rt_data["natural_language"])

            # Endpoint 15 returns real P3 Grad-CAM narrative (not awaiting stub)
            r_exp_bld = client.get(f"/api/explanations/building/BLD-0001?db_path={db_file}")
            self.assertEqual(r_exp_bld.status_code, 200)
            bld_data = r_exp_bld.json()
            self.assertTrue(bld_data["validation_passed"])
            self.assertNotIn("Awaiting Portion 3", bld_data["natural_language"])
            self.assertIn("gradcam_image_url", bld_data)

            # Live overlay PNG endpoint returns 404 cleanly for seeded building without .npy,
            # and 200 image/png when .npy exports exist
            r_png_404 = client.get(f"/api/explanations/building/BLD-0001/overlay.png?db_path={db_file}")
            self.assertEqual(r_png_404.status_code, 404)

            import explain.gradcam as gx
            from unittest import mock

            exp_tmp = Path(td) / "live_exports"
            exp_tmp.mkdir(parents=True, exist_ok=True)
            np.save(exp_tmp / "houston-east_LIVE_BLD-0001_activations.npy", np.ones((4, 8, 8), dtype=np.float32))
            np.save(exp_tmp / "houston-east_LIVE_BLD-0001_gradients.npy", np.ones((4, 8, 8), dtype=np.float32))
            with mock.patch.object(gx, "DEFAULT_EXPORT_DIR", exp_tmp):
                r_exp_with_png = client.get(f"/api/explanations/building/BLD-0001?db_path={db_file}")
                self.assertEqual(r_exp_with_png.status_code, 200)
                self.assertTrue(r_exp_with_png.json()["validation_passed"])
                self.assertIn("[P3 GRAD-CAM]", r_exp_with_png.json()["natural_language"])
                self.assertEqual(
                    r_exp_with_png.json()["gradcam_image_url"],
                    "/api/explanations/building/BLD-0001/overlay.png",
                )
                r_png_200 = client.get(f"/api/explanations/building/BLD-0001/overlay.png?db_path={db_file}")
                self.assertEqual(r_png_200.status_code, 200)
                self.assertEqual(r_png_200.headers.get("content-type"), "image/png")
                self.assertTrue(r_png_200.content.startswith(b"\x89PNG"))

            # Endpoint 13 What-If narrative comes from Portion 3 and respects < 500 ms
            r_wi = client.post(
                f"/api/routes/what-if?db_path={db_file}",
                json={"blocked_edges": ["E-01"], "flood_delta": 0.25},
            )
            self.assertEqual(r_wi.status_code, 200)
            self.assertLess(r_wi.json()["compute_ms"], 500.0)
            self.assertNotIn("[TEMPLATE FALLBACK] What-If dry run", r_wi.json()["narrative_text"])

    def test_zero_sql_and_no_yousef_legacy_folder_or_refs(self) -> None:
        legacy_folder = "disaster_platform " + "yousef"
        self.assertFalse((ROOT_DIR / legacy_folder).exists(), f"Dumped source folder still exists: {legacy_folder}")
        self.assertTrue((ROOT_DIR / "data" / "models" / "mistral-7b.gguf").is_file())

        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        explain_files = sorted((ROOT_DIR / "explain").glob("*.py"))
        self.assertGreaterEqual(len(explain_files), 6)
        for py_file in explain_files:
            src = py_file.read_text(encoding="utf-8")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Raw SQL '{kw}' found in {py_file.name}")

        audited_files = [
            *explain_files,
            ROOT_DIR / "main.py",
            ROOT_DIR / "models.py",
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


if __name__ == "__main__":
    unittest.main()
