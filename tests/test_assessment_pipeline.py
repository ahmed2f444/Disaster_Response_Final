"""Phase 3 Integration & TDD Tests for Portion 1 (`damage/assess.py`) & Portion 5 Backbone."""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _encode_png_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _synthetic_disaster_pair(size: int = 256) -> tuple[str, str]:
    pre = Image.new("RGB", (size, size), color=(34, 68, 42))
    post = Image.new("RGB", (size, size), color=(38, 62, 44))
    draw_pre = ImageDraw.Draw(pre)
    draw_post = ImageDraw.Draw(post)
    # Draw two crisp rectangular structures with visible post-event rooftop change
    draw_pre.rectangle([40, 40, 104, 104], fill=(180, 180, 175), outline=(220, 220, 220))
    draw_post.rectangle([40, 40, 104, 104], fill=(125, 70, 55), outline=(90, 60, 50))
    draw_pre.rectangle([140, 130, 210, 200], fill=(195, 195, 190), outline=(230, 230, 230))
    draw_post.rectangle([140, 130, 210, 200], fill=(185, 182, 178), outline=(215, 215, 210))
    return _encode_png_uri(pre), _encode_png_uri(post)


class TestPhase3AssessmentIntegration(unittest.TestCase):
    """Verify contract normalization, assessment plug-in execution, Grad-CAM persistence, and security."""

    def test_hyphenated_damage_class_normalization_in_db_and_models(self) -> None:
        import db
        import models

        self.assertEqual(db.normalize_damage_class("no-damage"), (0, "no_damage"))
        self.assertEqual(db.normalize_damage_class("minor-damage"), (1, "minor"))
        self.assertEqual(db.normalize_damage_class("major-damage"), (2, "major"))
        self.assertEqual(db.normalize_damage_class("destroyed"), (3, "destroyed"))

        bld = models.Building(
            building_id="BLD-HYP-1",
            geometry={"type": "Polygon", "coordinates": [[[-95.14, 29.81], [-95.13, 29.81], [-95.13, 29.82], [-95.14, 29.81]]]},
            damage_class="major-damage",
            confidence=0.77,
            last_updated="2026-11-10T20:00:00Z",
        )
        self.assertEqual(bld.damage_class, models.DamageClassEnum.MAJOR)
        self.assertEqual(bld.class_label, "major")

        req = models.UploadAssessmentRequest(
            pre_tile_ref="tile_pre.png",
            post_tile_ref="tile_post.png",
            sector_id="houston-east",
        )
        self.assertEqual(req.pre_tile_ref, "tile_pre.png")
        self.assertEqual(req.post_tile_ref, "tile_post.png")

    def test_metadata_only_upload_stays_pending_without_fake_buildings(self) -> None:
        import main
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "meta_only.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            client = TestClient(main.app)

            r_up = client.post(
                f"/api/assessment/upload?db_path={db_file}",
                json={"sector_id": "houston-east"},
            )
            self.assertEqual(r_up.status_code, 200)
            body = r_up.json()
            self.assertEqual(body["status"], "pending")
            r_st = client.get(f"/api/assessment/status/{body['task_id']}?db_path={db_file}")
            self.assertEqual(r_st.status_code, 200)
            self.assertEqual(r_st.json()["status"], "pending")
            self.assertEqual(r_st.json()["buildings_detected"], 20)

    def test_live_upload_inference_persists_buildings_gradcam_and_embedding(self) -> None:
        import damage.assess as assess_mod
        import main
        import seed

        pre_uri, post_uri = _synthetic_disaster_pair(256)
        footprints = [
            {"uid": "SYN-01", "polygon_px": [[40, 40], [104, 40], [104, 104], [40, 104], [40, 40]]},
            {"uid": "SYN-02", "polygon_px": [[140, 130], [210, 130], [210, 200], [140, 200], [140, 130]]},
        ]

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "live_assess.db")
            export_tmp = Path(td) / "exports"
            seed.seed_sector(db_path=db_file, num_buildings=20)
            client = TestClient(main.app)

            with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_tmp):
                r_up = client.post(
                    f"/api/assessment/upload?db_path={db_file}",
                    json={
                        "pre_disaster_image": pre_uri,
                        "post_disaster_image": post_uri,
                        "sector_id": "houston-east",
                        "footprints": footprints,
                    },
                )
            self.assertEqual(r_up.status_code, 200, r_up.text)
            up_data = r_up.json()
            self.assertEqual(up_data["status"], "completed")
            task_id = up_data["task_id"]

            r_st = client.get(f"/api/assessment/status/{task_id}?db_path={db_file}")
            self.assertEqual(r_st.status_code, 200)
            st_data = r_st.json()
            self.assertEqual(st_data["status"], "completed")
            self.assertEqual(st_data["progress_pct"], 100.0)
            self.assertEqual(st_data["buildings_detected"], 2)

            r_blds = client.get(f"/api/buildings?db_path={db_file}")
            self.assertEqual(r_blds.status_code, 200)
            assessed = [b for b in r_blds.json()["buildings"] if b["data_source"] == "resnet18_unet_pretrained"]
            self.assertEqual(len(assessed), 2)
            for bld in assessed:
                self.assertIn(bld["damage_class"], {"no_damage", "minor", "major", "destroyed"})
                self.assertTrue(0.0 <= bld["confidence"] <= 1.0)
                self.assertTrue(bld["embedding_ref"].startswith("emb://resnet18-unet/"))
                self.assertGreaterEqual(len(bld["gradcam_attribution"]), 1)

                r_exp = client.get(f"/api/explanations/building/{bld['building_id']}?db_path={db_file}")
                self.assertEqual(r_exp.status_code, 200)

            r_state = client.get(f"/api/state?db_path={db_file}")
            self.assertEqual(r_state.status_code, 200)
            state = r_state.json()
            geojson_bytes = len(
                json.dumps({"b": state["buildings_geojson"], "r": state["roads_geojson"]}, separators=(",", ":")).encode("utf-8")
            )
            self.assertLess(geojson_bytes, 100 * 1024)
            total_features = len(state["buildings_geojson"]["features"]) + len(state["roads_geojson"]["features"])
            self.assertLess(total_features, 10_000)

    def test_live_upload_without_footprints_completes_within_geojson_budget(self) -> None:
        import damage.assess as assess_mod
        import main
        import seed

        pre_uri, post_uri = _synthetic_disaster_pair(256)
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "auto_mask.db")
            export_tmp = Path(td) / "exports"
            seed.seed_sector(db_path=db_file, num_buildings=20)
            client = TestClient(main.app)

            with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_tmp):
                r_up = client.post(
                    f"/api/assessment/upload?db_path={db_file}",
                    json={
                        "pre_disaster_image": pre_uri,
                        "post_disaster_image": post_uri,
                        "sector_id": "houston-east",
                    },
                )
            self.assertEqual(r_up.status_code, 200, r_up.text)
            self.assertEqual(r_up.json()["status"], "completed")

            r_state = client.get(f"/api/state?db_path={db_file}")
            self.assertEqual(r_state.status_code, 200)
            state = r_state.json()
            geojson_bytes = len(
                json.dumps({"b": state["buildings_geojson"], "r": state["roads_geojson"]}, separators=(",", ":")).encode("utf-8")
            )
            self.assertLess(geojson_bytes, 100 * 1024)

    def test_upload_rejects_path_traversal_image_references(self) -> None:
        import main
        import seed

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "sec_upload.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            client = TestClient(main.app)

            r_bad = client.post(
                f"/api/assessment/upload?db_path={db_file}",
                json={
                    "pre_disaster_image": str(ROOT_DIR / "main.py"),
                    "post_disaster_image": str(ROOT_DIR / "db.py"),
                    "sector_id": "houston-east",
                },
            )
            self.assertIn(r_bad.status_code, (400, 403, 422))

            r_ref_bad = client.post(
                f"/api/assessment/upload?db_path={db_file}",
                json={
                    "pre_tile_ref": "../main.py",
                    "post_tile_ref": "../db.py",
                    "sector_id": "houston-east",
                },
            )
            self.assertEqual(r_ref_bad.status_code, 403)

    def test_webgis_frontend_upload_wiring_and_honest_model_labels(self) -> None:
        html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT_DIR / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("ResNet-18 U-Net", html)
        self.assertIn("unet_resnet18_best.pth", html)
        self.assertIn("selectedPreTileDataUrl", js)
        self.assertIn("selectedPostTileDataUrl", js)
        self.assertIn("FileReader", js)
        self.assertIn("pre_disaster_image", js)
        self.assertIn("post_disaster_image", js)
        self.assertIn('"no-damage": "no_damage"', js)
        self.assertIn("Decoder Saliency", js)
        self.assertNotIn("onclick=", js)

    def test_repeated_dataset_fixture_upload_is_deterministic_preserves_feedback_and_keeps_marl_consistent(self) -> None:
        import db
        import damage.assess as assess_mod
        import main
        import seed

        t0_path = ROOT_DIR / "test images" / "test_fixture_T0_pre_disaster.png"
        t1_path = ROOT_DIR / "test images" / "test_fixture_T1_post_disaster.png"
        self.assertTrue(t0_path.is_file() and t1_path.is_file(), "Bundled T0/T1 test fixtures missing")

        pre_uri = "data:image/png;base64," + base64.b64encode(t0_path.read_bytes()).decode("ascii")
        post_uri = "data:image/png;base64," + base64.b64encode(t1_path.read_bytes()).decode("ascii")

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "repeat_sat_demo.db")
            export_tmp = Path(td) / "exports"
            seed.seed_sector(db_path=db_file, num_buildings=20)
            with TestClient(main.app) as client:
                with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_tmp):
                    r1 = client.post(
                        f"/api/assessment/upload?db_path={db_file}",
                        json={"pre_disaster_image": pre_uri, "post_disaster_image": post_uri, "sector_id": "houston-east"},
                    )
                self.assertEqual(r1.status_code, 200, r1.text)
                self.assertEqual(r1.json()["status"], "completed")

                assessed_1 = sorted(
                    [b for b in db.get_buildings(db_path=db_file) if b["data_source"] == "resnet18_unet_pretrained"],
                    key=lambda item: item["building_id"],
                )
                self.assertGreater(len(assessed_1), 0)
                ids_1 = [b["building_id"] for b in assessed_1]
                routes_1 = db.get_routes(include_counterfactuals=False, db_path=db_file)
                self.assertTrue(routes_1[0]["route_id"].startswith("PLAN-P2-"))
                route_id_1 = routes_1[0]["route_id"]
                shap_1 = routes_1[0]["shap_values"]
                eta_1 = routes_1[0]["eta_seconds"]
                self.assertEqual(len(shap_1), 7)

                # Re-upload #2 (identical T0+T1, no edits) -> pure idempotent consistent capture
                with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_tmp):
                    r2 = client.post(
                        f"/api/assessment/upload?db_path={db_file}",
                        json={"pre_disaster_image": pre_uri, "post_disaster_image": post_uri, "sector_id": "houston-east"},
                    )
                self.assertEqual(r2.status_code, 200, r2.text)
                self.assertIn("Consistent capture", r2.json()["message"])
                routes_2 = db.get_routes(include_counterfactuals=False, db_path=db_file)
                self.assertEqual(routes_2[0]["route_id"], route_id_1)
                self.assertEqual(routes_2[0]["eta_seconds"], eta_1)
                self.assertEqual(routes_2[0]["shap_values"], shap_1)

                # Submit responder correction against a detected satellite building -> preserved on re-upload #3
                target_bld = ids_1[0]
                r_corr = client.post(
                    f"/api/buildings/{target_bld}/correct?db_path={db_file}",
                    json={"corrected_damage_class": "major", "responder_id": "FIELD_QA_01", "note": "Verified roof damage"},
                )
                self.assertEqual(r_corr.status_code, 200, r_corr.text)
                stats_before = db.get_feedback_stats(db_path=db_file)

                with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_tmp):
                    r3 = client.post(
                        f"/api/assessment/upload?db_path={db_file}",
                        json={"pre_disaster_image": pre_uri, "post_disaster_image": post_uri, "sector_id": "houston-east"},
                    )
                self.assertEqual(r3.status_code, 200, r3.text)

                assessed_3 = sorted(
                    [b for b in db.get_buildings(db_path=db_file) if b["data_source"] == "resnet18_unet_pretrained"],
                    key=lambda item: item["building_id"],
                )
                self.assertEqual([b["building_id"] for b in assessed_3], ids_1)
                corrected_row = db.get_building(target_bld, db_path=db_file)
                self.assertEqual(corrected_row["damage_class"], "major")
                self.assertEqual(corrected_row["confidence"], 1.0)
                self.assertEqual(db.get_feedback_stats(db_path=db_file), stats_before)

                routes_3 = db.get_routes(include_counterfactuals=False, db_path=db_file)
                route_id_3 = routes_3[0]["route_id"]
                shap_3 = routes_3[0]["shap_values"]
                eta_3 = routes_3[0]["eta_seconds"]

                # Re-upload #4 (identical T0+T1 after correction is active) -> stays consistent again
                with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_tmp):
                    r4 = client.post(
                        f"/api/assessment/upload?db_path={db_file}",
                        json={"pre_disaster_image": pre_uri, "post_disaster_image": post_uri, "sector_id": "houston-east"},
                    )
                self.assertEqual(r4.status_code, 200, r4.text)
                self.assertIn("Consistent capture", r4.json()["message"])
                routes_4 = db.get_routes(include_counterfactuals=False, db_path=db_file)
                self.assertEqual(routes_4[0]["route_id"], route_id_3)
                self.assertEqual(routes_4[0]["eta_seconds"], eta_3)
                self.assertEqual(routes_4[0]["shap_values"], shap_3)


if __name__ == "__main__":
    unittest.main()
