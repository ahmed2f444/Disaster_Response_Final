"""Granular Unit Test Suite for Portion 5 (Ahmed) across Phases 1–6.

Independently tests each module, helper function, schema guard, validation gate branch,
plug-in delegation hook, static asset split, documentation contract, and directory parity.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import db
import main
import models
import seed


class TestPhase1SkeletonParity(unittest.TestCase):
    """Phase 1: Core directory skeleton and 0 B teammate stubs match disaster_platform."""

    def test_expected_directories_and_zero_byte_stubs(self) -> None:
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
            path = ROOT_DIR / rel
            self.assertTrue(path.is_file(), f"Missing integrated module: {rel}")
            self.assertGreater(path.stat().st_size, 0, f"Integrated module {rel} must be populated")

        stub_files = [
            "scripts/download_osm.py", "scripts/download_xbd.py",
            "data/exports/.gitkeep", "data/models/.gitkeep", "data/osm/.gitkeep", "data/xbd/.gitkeep",
        ]
        self.assertEqual(len(stub_files), 6)
        for rel in stub_files:
            path = ROOT_DIR / rel
            self.assertTrue(path.is_file(), f"Missing stub: {rel}")
            self.assertEqual(path.stat().st_size, 0, f"Stub {rel} must stay 0 B")

    def test_no_legacy_src_or_hazards_folders(self) -> None:
        legacy_items = (
            "src",
            "hazards",
            "negm " + "part",
            "disaster_platform " + "yousef",
            "Dockerfile",
            "docker-compose.yml",
            "package_deliverable.bat",
        )
        for legacy in legacy_items:
            self.assertFalse((ROOT_DIR / legacy).exists(), f"Legacy item still present: {legacy}")
        self.assertTrue((ROOT_DIR / "run_demo.bat").is_file(), "Expected run_demo.bat launcher in deliverable root")


class TestPhase2ModelsDbAndSeed(unittest.TestCase):
    """Phase 2: Pydantic V2 contracts (models.py), SQLite WAL helpers (db.py), and standalone seeder (seed.py)."""

    def test_models_validation_and_section4_2_road_attributes(self) -> None:
        road_default = models.RoadEdgeState(edge_id="E-01", state=models.RoadStateEnum.OPEN)
        self.assertFalse(road_default.closure_state)
        self.assertEqual(road_default.capacity, 1.0)
        self.assertEqual(road_default.vehicle_restriction, "none")

        road_custom = models.RoadEdgeState(
            edge_id="E-03",
            state="blocked",
            closure_state=True,
            hazard_level=0.62,
            capacity=0.0,
            vehicle_restriction="closed",
        )
        self.assertTrue(road_custom.closure_state)
        self.assertEqual(road_custom.hazard_level, 0.62)

        bld_canonical = models.Building(
            building_id="BLD-CAN",
            geometry={"type": "Polygon", "coordinates": []},
            damage_class="destroyed",
            confidence=0.92,
            last_updated="2026-11-10T12:00:00Z",
        )
        self.assertEqual(bld_canonical.damage_class, models.DamageClassEnum.DESTROYED)
        self.assertEqual(bld_canonical.class_label, "destroyed")

        bld_legacy_int = models.Building(
            building_id="BLD-INT",
            geometry={"type": "Polygon", "coordinates": []},
            damage_class=2,
            confidence=0.81,
            last_updated="2026-11-10T12:00:00Z",
        )
        self.assertEqual(bld_legacy_int.damage_class, models.DamageClassEnum.MAJOR)
        self.assertEqual(bld_legacy_int.class_label, "major")

        with self.assertRaises(ValidationError):
            models.Building(
                building_id="BLD-BAD-DMG",
                geometry={"type": "Polygon", "coordinates": []},
                damage_class="catastrophic",
                confidence=0.8,
                last_updated="2026-11-10T12:00:00Z",
            )

        with self.assertRaises(ValidationError):
            models.Building(
                building_id="BLD-ERR",
                geometry={"type": "Polygon", "coordinates": []},
                damage_class="no_damage",
                confidence=1.25,  # violates le=1.0
                last_updated="2026-11-10T12:00:00Z",
            )

        with self.assertRaises(ValidationError):
            models.Route(route_id="RT-ERR", unit_id="MED-01", incident_id="INC-01", eta_seconds=-10.0)

        with self.assertRaises(ValidationError):
            models.RoadEdgeState(edge_id="E-CAP", state="open", capacity=2.5)

        with self.assertRaises(ValidationError):
            models.Explanation(
                explanation_id="EXP-ERR",
                target_type="vehicle",
                target_id="BLD-1",
                natural_language="Invalid target",
                attribution_method="shap",
                system_uncertainty=0.1,
                timestamp="2026-11-10T12:00:00Z",
            )

        with self.assertRaises(ValidationError):
            models.IncidentCreateRequest(lat=999.0, lon=-999.0)

        with self.assertRaises(ValidationError):
            models.FeedbackLog(
                target_id="BLD-1",
                prediction="minor",
                uncertainty=-0.5,
                responder_action="confirm",
            )

    def test_db_init_wal_sidecar_reset_and_legacy_column_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "wal_reset.db"
            wal_path = Path(f"{db_path}-wal")
            shm_path = Path(f"{db_path}-shm")
            db_path.write_text("stale", encoding="utf-8")
            wal_path.write_text("stale-wal", encoding="utf-8")
            shm_path.write_text("stale-shm", encoding="utf-8")

            db.init_db(db_path=str(db_path), reset=True)
            self.assertFalse(wal_path.exists())
            self.assertFalse(shm_path.exists())

            # Simulate legacy 9-column road_edges database without Section 4.2 columns
            legacy_path = Path(td) / "legacy_9col.db"
            with db.get_connection(str(legacy_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE road_edges (
                        edge_id TEXT PRIMARY KEY,
                        u_node INTEGER NOT NULL,
                        v_node INTEGER NOT NULL,
                        geom_geojson TEXT NOT NULL,
                        length_m REAL NOT NULL,
                        base_travel_sec REAL NOT NULL,
                        flood_depth_m REAL NOT NULL DEFAULT 0.0,
                        debris_score REAL NOT NULL DEFAULT 0.0,
                        edge_status TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO road_edges VALUES ('E-LEGACY', 0, 1, '{\"type\":\"LineString\",\"coordinates\":[[0,0],[1,1]]}', 100.0, 10.0, 0.0, 0.0, 'OPEN')"
                )
                conn.execute(
                    "INSERT INTO road_edges VALUES ('E-LEGACY-HAZ', 1, 2, '{\"type\":\"LineString\",\"coordinates\":[[0,0],[1,1]]}', 100.0, 10.0, 0.24, 0.15, 'HAZARDOUS')"
                )
                conn.execute(
                    "INSERT INTO road_edges VALUES ('E-LEGACY-BLK', 2, 3, '{\"type\":\"LineString\",\"coordinates\":[[0,0],[1,1]]}', 100.0, 10.0, 0.62, 0.40, 'BLOCKED')"
                )
                conn.commit()

            db.init_db(db_path=str(legacy_path), reset=False)
            edges = {e["edge_id"]: e for e in db.get_road_edges(db_path=str(legacy_path))}
            self.assertEqual(len(edges), 3)
            self.assertFalse(edges["E-LEGACY"]["closure_state"])
            self.assertEqual(edges["E-LEGACY"]["capacity"], 1.0)
            self.assertEqual(edges["E-LEGACY"]["vehicle_restriction"], "none")
            self.assertFalse(edges["E-LEGACY-HAZ"]["closure_state"])
            self.assertGreaterEqual(edges["E-LEGACY-HAZ"]["hazard_level"], 0.24)
            self.assertEqual(edges["E-LEGACY-HAZ"]["capacity"], 0.5)
            self.assertEqual(edges["E-LEGACY-HAZ"]["vehicle_restriction"], "high_clearance_only")
            self.assertTrue(edges["E-LEGACY-BLK"]["closure_state"])
            self.assertEqual(edges["E-LEGACY-BLK"]["capacity"], 0.0)
            self.assertEqual(edges["E-LEGACY-BLK"]["vehicle_restriction"], "closed")

            # Simulate legacy integer buildings.damage_class table migrating cleanly to Section 3.2 enum strings
            legacy_bld_path = Path(td) / "legacy_int_bld.db"
            with db.get_connection(str(legacy_bld_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE buildings (
                        building_id TEXT PRIMARY KEY,
                        geom_geojson TEXT NOT NULL,
                        centroid_lat REAL NOT NULL,
                        centroid_lon REAL NOT NULL,
                        damage_class INTEGER NOT NULL CHECK (damage_class BETWEEN 0 AND 3),
                        class_label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        max_prob REAL NOT NULL DEFAULT 0.90,
                        status_flag TEXT NOT NULL DEFAULT 'CALIBRATED',
                        gradcam_attribution_json TEXT NOT NULL DEFAULT '[]',
                        shap_top3_json TEXT NOT NULL DEFAULT '[]',
                        embedding_ref TEXT NOT NULL DEFAULT '',
                        last_updated TEXT NOT NULL,
                        data_source TEXT NOT NULL DEFAULT 'xBD_houston_east'
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO buildings (building_id, geom_geojson, centroid_lat, centroid_lon, damage_class, class_label, confidence, last_updated) "
                    "VALUES ('BLD-LEG', '{\"type\":\"Polygon\",\"coordinates\":[]}', 29.8, -95.1, 3, 'destroyed', 0.94, '2026-11-10T12:00:00Z')"
                )
                conn.execute(
                    "INSERT INTO buildings (building_id, geom_geojson, centroid_lat, centroid_lon, damage_class, class_label, confidence, status_flag, last_updated) "
                    "VALUES ('BLD-LEG-LOW', '{\"type\":\"Polygon\",\"coordinates\":[]}', 29.8, -95.1, 1, 'minor', 0.52, 'LOW_CONFIDENCE', '2026-11-10T12:00:00Z')"
                )
                conn.commit()

            db.init_db(db_path=str(legacy_bld_path), reset=False)
            migrated = db.get_building("BLD-LEG", db_path=str(legacy_bld_path))
            self.assertIsNotNone(migrated)
            self.assertEqual(migrated["damage_class"], "destroyed")
            self.assertEqual(migrated["damage_tier"], 3)
            self.assertEqual(migrated["class_label"], "destroyed")
            migrated_low = db.get_building("BLD-LEG-LOW", db_path=str(legacy_bld_path))
            self.assertEqual(migrated_low["confidence"], 0.52)

    def test_db_crud_helpers_empty_stats_and_audit_chain_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "crud.db")
            db.init_db(db_path=db_file, reset=True)

            # Empty DB feedback stats must safely return correction_rate 0.0
            empty_stats = db.get_feedback_stats(db_path=db_file)
            self.assertEqual(empty_stats, {"total_count": 0, "confirmed": 0, "corrected": 0, "skipped": 0, "correction_rate": 0.0})

            ok0, cnt0, _ = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(ok0)
            self.assertEqual(cnt0, 0)

            # Building assessment & damaged queries (string canonical + legacy integer thresholds)
            db.insert_building(
                {
                    "building_id": "BLD-U1",
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "centroid_lat": 29.8,
                    "centroid_lon": -95.1,
                    "damage_class": "minor",
                    "class_label": "minor",
                    "confidence": 0.6,
                },
                db_path=db_file,
            )
            self.assertEqual(len(db.get_damaged_buildings(min_damage_class="major", db_path=db_file)), 0)
            self.assertEqual(len(db.get_damaged_buildings(min_damage_class=2, db_path=db_file)), 0)
            updated = db.update_building_assessment(
                "BLD-U1",
                "destroyed",
                "destroyed",
                0.91,
                "CALIBRATED",
                "CONVNEXT_TEST",
                [{"feature": "roof", "value": 0.8}],
                gradcam_attribution=[{"region": {"type": "Polygon", "coordinates": []}, "weight": 0.94}],
                embedding_ref="emb://dinov2/BLD-U1",
                db_path=db_file,
            )
            self.assertTrue(updated)
            self.assertEqual(len(db.get_damaged_buildings(min_damage_class="major", db_path=db_file)), 1)
            self.assertEqual(len(db.get_damaged_buildings(min_damage_class=2, db_path=db_file)), 1)
            b = db.get_building("BLD-U1", db_path=db_file)
            self.assertEqual(b["damage_class"], "destroyed")
            self.assertEqual(b["damage_tier"], 3)
            self.assertEqual(b["embedding_ref"], "emb://dinov2/BLD-U1")
            self.assertEqual(b["gradcam_attribution"][0]["weight"], 0.94)
            self.assertTrue(
                db.update_building_assessment(
                    "BLD-U1", "", "major", 0.88, "CALIBRATED", "CONVNEXT_TEST", [], db_path=db_file
                )
            )
            self.assertEqual(db.get_building("BLD-U1", db_path=db_file)["damage_class"], "major")

            # Road edge update + filtering
            db.insert_road_edge(
                {
                    "edge_id": "E-U1",
                    "u_node": 0,
                    "v_node": 1,
                    "geometry": {"type": "LineString", "coordinates": [[-95.18, 29.85], [-95.14, 29.85]]},
                    "edge_status": "OPEN",
                },
                db_path=db_file,
            )
            db.update_road_edge(
                "E-U1",
                0.65,
                0.4,
                "BLOCKED",
                closure_state=True,
                hazard_level=0.65,
                capacity=0.0,
                vehicle_restriction="closed",
                db_path=db_file,
            )
            self.assertEqual(len(db.get_road_edges(status="BLOCKED", db_path=db_file)), 1)
            self.assertEqual(len(db.get_road_edges(status="OPEN", db_path=db_file)), 0)

            # Incident status update
            db.insert_incident(
                {"incident_id": "INC-U1", "location": {"lat": 29.8, "lon": -95.1}, "status": "open"},
                db_path=db_file,
            )
            self.assertTrue(db.update_incident_status("INC-U1", "resolved", db_path=db_file))
            self.assertEqual(db.get_incident("INC-U1", db_path=db_file)["status"], "resolved")

            # Route counterfactual isolation
            db.insert_route({"route_id": "RT-LIVE", "is_counterfactual": 0, "eta_seconds": 100.0}, db_path=db_file)
            db.insert_route({"route_id": "RT-WHATIF", "is_counterfactual": 1, "eta_seconds": 250.0}, db_path=db_file)
            self.assertEqual(len(db.get_routes(include_counterfactuals=False, db_path=db_file)), 1)
            self.assertEqual(len(db.get_routes(include_counterfactuals=True, db_path=db_file)), 2)

            # Explanations boolean normalization
            db.insert_explanation(
                {
                    "explanation_id": "EXP-U1",
                    "target_type": "route",
                    "target_id": "RT-LIVE",
                    "natural_language": "Deterministic route explanation ETA 100.0s.",
                    "attribution_method": "shap",
                    "system_uncertainty": 0.1,
                    "llm_used": True,
                    "validation_passed": False,
                },
                db_path=db_file,
            )
            exp = db.get_explanation_by_target("route", "RT-LIVE", db_path=db_file)
            self.assertIs(exp["llm_used"], True)
            self.assertIs(exp["validation_passed"], False)
            self.assertEqual(len(db.get_explanations(target_id="RT-LIVE", db_path=db_file)), 1)

            # Feedback log stats & Stage 10 audit chain sequence gap check
            fb_res = db.log_feedback(
                {
                    "target_id": "BLD-U1",
                    "prediction": "minor",
                    "uncertainty": 0.4,
                    "responder_action": "confirm",
                    "timestamp": "2026-11-10T10:00:00Z",
                },
                db_path=db_file,
            )
            self.assertEqual(fb_res["timestamp"], "2026-11-10T10:00:00Z")
            db.log_feedback({"target_id": "BLD-U1", "prediction": "minor", "uncertainty": 0.4, "responder_action": "correct", "corrected_label": "major"}, db_path=db_file)
            stats = db.get_feedback_stats(db_path=db_file)
            self.assertEqual(stats["correction_rate"], 0.5)
            self.assertEqual(len(db.get_feedback_logs(limit=5, db_path=db_file)), 2)

            db.append_audit_event("PORTION4", "MODEL_PROMOTED", {"version": "v1.1"}, db_path=db_file)
            db.append_audit_event("COMMANDER:CMD_01", "APPROVED", {"route_id": "RT-LIVE"}, route_id="RT-LIVE", db_path=db_file)
            ok_chain, count_chain, _ = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(ok_chain)
            self.assertEqual(count_chain, 2)

            with db.get_connection(db_file) as conn:
                conn.execute("DELETE FROM audit_log WHERE seq_id = 1")
                conn.commit()
            ok_gap, seq_gap, msg_gap = db.verify_audit_chain(db_path=db_file)
            self.assertFalse(ok_gap)
            self.assertEqual(seq_gap, 1)
            self.assertIn("Sequence gap", msg_gap)

    def test_seed_minimum_20_and_baseline_220_sectors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db20 = str(Path(td) / "s20.db")
            res20 = seed.seed_sector(db_path=db20, num_buildings=20)
            self.assertEqual(res20["buildings"], 20)
            self.assertEqual(res20["roads_open"], 19)
            self.assertEqual(res20["roads_hazardous"], 7)
            self.assertEqual(res20["roads_blocked"], 0)
            self.assertEqual(res20["initial_plan_eta_sec"], 810)

            routes = db.get_routes(db_path=db20)
            self.assertEqual(routes[0]["plan_id"], "PLAN-BASELINE-01")
            self.assertFalse(routes[0]["mask_altered"])
            self.assertEqual(routes[0]["ppo_proposed_path"], routes[0]["masked_path"])


class TestPhase3MainBackboneAndGuards(unittest.TestCase):
    """Phase 3: main.py plugin hooks, GeoJSON budget, Validation Gate, 404/422/501 handling."""

    def test_load_plugin_callable_and_drift_delegation(self) -> None:
        self.assertTrue(callable(main._load_plugin_callable("routing.generate", "compute_multi_unit_routes")))
        self.assertTrue(callable(main._load_plugin_callable("feedback.drift", "compute_drift_distance")))
        self.assertTrue(callable(main._load_plugin_callable("explain.generate", "explain_route")))
        self.assertTrue(callable(main._load_plugin_callable("explain.generate", "explain_building")))
        self.assertIsNone(main._load_plugin_callable("nonexistent.module.xyz", "fn"))

        self.assertGreater(main._compute_drift_distance({"total_count": 20, "corrected": 12, "correction_rate": 0.6}), 0.0)
        with mock.patch.object(main, "_load_plugin_callable", return_value=lambda stats: 0.42):
            self.assertEqual(main._compute_drift_distance({"correction_rate": 0.5}), 0.42)

    def test_enforce_geojson_budget_caps_features_and_bytes(self) -> None:
        huge_poly = {
            "type": "Polygon",
            "coordinates": [[[round(-95.1 + i * 0.0001, 5), 29.8] for i in range(45)]],
        }
        many_buildings = [
            {
                "type": "Feature",
                "geometry": huge_poly,
                "properties": {
                    "building_id": f"BLD-{i:05d}",
                    "damage_class": "major",
                    "class_label": "major",
                    "confidence": 0.85,
                    "max_prob": 0.85,
                    "status_flag": "CALIBRATED",
                    "embedding_ref": "emb://dinov2/huge/" + ("x" * 120),
                    "data_source": "xBD_houston_east",
                },
            }
            for i in range(10_050)
        ]
        edges = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[-95.18, 29.85], [-95.14, 29.85]]},
                "properties": {"edge_id": f"E-{i:02d}", "edge_status": "OPEN"},
            }
            for i in range(12)
        ]

        bounded_b, bounded_e = main._enforce_geojson_budget(many_buildings, edges)
        self.assertEqual(len(bounded_e), 12)
        self.assertLess(len(bounded_b) + len(bounded_e), main.MAX_GEOJSON_FEATURES)

        serialized_bytes = len(
            json.dumps(
                {
                    "buildings": {"type": "FeatureCollection", "features": bounded_b},
                    "roads": {"type": "FeatureCollection", "features": bounded_e},
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.assertLessEqual(serialized_bytes, main.MAX_GEOJSON_BYTES)

        # Oversized road edge collections must also stay bounded within Section 4.5 budget
        many_edges = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[-95.18, 29.85], [-95.14, 29.85]]},
                "properties": {"edge_id": f"E-{i:05d}", "edge_status": "OPEN"},
            }
            for i in range(10_050)
        ]
        bounded_b2, bounded_e2 = main._enforce_geojson_budget(many_buildings[:5], many_edges)
        self.assertLess(len(bounded_b2) + len(bounded_e2), main.MAX_GEOJSON_FEATURES)
        edge_budget_bytes = len(
            json.dumps(
                {
                    "buildings": {"type": "FeatureCollection", "features": bounded_b2},
                    "roads": {"type": "FeatureCollection", "features": bounded_e2},
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        self.assertLessEqual(edge_budget_bytes, main.MAX_GEOJSON_BYTES)

    def test_route_validation_gate_branches(self) -> None:
        route = {
            "route_id": "PLAN-1",
            "eta_seconds": 810.0,
            "mask_reason": "Avoid E-03",
            "road_states": [{"edge_id": "E-01"}, {"edge_id": "E-02"}, {"edge_id": "E-03"}],
            "routes": [{"edge_ids": ["E-01", "E-02"], "ppo_edge_ids": ["E-03"]}],
        }
        # Valid
        txt, ok = main._validate_route_explanation("Route uses E-01 and E-02 with ETA 810.0s.", route, llm_used=False)
        self.assertTrue(ok)
        self.assertIn("810.0s", txt)

        # Short text fails
        _, ok_short = main._validate_route_explanation("too short", route, llm_used=False)
        self.assertFalse(ok_short)

        # Wrong ETA fails
        txt_eta, ok_eta = main._validate_route_explanation("Detour ETA 420.0s via E-01.", route, llm_used=True)
        self.assertFalse(ok_eta)
        self.assertIn("[TEMPLATE FALLBACK]", txt_eta)

        # Unknown edge fails
        _, ok_edge = main._validate_route_explanation("Detour via unknown E-99 corridor.", route, llm_used=False)
        self.assertFalse(ok_edge)

        # LLM self-reported % confidence fails
        _, ok_pct = main._validate_route_explanation("Route ETA 810.0s with 98% LLM certainty.", route, llm_used=True)
        self.assertFalse(ok_pct)

        # Wrong ETA with "is" phrasing fails, and contradicting multiple ETAs fail
        _, ok_eta_is = main._validate_route_explanation("Route ETA is 120.0s via E-01.", route, llm_used=False)
        self.assertFalse(ok_eta_is)
        _, ok_eta_multi = main._validate_route_explanation("Route ETA 810.0s (actually ETA 120.0s) via E-01.", route, llm_used=False)
        self.assertFalse(ok_eta_multi)

        # Stored False stays False
        _, ok_stored = main._validate_route_explanation("Route uses E-01 ETA 810.0s.", route, llm_used=False, stored_validation_passed=False)
        self.assertFalse(ok_stored)

    def test_building_validation_gate_branches(self) -> None:
        bld = {"class_label": "minor", "confidence": 0.82}

        # Valid comparative prose passes (both conjunction & comma-separated negations)
        _, ok_cmp = main._validate_building_explanation(
            "ConvNeXt-Tiny classified minor (not major or destroyed). Confidence: 82.0%.",
            bld,
            llm_used=False,
        )
        self.assertTrue(ok_cmp)
        _, ok_comma = main._validate_building_explanation(
            "Building is minor (not no_damage, major, or destroyed). Confidence: 82.0%.",
            bld,
            llm_used=False,
        )
        self.assertTrue(ok_comma)

        # LLM-stated percentage confidence fails even if matching numerical value
        _, ok_llm_pct = main._validate_building_explanation(
            "Building shows minor damage at 82.0% certainty.",
            bld,
            llm_used=True,
        )
        self.assertFalse(ok_llm_pct)

        # Unnegated wrong label fails
        txt_wrong, ok_wrong = main._validate_building_explanation(
            "Building classified as destroyed with 82.0% confidence.",
            bld,
            llm_used=True,
        )
        self.assertFalse(ok_wrong)
        self.assertIn("[TEMPLATE FALLBACK]", txt_wrong)

        # Mismatched percentage fails
        _, ok_pct = main._validate_building_explanation(
            "Building shows minor damage at 45.0% confidence.",
            bld,
            llm_used=False,
        )
        self.assertFalse(ok_pct)

        # LLM mentioning confidence without verifiable percentage fails
        _, ok_unverif = main._validate_building_explanation(
            "High confidence minor damage assessment by assistant.",
            bld,
            llm_used=True,
        )
        self.assertFalse(ok_unverif)

    def test_plugin_transition_from_501_to_200_and_error_codes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "plugin_transition.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)
            client = TestClient(main.app)

            # When routing plugin is absent -> 501 shell guard
            with mock.patch.object(main, "_load_plugin_callable", return_value=None):
                self.assertEqual(client.post(f"/api/routes/dispatch?db_path={db_file}", json={}).status_code, 501)
                self.assertEqual(client.post(f"/api/routes/what-if?db_path={db_file}", json={}).status_code, 501)
                self.assertEqual(client.post(f"/api/what-if?db_path={db_file}", json={}).status_code, 501)

            # Live Portion 2 routing integration -> 200 OK
            r_live_disp = client.post(f"/api/routes/dispatch?db_path={db_file}", json={"water_level_delta_m": 0.0})
            self.assertEqual(r_live_disp.status_code, 200)
            self.assertLess(r_live_disp.json()["compute_ms"], 500.0)
            r_live_wi = client.post(
                f"/api/routes/what-if?db_path={db_file}",
                json={"blocked_edges": ["E-04"], "flood_delta": 0.25},
            )
            self.assertEqual(r_live_wi.status_code, 200)
            self.assertLess(r_live_wi.json()["compute_ms"], 500.0)

            # 404 & 422 error checks + canonical vs tier alias damage_class filtering
            r_dest = client.get(f"/api/buildings?damage_class=destroyed&db_path={db_file}")
            r_tier3 = client.get(f"/api/buildings?damage_class=3&db_path={db_file}")
            self.assertEqual(r_dest.status_code, 200)
            self.assertEqual(r_tier3.status_code, 200)
            self.assertEqual(r_dest.json()["total_count"], r_tier3.json()["total_count"])
            self.assertTrue(all(b["damage_class"] == "destroyed" for b in r_dest.json()["buildings"]))
            self.assertEqual(client.get(f"/api/buildings?damage_class=bogus_tier&db_path={db_file}").status_code, 422)
            self.assertEqual(client.get(f"/api/buildings/BLD-9999?db_path={db_file}").status_code, 404)
            self.assertEqual(client.get(f"/api/routes/RT-NONE?db_path={db_file}").status_code, 404)
            self.assertEqual(client.post(f"/api/routes/RT-NONE/approve?db_path={db_file}", json={"commander_id": "CMD_1"}).status_code, 404)
            self.assertEqual(client.post(f"/api/routes/RT-NONE/reject?db_path={db_file}", json={"commander_id": "CMD_1"}).status_code, 404)
            self.assertEqual(client.post(f"/api/routes/PLAN-BASELINE-01/approve?db_path={db_file}", json={"commander_id": "X"}).status_code, 422)
            self.assertEqual(client.post(f"/api/routes/what-if?db_path={db_file}", json={"flood_delta": 99.0}).status_code, 422)

            # Simulate teammate plug-in callables -> automatic 200 transition
            mock_plan = {
                "route_id": "PLAN-P2-MOCK",
                "plan_id": "PLAN-P2-MOCK",
                "unit_id": "MED-01",
                "incident_id": "INC-01",
                "eta_seconds": 300.0,
                "total_eta_sec": 300.0,
                "blocked_edges": ["E-03"],
                "routes": [],
            }
            mock_whatif = {
                "plan_id": "WHATIF-P2",
                "compute_ms": 3.5,
                "flood_delta_m": 0.25,
                "newly_blocked_edges": ["E-03", "E-04"],
                "baseline_eta_sec": 810.0,
                "whatif_eta_sec": 900.0,
                "eta_delta_sec": 90.0,
                "routes": [],
            }

            def _fake_loader(mod_name: str, attr_name: str):
                if (mod_name, attr_name) == ("routing.generate", "compute_multi_unit_routes"):
                    return lambda db_path: mock_plan
                if (mod_name, attr_name) == ("routing.graph", "update_road_hazards"):
                    return lambda db_path, water_level_delta_m: {"OPEN": 8}
                if (mod_name, attr_name) == ("routing.generate", "run_counterfactual"):
                    return lambda blocked_edges, flood_delta, db_path: mock_whatif
                if (mod_name, attr_name) == ("explain.generate", "generate_narrative"):
                    return lambda res, use_llm_stretch=False: ("P3 narrative ETA 900.0s", "JINJA2_TEMPLATE")
                return None

            with mock.patch.object(main, "_load_plugin_callable", side_effect=_fake_loader):
                r_disp = client.post(f"/api/routes/dispatch?db_path={db_file}", json={"water_level_delta_m": 0.2})
                self.assertEqual(r_disp.status_code, 200)
                self.assertEqual(r_disp.json()["route_id"], "PLAN-P2-MOCK")

                r_wi = client.post(f"/api/routes/what-if?db_path={db_file}", json={"blocked_edges": ["E-04"], "flood_delta": 0.25})
                self.assertEqual(r_wi.status_code, 200)
                self.assertEqual(r_wi.json()["whatif_eta_sec"], 900.0)
                self.assertEqual(r_wi.json()["narrative_text"], "P3 narrative ETA 900.0s")

            # Reset sector works cleanly
            r_rst = client.post(f"/api/reset-sector?db_path={db_file}")
            self.assertEqual(r_rst.status_code, 200)
            self.assertEqual(r_rst.json()["status"], "reset_success")

    def test_security_path_traversal_cors_and_endpoint_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "sec_guard.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)
            client = TestClient(main.app)

            # 1. Path traversal & arbitrary file access/creation denied with 400
            for bad_path in ("../../outside.db", "../../../etc/passwd", "C:/Windows/win.ini", f"{td}/bad_ext.txt"):
                self.assertEqual(client.get(f"/api/state?db_path={bad_path}").status_code, 400)
                self.assertEqual(client.get(f"/api/buildings?db_path={bad_path}").status_code, 400)
                self.assertEqual(client.post(f"/api/reset-sector?db_path={bad_path}").status_code, 400)

            # 2. CORS preflight denies untrusted external origins & disables credentials with wildcard
            preflight = client.options(
                "/api/state",
                headers={
                    "Origin": "https://evil-attacker.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            self.assertNotEqual(preflight.headers.get("access-control-allow-origin"), "https://evil-attacker.example")
            self.assertNotEqual(preflight.headers.get("access-control-allow-origin"), "*")
            self.assertNotEqual(preflight.headers.get("access-control-allow-credentials"), "true")

            # 3. Missing route explanation must return 404 (consistent with route/building detail endpoints)
            self.assertEqual(client.get(f"/api/explanations/route/RT-NONE?db_path={db_file}").status_code, 404)

            # 4. Legacy hidden /api/approve and /api/reject validate route existence & commander_id cleanly
            self.assertEqual(
                client.post(f"/api/approve?db_path={db_file}", json={"plan_id": "RT-NONE", "commander_id": "CMD_1"}).status_code,
                404,
            )
            self.assertEqual(
                client.post(f"/api/reject?db_path={db_file}", json={"plan_id": "PLAN-BASELINE-01", "commander_id": "X"}).status_code,
                422,
            )

            # 5. Uninitialized valid temp .db works on /api/assessment/upload without 500
            fresh_db = str(Path(td) / "fresh_upload.db")
            r_up = client.post(f"/api/assessment/upload?db_path={fresh_db}", json={"sector_id": "houston-east"})
            self.assertEqual(r_up.status_code, 200)

            # 6. Invalid plugin payload (negative eta_seconds) yields 502 instead of unhandled 500
            bad_plugin_plan = {
                "route_id": "PLAN-BAD",
                "unit_id": "MED-01",
                "incident_id": "INC-01",
                "eta_seconds": -5.0,
            }
            with mock.patch.object(
                main,
                "_load_plugin_callable",
                side_effect=lambda mod, fn: (lambda db_path: bad_plugin_plan) if (mod, fn) == ("routing.generate", "compute_multi_unit_routes") else None,
            ):
                r_bad_disp = client.post(f"/api/routes/dispatch?db_path={db_file}", json={})
                self.assertEqual(r_bad_disp.status_code, 502)


class TestPhase4StaticFrontend(unittest.TestCase):
    """Phase 4 / Task 3: Split static/ assets, offline Leaflet v1.9.4 serving, and same-origin JS wiring."""

    def test_static_assets_and_same_origin_fix(self) -> None:
        client = TestClient(main.app)
        for route in ("/", "/static/style.css", "/static/app.js", "/static/leaflet.js", "/static/leaflet.css"):
            self.assertEqual(client.get(route).status_code, 200, f"Asset route failed: {route}")

        # Legacy Three.js bundles must be removed and return 404
        for legacy_route in ("/static/three.min.js", "/static/OrbitControls.js", "/three.min.js", "/OrbitControls.js"):
            self.assertEqual(client.get(legacy_route).status_code, 404, f"Legacy Three.js route still active: {legacy_route}")
        self.assertFalse((ROOT_DIR / "static" / "three.min.js").exists())
        self.assertFalse((ROOT_DIR / "static" / "OrbitControls.js").exists())

        leaflet_js = client.get("/static/leaflet.js").text
        self.assertIn("1.9.4", leaflet_js)

        html = client.get("/").text
        self.assertNotIn("<style>", html)
        self.assertIn('href="/static/leaflet.css"', html)
        self.assertIn('href="/static/style.css"', html)
        self.assertIn('src="/static/leaflet.js"', html)
        self.assertIn('src="/static/app.js"', html)
        self.assertIn('id="leafletMap"', html)
        for forbidden_html in ("three.min.js", "OrbitControls.js", "threeCanvas", "cdnjs.cloudflare.com", "cdn.jsdelivr.net"):
            self.assertNotIn(forbidden_html, html)

        js = client.get("/static/app.js").text
        self.assertIn("window.location.protocol === 'file:'", js)
        self.assertNotIn("window.location.port !== '8000'", js)
        self.assertIn("L.map(", js)
        self.assertIn("L.circleMarker(", js)
        self.assertIn("L.canvas(", js)
        self.assertIn("performance.now()", js)
        self.assertIn("window.__lastMapRenderMs", js)
        self.assertIn("renderState(", js)
        self.assertIn("hudRenderMsVal", js)
        self.assertIn("hudGeoJsonKbVal", js)
        self.assertNotIn("THREE.", js)
        self.assertIn("/api/routes/what-if", js)
        self.assertIn("res.status === 501", js)
        self.assertIn("s3UncertaintyBadge", js)


class TestPhase5And6DocsConfigAndCleanup(unittest.TestCase):
    """Phases 5 & 6: requirements.txt, .gitignore, docs/api.md endpoint parity, and no stale refs."""

    def test_requirements_and_gitignore_parity(self) -> None:
        reqs = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
        for pkg in (
            "fastapi", "uvicorn", "pydantic", "python-multipart", "numpy", "pillow",
            "requests", "python-dotenv", "networkx", "gymnasium", "stable-baselines3",
            "matplotlib", "pytest", "httpx",
        ):
            self.assertIn(pkg, reqs)
        for dropped in ("jinja2", "reportlab"):
            self.assertNotIn(dropped, reqs)

        for mod_name in (
            "fastapi", "uvicorn", "pydantic", "python_multipart", "numpy", "PIL",
            "requests", "dotenv", "networkx", "gymnasium", "stable_baselines3",
            "matplotlib", "pytest", "httpx",
        ):
            self.assertIsNotNone(importlib.import_module(mod_name), f"Failed importing {mod_name}")

        gi = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("db.sqlite", "data/*.db-wal", "data/*.db-shm", "data/xbd/", "data/models/", "*.pt", "*.gguf", ".venv/", "!static/images/*.png"):
            self.assertIn(pattern, gi)

    def test_api_md_documents_all_18_openapi_paths_and_no_stale_refs(self) -> None:
        api_doc = (ROOT_DIR / "docs" / "api.md").read_text(encoding="utf-8")
        openapi_paths = set(main.app.openapi().get("paths", {}).keys())
        self.assertEqual(len(openapi_paths), 17)  # 17 unique paths with 18 HTTP method operations
        for path in openapi_paths:
            self.assertIn(path, api_doc, f"Endpoint path {path} missing from docs/api.md")

        readme_doc = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("leaflet", readme_doc.lower())
        self.assertIn("no_damage", api_doc)
        self.assertIn("destroyed", api_doc)
        self.assertIn("leaflet", api_doc.lower())

        for fname in ("README.md", "docs/api.md", "main.py", "db.py", "models.py", "seed.py", "static/index.html", "static/app.js"):
            txt = (ROOT_DIR / fname).read_text(encoding="utf-8")
            for stale in (
                "src/backend",
                "src.backend",
                "seed_mock_sector",
                "web_app.py",
                "native_desktop_app",
                "generate_workflow_pdf",
                "three.min.js",
                "OrbitControls.js",
                "threeCanvas",
                "Three.js",
            ):
                self.assertNotIn(stale, txt, f"Stale reference '{stale}' found in {fname}")


class TestPhase7UnifiedDocCompletenessAndUiWiring(unittest.TestCase):
    """Phase 7: Unified System Document Section 4.2, Section 7 (+-0.5s gate), Stage 8/10 & 6-screen UI wiring."""

    def test_section4_2_road_state_mapping_and_stage10_model_audit(self) -> None:
        self.assertEqual(db.derive_road_display_state(False, 0.04, 1.0), "open")
        self.assertEqual(db.derive_road_display_state(False, 0.24, 0.5), "hazardous")
        self.assertEqual(db.derive_road_display_state(True, 0.04, 1.0), "blocked")
        self.assertEqual(db.derive_road_display_state(False, 0.04, 0.0), "blocked")

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "stage10.db")
            db.init_db(db_path=db_file, reset=True)
            entry = db.record_model_transition(
                model_version="ConvNeXt-Tiny-v1.1",
                adapter_version="lora-harvey-a1",
                action="MODEL_PROMOTED",
                gate_checks={"f1_improved": True, "forgetting_ok": True, "geo_holdout": True, "calibration": True, "latency_ok": True},
                db_path=db_file,
            )
            self.assertEqual(entry["event_type"], "MODEL_PROMOTED")
            ok_chain, count, _ = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(ok_chain)
            self.assertEqual(count, 1)

    def test_section7_half_second_eta_gate_and_gtf_prohibitions(self) -> None:
        route = {
            "route_id": "PLAN-1",
            "eta_seconds": 810.0,
            "mask_altered": True,
            "mask_reason": "Avoid E-03",
            "road_states": [{"edge_id": "E-01"}, {"edge_id": "E-02"}, {"edge_id": "E-03"}],
            "routes": [{"edge_ids": ["E-01", "E-02"], "ppo_edge_ids": ["E-03"]}],
        }
        # Within +-0.5s passes
        _, ok_within = main._validate_route_explanation("Route uses E-01 ETA 810.4s.", route, llm_used=True)
        self.assertTrue(ok_within)
        # Outside +-0.5s (0.8s drift) must fail per Section 7 +-0.5s gate
        txt_outside, ok_outside = main._validate_route_explanation("Route uses E-01 ETA 810.8s.", route, llm_used=True)
        self.assertFalse(ok_outside)
        self.assertIn("[TEMPLATE FALLBACK]", txt_outside)

        # GTF prohibitions: hedging, ungrounded confidence without %, PPO override, mask_altered contradiction
        for bad_llm in (
            "Route uses E-01 ETA 810.0s and is probably safe.",
            "Route uses E-01 ETA 810.0s with high confidence.",
            "Override the PPO route and take E-01 ETA 810.0s.",
            "Route uses E-01 ETA 810.0s; mask did not alter the path.",
        ):
            _, ok_llm = main._validate_route_explanation(bad_llm, route, llm_used=True)
            self.assertFalse(ok_llm, f"Expected GTF gate rejection for: {bad_llm}")

    def test_whatif_endpoint_runs_validation_gate_on_plugin_narrative(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "whatif_gate.db")
            seed.seed_sector(db_path=db_file, num_buildings=25)
            client = TestClient(main.app)

            mock_whatif = {
                "plan_id": "WHATIF-GATE",
                "compute_ms": 2.7,
                "flood_delta_m": 0.25,
                "newly_blocked_edges": ["E-04"],
                "baseline_eta_sec": 810.0,
                "whatif_eta_sec": 900.0,
                "eta_delta_sec": 90.0,
                "routes": [],
            }

            def _fake_loader(mod_name: str, attr_name: str):
                if (mod_name, attr_name) == ("routing.generate", "run_counterfactual"):
                    return lambda blocked_edges, flood_delta, db_path: mock_whatif
                if (mod_name, attr_name) == ("explain.generate", "generate_narrative"):
                    return lambda res, use_llm_stretch=False: ("Hallucinated ETA 420.0s with 99% certainty", "LLM_ADVISORY")
                return None

            with mock.patch.object(main, "_load_plugin_callable", side_effect=_fake_loader):
                resp = client.post(f"/api/routes/what-if?db_path={db_file}", json={"blocked_edges": ["E-04"], "flood_delta": 0.25})
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertIn("[TEMPLATE FALLBACK]", data["narrative_text"])
                self.assertEqual(data["narrative_mode"], "JINJA2_TEMPLATE")

    def test_six_screen_navigation_airgapped_html_and_app_js_consistency(self) -> None:
        html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT_DIR / "static" / "app.js").read_text(encoding="utf-8")

        # 1. All 6-screen navigation buttons exist in index.html & all JS getElementById targets exist
        for nav_id in ("navScreen1", "navScreen2", "navScreen3", "navScreen4", "navScreen5"):
            self.assertIn(f'id="{nav_id}"', html, f"Missing screen switcher button {nav_id}")
        self.assertIn("openApprovalModalManual()", html)
        self.assertNotIn("fonts.googleapis.com", html)
        self.assertNotIn("$T=1.28$", html)

        html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', html))
        js_dynamic_ids = set(re.findall(r'id=["\']([^"\']+)["\']', js))
        js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', js))
        self.assertEqual(js_ids - (html_ids | js_dynamic_ids), set(), f"Orphaned JS getElementById calls: {js_ids - (html_ids | js_dynamic_ids)}")

        # Every inline HTML handler must exist in app.js
        inline_fns = set(re.findall(r'on(?:click|change)=["\']([A-Za-z_]\w*)\(', html))
        for fn_name in inline_fns:
            self.assertTrue(
                (f"function {fn_name}" in js) or (f"window.{fn_name}" in js),
                f"Missing JS handler '{fn_name}' referenced in static/index.html",
            )

        # 2. Mismatch 1: Building inspection uses Grad-CAM, not per-building SHAP
        self.assertNotIn("Top-3 SHAP Attributions (ConvNeXt-Tiny)", js)
        self.assertIn("gradcam_attribution", js)

        # 3. Screen 4 1-tap Confirm / Correct / Skip + XSS escapeHtml
        self.assertIn("submitBuildingFeedback('confirm')", html)
        self.assertIn("submitBuildingFeedback('skip')", html)
        self.assertIn("function escapeHtml(", js)

        # 4. Coordinates in app.js NODE_META match seed.py NODES
        for node_id, (lon, lat) in seed.NODES.items():
            self.assertIn(f"{node_id}: {{", js)
            self.assertIn(f"lon: {lon:.3f}", js)
            self.assertIn(f"lat: {lat:.3f}", js)

    def test_qa_audit_fixes_and_regressions(self) -> None:
        """QA regression suite covering P5 contracts, security guards, honest stubs, and UI fixes."""
        # 1. Validation gate regex supports 'ETA of N seconds' & building 'no damage' contradictions
        route_ref = {"route_id": "PLAN-BASELINE-01", "eta_seconds": 810.0, "road_states": [], "routes": []}
        _, ok_bad_eta = main._validate_route_explanation("Follow E-01 for an ETA of 120 seconds.", route_ref, llm_used=False)
        self.assertFalse(ok_bad_eta)
        _, ok_good_eta = main._validate_route_explanation("Follow corridor for an ETA of 810 seconds.", route_ref, llm_used=False)
        self.assertTrue(ok_good_eta)

        bld_destroyed = {"damage_class": "destroyed", "confidence": 0.91}
        _, ok_no_dmg = main._validate_building_explanation("This structure shows no damage at all.", bld_destroyed, llm_used=False)
        self.assertFalse(ok_no_dmg)

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "qa_regress.db")
            seed.seed_sector(db_path=db_file, num_buildings=15)
            client = TestClient(main.app)

            # 2. Zero fake shap_top3 seeded on buildings (Mismatch 1)
            bld_rows = db.get_buildings(db_path=db_file)
            self.assertTrue(all(len(b.get("shap_top3", [])) == 0 for b in bld_rows))

            # 3. Nonexistent custom DB returns 404 (not 500) & creates no empty DB file
            missing_db = str(Path(td) / "nonexistent_qa.db")
            for ep in (
                f"/api/buildings?db_path={missing_db}",
                f"/api/routes?db_path={missing_db}",
                f"/api/audit-log?db_path={missing_db}",
                f"/api/adaptation/drift?db_path={missing_db}",
                f"/api/explanations/route/PLAN-BASELINE-01?db_path={missing_db}",
            ):
                self.assertEqual(client.get(ep).status_code, 404, f"Expected 404 for {ep}")
            self.assertFalse(Path(missing_db).exists())

            # 4. Building correction stores uncertainty = 1.0 - prev_confidence & feedback requires valid label
            bld_before = db.get_building("BLD-0001", db_path=db_file)
            expected_uncert = round(1.0 - float(bld_before["confidence"]), 4)
            r_corr = client.post(
                f"/api/buildings/BLD-0001/correct?db_path={db_file}",
                json={"corrected_damage_class": "destroyed", "responder_id": "QA_RESP"},
            )
            self.assertEqual(r_corr.status_code, 200)
            logs = db.get_feedback_logs(limit=5, db_path=db_file)
            matched_log = next(item for item in logs if item["target_id"] == "BLD-0001")
            self.assertAlmostEqual(float(matched_log["uncertainty"]), expected_uncert, places=4)

            self.assertEqual(
                client.post(
                    f"/api/feedback?db_path={db_file}",
                    json={"target_id": "BLD-0002", "prediction": "minor", "uncertainty": 0.2, "responder_action": "correct"},
                ).status_code,
                422,
            )

            # 5. Explanation gate persists validation_passed=False even when exp_row was None
            db.insert_route(
                {
                    "route_id": "RT-BAD-EXP",
                    "plan_id": "RT-BAD-EXP",
                    "unit_id": "MED-01",
                    "incident_id": "INC-01",
                    "eta_seconds": 810.0,
                    "explanation": "Take shortcut for ETA of 120 seconds.",
                    "uncertainty": 0.12,
                    "status": "pending_approval",
                },
                db_path=db_file,
            )
            r_bad_exp = client.get(f"/api/explanations/route/RT-BAD-EXP?db_path={db_file}")
            self.assertEqual(r_bad_exp.status_code, 200)
            self.assertFalse(r_bad_exp.json()["validation_passed"])
            stored_exp = db.get_explanation_by_target("route", "RT-BAD-EXP", db_path=db_file)
            self.assertIsNotNone(stored_exp)
            self.assertFalse(bool(stored_exp["validation_passed"]))

            # 6. Empty-body POST on /approve & /reject rejected (422) & 409 lock enforced atomically
            self.assertEqual(client.post(f"/api/routes/PLAN-BASELINE-01/approve?db_path={db_file}").status_code, 422)
            self.assertEqual(
                client.post(f"/api/routes/PLAN-BASELINE-01/approve?db_path={db_file}", json={"commander_id": "CMD_1"}).status_code,
                200,
            )
            self.assertEqual(
                client.post(f"/api/routes/PLAN-BASELINE-01/approve?db_path={db_file}", json={"commander_id": "CMD_1"}).status_code,
                409,
            )

        # 7. UI Security & Honesty checks on static/app.js and static/index.html
        html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT_DIR / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("onclick=", js, "Inline onclick template strings are disallowed in app.js (DOM XSS guard)")
        self.assertNotIn("3a7f49b128c70e4bb30dc74ef1a690e0b7a8d56c4293f7da21e64627b089c19f", html)
        self.assertLess(html.index('id="toastHud"'), html.index('<div class="map-shell"'))
        self.assertIn("buildContinuousEdgeLatLngs", js)

    def test_bind_available_socket_falls_back_when_port_occupied(self) -> None:
        import socket

        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(1)
            occupied_port = int(blocker.getsockname()[1])

            bound_sock, actual_port = main._bind_available_socket(
                host="127.0.0.1",
                preferred_port=occupied_port,
                max_attempts=5,
            )
            try:
                self.assertGreater(actual_port, occupied_port)
                self.assertLess(actual_port, occupied_port + 5)
            finally:
                bound_sock.close()
        finally:
            blocker.close()


if __name__ == "__main__":
    unittest.main()

