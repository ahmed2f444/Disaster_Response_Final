"""Unit and QA tests for Portion 2 (`routing/`) integrated modules."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import db
import seed


class TestPhase1RoutingGraphAndMask(unittest.TestCase):
    """Phase 1: Road graph bridge, Section 4.2 road state mapping, hazard updates, and safety mask."""

    def test_road_state_label_supports_section4_2_and_normalized_scales(self) -> None:
        from routing.graph import road_state_label

        # Section 4.2 DB attributes (4 attrs -> 3 display states)
        self.assertEqual(
            road_state_label({"closure_state": False, "hazard_level": 0.04, "capacity": 1.0, "vehicle_restriction": "none"}),
            "open",
        )
        self.assertEqual(
            road_state_label({"closure_state": False, "hazard_level": 0.24, "capacity": 0.5, "vehicle_restriction": "high_clearance_only"}),
            "hazardous",
        )
        self.assertEqual(
            road_state_label({"closure_state": True, "hazard_level": 0.04, "capacity": 1.0, "vehicle_restriction": "none"}),
            "blocked",
        )
        self.assertEqual(
            road_state_label({"closure_state": False, "hazard_level": 0.04, "capacity": 0.0, "vehicle_restriction": "none"}),
            "blocked",
        )
        self.assertEqual(
            road_state_label({"closure_state": False, "hazard_level": 0.62, "capacity": 1.0, "vehicle_restriction": "none"}),
            "blocked",
        )
        # Negm normalized [0, 1] GraphML attributes
        self.assertEqual(road_state_label({"hazard": 0.0, "closed": 0, "capacity": 900.0}), "open")
        self.assertEqual(road_state_label({"hazard": 0.55, "closed": 0, "capacity": 900.0}), "hazardous")
        self.assertEqual(road_state_label({"hazard": 0.95, "closed": 0, "capacity": 900.0}), "blocked")
        self.assertEqual(road_state_label({"hazard": 0.0, "closed": 1, "capacity": 900.0}), "blocked")

    def test_build_graph_from_db_and_state_overlay_matches_seed_sector(self) -> None:
        from routing.graph import (
            HOUSTON_GRAPH_PATH,
            build_graph_from_db,
            edge_id,
            export_houston_graphml,
            load_graph,
            road_state_label,
            state_from_db,
        )

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "graph_phase1.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            G = build_graph_from_db(db_path=db_file)
            self.assertEqual(G.number_of_nodes(), 13)
            self.assertEqual(G.number_of_edges(), 26)
            self.assertEqual(G.graph["provenance"], "houston_east_db")

            max_deg = max(G.degree(n) for n in G.nodes)
            self.assertEqual(max_deg, 5)

            base_state = state_from_db(db_path=db_file)
            self.assertEqual(len(base_state), 26)
            counts = {"open": 0, "hazardous": 0, "blocked": 0}
            for attrs in base_state.values():
                lbl = road_state_label(attrs)
                counts[lbl] += 1
            self.assertEqual(counts, {"open": 19, "hazardous": 7, "blocked": 0})

            # Dry-run surge + manual block overlay must increase blocked count without mutating DB
            surge_state = state_from_db(db_path=db_file, blocked_edges=["E-01"], flood_delta=0.35)
            self.assertEqual(road_state_label(surge_state[edge_id("n0", "n1")]), "blocked")
            # Explicit opened_edges clears surge/closure override in overlay
            reopened_state = state_from_db(db_path=db_file, blocked_edges=[], opened_edges=["E-03"], flood_delta=0.35)
            self.assertEqual(road_state_label(reopened_state[edge_id("n3", "n4")]), "open")
            # Original DB untouched
            self.assertEqual(
                next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-01")["edge_status"],
                "OPEN",
            )

            # GraphML export & reload roundtrip preserves all 13 nodes & 26 edges
            gml_path = Path(td) / "houston_test.graphml"
            export_houston_graphml(db_path=db_file, out_path=gml_path)
            G_reloaded = load_graph(gml_path)
            self.assertEqual(G_reloaded.number_of_nodes(), 13)
            self.assertEqual(G_reloaded.number_of_edges(), 26)
            self.assertTrue(HOUSTON_GRAPH_PATH.is_file())

    def test_update_road_hazards_persists_via_db_helpers(self) -> None:
        from routing.graph import update_road_hazards

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "hazards_phase1.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            summary = update_road_hazards(db_path=db_file, water_level_delta_m=0.35)
            self.assertEqual(sum(summary.values()), 26)
            # Previously HAZARDOUS edges (0.19..0.24m) + 0.35m >= 0.50m -> become BLOCKED
            self.assertGreater(summary["BLOCKED"], 3)

    def test_apply_mask_detours_around_blocked_and_flooded_houston_edges(self) -> None:
        from routing.graph import build_graph_from_db, is_blocked, state_from_db
        from routing.mask import apply_mask

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "mask_phase1.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            G = build_graph_from_db(db_path=db_file)
            state = state_from_db(db_path=db_file, blocked_edges=["E-10"])

            # MED-01 unmasked shortcut tries E-09 (n1-n4) -> E-10 (n4-n7, BLOCKED) -> E-06 (n7-n8)
            proposed = ["n0", "n1", "n4", "n7", "n8"]
            masked, changed, reason = apply_mask(G, proposed, state, target="n8")
            self.assertTrue(changed)
            self.assertIn("E-10", reason)
            self.assertEqual(masked[0], "n0")
            self.assertEqual(masked[-1], "n8")
            for a, b in zip(masked[:-1], masked[1:]):
                eid = G[a][b]["pair_id"]
                self.assertFalse(is_blocked(state[eid]))

            # Safe path stays untouched
            safe_proposed = ["n0", "n1", "n2", "n5", "n8"]
            masked_safe, changed_safe, reason_safe = apply_mask(G, safe_proposed, state, target="n8")
            self.assertFalse(changed_safe)
            self.assertEqual(masked_safe, safe_proposed)
            self.assertEqual(reason_safe, "")

            # Vehicle restriction ("no_heavy") blocks transport units but permits light rescue
            restricted_state = dict(state)
            e01_pair = G["n0"]["n1"]["pair_id"]
            restricted_state[e01_pair] = {**state[e01_pair], "vehicle_restriction": "no_heavy"}
            m_trn, ch_trn, rsn_trn = apply_mask(
                G, ["n0", "n1", "n2"], restricted_state, target="n2", unit_capability="transport"
            )
            self.assertTrue(ch_trn)
            self.assertIn("E-01", rsn_trn)
            self.assertNotIn(("n0", "n1"), list(zip(m_trn[:-1], m_trn[1:])))

    def test_zero_sql_and_no_negm_part_refs_in_phase1_routing_modules(self) -> None:
        legacy_tag = "negm " + "part"
        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        for fname in ("graph.py", "mask.py"):
            fpath = ROOT_DIR / "routing" / fname
            self.assertGreater(fpath.stat().st_size, 0, f"routing/{fname} should be populated")
            src = fpath.read_text(encoding="utf-8")
            self.assertNotIn(legacy_tag, src.lower(), f"Stale legacy folder ref in routing/{fname}")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Raw SQL '{kw}' found in routing/{fname}")


class TestPhase2RoutingEnvAndPPO(unittest.TestCase):
    """Phase 2: Gymnasium RoutingEnv, PPO training artifacts, >90% reachability, and safety mask."""

    def test_houston_routing_env_passes_gymnasium_check_env(self) -> None:
        from gymnasium.utils.env_checker import check_env
        from routing.env import RoutingEnv
        from routing.graph import build_graph_from_db

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "env_phase2.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            G = build_graph_from_db(db_path=db_file)
            env = RoutingEnv(G, seed=1)
            self.assertEqual(env.ix.n, 13)
            self.assertEqual(len(env.ix.edges), 26)
            self.assertEqual(env.ix.max_deg, 5)
            self.assertEqual(env.ix.obs_dim, 105)
            check_env(env, skip_render_check=True)

    def test_ppo_policy_loads_reaches_incidents_and_mask_is_always_safe(self) -> None:
        import itertools
        from stable_baselines3 import PPO
        from routing.env import RoadGraphIndex, build_obs
        from routing.graph import HOUSTON_GRAPH_PATH, is_blocked, load_graph, state_from_db
        from routing.mask import apply_mask
        from routing.train import MODEL_PATH

        self.assertTrue(MODEL_PATH.is_file(), f"Missing trained PPO checkpoint: {MODEL_PATH}")
        G = load_graph(HOUSTON_GRAPH_PATH)
        ix = RoadGraphIndex(G)
        self.assertEqual(ix.obs_dim, 105)

        model = PPO.load(str(MODEL_PATH), device="cpu")
        self.assertEqual(tuple(model.observation_space.shape), (105,))
        self.assertEqual(int(model.action_space.n), 5)

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "ppo_eval.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            state = state_from_db(db_path=db_file)
            vis = {e: int(is_blocked(s) and s.get("reported", 1) == 1) for e, s in state.items()}

            pairs = list(itertools.permutations(ix.nodes, 2))
            reached = 0
            for start, target in pairs:
                cur, path = start, [start]
                for step in range(3 * ix.n):
                    if cur == target:
                        break
                    obs = build_obs(ix, cur, target, state, vis, 1.0 - step / (3 * ix.n))
                    action, _ = model.predict(obs, deterministic=True)
                    nbrs = ix.nbrs[cur]
                    if int(action) >= len(nbrs):
                        break
                    cur = nbrs[int(action)]
                    path.append(cur)
                if path[-1] == target:
                    reached += 1

                masked, _, _ = apply_mask(G, path, state, target=target)
                self.assertEqual(masked[0], start)
                self.assertEqual(masked[-1], target)
                for a, b in zip(masked[:-1], masked[1:]):
                    self.assertFalse(is_blocked(state[G[a][b]["pair_id"]]))

            reach_rate = reached / len(pairs)
            self.assertGreater(reach_rate, 0.90, f"PPO reachability too low: {reach_rate:.3f}")

    def test_zero_sql_and_no_negm_part_refs_in_phase2_routing_modules(self) -> None:
        legacy_tag = "negm " + "part"
        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        for fname in ("env.py", "train.py"):
            fpath = ROOT_DIR / "routing" / fname
            self.assertGreater(fpath.stat().st_size, 0, f"routing/{fname} should be populated")
            src = fpath.read_text(encoding="utf-8")
            self.assertNotIn(legacy_tag, src.lower(), f"Stale legacy folder ref in routing/{fname}")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Raw SQL '{kw}' found in routing/{fname}")


class TestPhase3RoutingFeaturesAndEngine(unittest.TestCase):
    """Phase 3: Routing features, SHAP attribution format, multi-unit dispatch, What-If, and <500ms latency."""

    def test_compute_features_and_shap_attributions_on_proposed_path(self) -> None:
        from routing.features import FEATURE_NAMES, compute_features, features_to_shap_attributions
        from routing.graph import build_graph_from_db, state_from_db

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "feats_phase3.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)
            G = build_graph_from_db(db_path=db_file)
            state = state_from_db(db_path=db_file, blocked_edges=["E-10"])

            # Proposed path crossing hazardous E-09 (n1-n4) and blocked E-10 (n4-n7)
            proposed = ["n0", "n1", "n4", "n7", "n8"]
            feats = compute_features(G, state, proposed, severity="critical", unit_capability="medical", required_capability="medical")
            self.assertEqual(set(FEATURE_NAMES), set(feats.keys()))
            self.assertGreater(feats["travel_time_s"], 0.0)
            self.assertGreater(feats["closure_penalty"], 0.0)
            self.assertEqual(feats["incident_severity"], 1.0)
            self.assertEqual(feats["unit_capability_match"], 1.0)

            shap_items = features_to_shap_attributions(feats, mask_altered=True)
            self.assertEqual(len(shap_items), 7)
            expected_keys = {
                "hazard_distance", "travel_time", "road_capacity",
                "closure_state", "vehicle_restriction", "incident_severity", "unit_capability",
            }
            self.assertEqual({item["feature"] for item in shap_items}, expected_keys)
            for item in shap_items:
                self.assertGreaterEqual(item["value"], 0.0)
                self.assertLessEqual(item["value"], 1.0)

    def test_compute_multi_unit_routes_matches_dispatch_contract_and_gtf_gate(self) -> None:
        import main
        import models
        from routing.generate import compute_multi_unit_routes

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "dispatch_phase3.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            plan = compute_multi_unit_routes(db_path=db_file)
            validated = models.DispatchComputeResponse(**plan)
            self.assertEqual(validated.status, models.RouteStatusEnum.PENDING_APPROVAL)
            self.assertEqual(len(validated.routes), 3)
            self.assertGreater(validated.eta_seconds, 0.0)
            self.assertLess(validated.compute_ms, 500.0)
            self.assertTrue(validated.ppo_proposed_path)
            self.assertTrue(validated.masked_path)
            self.assertEqual(len(validated.shap_values), 7)

            # Sub-routes must have curved coordinates and valid E-XX edge IDs
            for sub in validated.routes:
                self.assertTrue(sub["reachable"])
                self.assertGreater(len(sub["edge_ids"]), 0)
                self.assertTrue(all(eid.startswith("E-") for eid in sub["edge_ids"]))
                self.assertGreater(len(sub["masked_path"]), len(sub["node_path"]))

            # Explanation must pass Portion 5 Validation & ETA Gate
            txt, gate_ok = main._validate_route_explanation(
                candidate_text=validated.explanation,
                route=validated.model_dump(mode="json"),
                llm_used=False,
            )
            self.assertTrue(gate_ok, f"Route narrative rejected by P5 Validation Gate: {txt}")

    def test_run_counterfactual_is_dry_run_safe_and_under_500ms(self) -> None:
        from routing.generate import compute_multi_unit_routes, run_counterfactual

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "whatif_phase3.db")
            seed.seed_sector(db_path=db_file, num_buildings=20)

            # Warm engine cache with first call, then benchmark repeated reroutes
            _ = compute_multi_unit_routes(db_path=db_file)
            res = run_counterfactual(blocked_edges=["E-01"], flood_delta=0.25, db_path=db_file)

            self.assertIn("E-01", res["newly_blocked_edges"])
            self.assertGreater(res["whatif_eta_sec"], 0.0)
            self.assertLess(res["compute_ms"], 500.0, f"Reroute exceeded 500ms: {res['compute_ms']}ms")
            self.assertEqual(len(res["routes"]), 3)
            for sub in res["routes"]:
                self.assertNotIn("E-01", sub["edge_ids"])

            # Dry-run must NOT mutate persisted baseline E-01 status in DB
            e01 = next(e for e in db.get_road_edges(db_path=db_file) if e["edge_id"] == "E-01")
            self.assertEqual(e01["edge_status"], "OPEN")

    def test_zero_sql_and_no_legacy_refs_in_phase3_routing_modules(self) -> None:
        legacy_tag = "negm " + "part"
        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        for fname in ("__init__.py", "features.py", "generate.py"):
            fpath = ROOT_DIR / "routing" / fname
            self.assertGreater(fpath.stat().st_size, 0, f"routing/{fname} should be populated")
            src = fpath.read_text(encoding="utf-8")
            self.assertNotIn(legacy_tag, src.lower(), f"Stale legacy folder ref in routing/{fname}")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Raw SQL '{kw}' found in routing/{fname}")

    def test_houston_graphml_and_ppo_checkpoint_dimensions(self) -> None:
        from stable_baselines3 import PPO
        from routing.env import RoutingEnv
        from routing.graph import HOUSTON_GRAPH_PATH, load_graph
        from routing.train import MODEL_PATH

        self.assertTrue(HOUSTON_GRAPH_PATH.is_file(), "Missing Houston-East GraphML")
        self.assertTrue(MODEL_PATH.is_file(), "Missing trained routing_ppo.zip")

        G = load_graph(HOUSTON_GRAPH_PATH)
        self.assertEqual(G.number_of_nodes(), 13)
        self.assertEqual(G.number_of_edges(), 26)

        env = RoutingEnv(G)
        model = PPO.load(str(MODEL_PATH), device="cpu")
        self.assertEqual(tuple(model.observation_space.shape), (105,))
        self.assertEqual(tuple(model.observation_space.shape), tuple(env.observation_space.shape))
        self.assertEqual(int(model.action_space.n), 5)

    def test_osm_xml_build_sync_preserve_coords_and_route_fallback(self) -> None:
        """OSM graphs preserve real coords across DB reload and route without PPO shape crashes."""
        import tempfile
        import db
        import seed
        from routing.generate import compute_multi_unit_routes, run_counterfactual
        from routing.graph import build_from_osm, build_graph_from_db, graph_bounds, sync_osm_graph_to_db

        # Non-Houston drivable street network fixture (New Bern, NC) with intermediate curve node
        osm_xml = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="unit-test">
  <node id="1" lat="35.1060" lon="-77.0500" version="1"/>
  <node id="2" lat="35.1072" lon="-77.0480" version="1"/>
  <node id="3" lat="35.1060" lon="-77.0460" version="1"/>
  <node id="4" lat="35.1090" lon="-77.0500" version="1"/>
  <node id="5" lat="35.1090" lon="-77.0460" version="1"/>
  <node id="6" lat="35.1110" lon="-77.0480" version="1"/>
  <way id="101" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/>
    <tag k="highway" v="residential"/><tag k="name" v="Craven St"/>
  </way>
  <way id="102" version="1">
    <nd ref="1"/><nd ref="4"/>
    <tag k="highway" v="secondary"/><tag k="name" v="Broad St"/>
  </way>
  <way id="103" version="1">
    <nd ref="4"/><nd ref="5"/>
    <tag k="highway" v="secondary"/><tag k="name" v="Middle St"/>
  </way>
  <way id="104" version="1">
    <nd ref="5"/><nd ref="3"/>
    <tag k="highway" v="tertiary"/><tag k="name" v="Pollock St"/>
  </way>
  <way id="105" version="1">
    <nd ref="3"/><nd ref="6"/>
    <tag k="highway" v="residential"/><tag k="name" v="Union St"/>
  </way>
  <way id="106" version="1">
    <nd ref="6"/><nd ref="4"/>
    <tag k="highway" v="residential"/><tag k="name" v="East St"/>
  </way>
  <way id="107" version="1">
    <nd ref="5"/><nd ref="1"/>
    <tag k="highway" v="residential"/><tag k="name" v="Diag St"/>
  </way>
</osm>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            osm_path = tmp_path / "fixture.osm"
            osm_path.write_text(osm_xml, encoding="utf-8")
            db_path = str(tmp_path / "osm_test.db")
            db.init_db(db_path)

            G = build_from_osm(osm_path, n_nodes=5, mark_hazards=False)
            self.assertGreaterEqual(G.number_of_nodes(), 4)
            self.assertGreaterEqual(G.number_of_edges(), 5)
            # Intermediate shape node '2' on way 101 must yield a curved >=3-point geom_json
            curved = [
                d.get("geom_json")
                for _, _, d in G.edges(data=True)
                if d.get("geom_json") and len(d["geom_json"].get("coordinates", [])) >= 3
            ]
            self.assertTrue(curved, "Expected at least one curved OSM edge geometry")
            for _, _, d in G.edges(data=True):
                self.assertEqual(float(d.get("hazard", 0.0)), 0.0)
                coords = (d.get("geom_json") or {}).get("coordinates", [])
                self.assertGreaterEqual(len(coords), 2)
                for lon, lat in coords:
                    self.assertTrue(-77.06 <= lon <= -77.03, f"Unexpected lon {lon}")
                    self.assertTrue(35.09 <= lat <= 35.12, f"Unexpected lat {lat}")

            synced_bounds = sync_osm_graph_to_db(osm_path, db_path=db_path, n_nodes=5, mark_hazards=False)
            self.assertEqual(len(synced_bounds), 4)
            G_db = build_graph_from_db(db_path=db_path)
            self.assertEqual(G_db.number_of_nodes(), G.number_of_nodes())
            min_lon, min_lat, max_lon, max_lat = graph_bounds(G_db)
            self.assertLess(min_lon, -77.04)
            self.assertGreater(min_lat, 35.09)
            self.assertLess(max_lat, 35.12)

            # Node coordinates must NOT be overwritten by seed.NODES (Houston-East)
            houston_lons = {round(pt[0], 3) for pt in seed.NODES.values()}
            for _, ndata in G_db.nodes(data=True):
                self.assertNotIn(round(float(ndata["lon"]), 3), houston_lons)

            # Routing on a 5-node OSM graph (obs_dim != 105) must fall back cleanly without crashing
            plan = compute_multi_unit_routes(db_path=db_path)
            unit_routes = plan["routes"]
            self.assertTrue(unit_routes, "Expected multi-unit routes on non-Houston OSM graph")
            for r in unit_routes:
                coords = r["geometry"]["coordinates"]
                self.assertGreaterEqual(len(coords), 2)
                for lon, lat in coords:
                    self.assertTrue(-77.06 <= lon <= -77.03)
                    self.assertTrue(35.09 <= lat <= 35.12)

            edge_id = db.get_road_edges(db_path=db_path)[0]["edge_id"]
            cf = run_counterfactual(blocked_edges=[edge_id], db_path=db_path)
            self.assertIn("routes", cf)
            self.assertGreaterEqual(len(cf["routes"][0]["geometry"]["coordinates"]), 2)

    def test_osm_upload_endpoint_syncs_osm_graph_into_state(self) -> None:
        """POST /api/assessment/upload with osm_xml builds OSM roads & routes in /api/state."""
        import base64
        import tempfile
        from unittest import mock
        from fastapi.testclient import TestClient
        import damage.assess as assess_mod
        import db
        import main

        osm_xml = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="unit-test">
  <node id="1" lat="35.1060" lon="-77.0500" version="1"/>
  <node id="2" lat="35.1072" lon="-77.0480" version="1"/>
  <node id="3" lat="35.1060" lon="-77.0460" version="1"/>
  <node id="4" lat="35.1090" lon="-77.0500" version="1"/>
  <node id="5" lat="35.1090" lon="-77.0460" version="1"/>
  <node id="6" lat="35.1110" lon="-77.0480" version="1"/>
  <way id="101" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/>
    <tag k="highway" v="residential"/><tag k="name" v="Craven St"/>
  </way>
  <way id="102" version="1">
    <nd ref="1"/><nd ref="4"/>
    <tag k="highway" v="secondary"/><tag k="name" v="Broad St"/>
  </way>
  <way id="103" version="1">
    <nd ref="4"/><nd ref="5"/>
    <tag k="highway" v="secondary"/><tag k="name" v="Middle St"/>
  </way>
  <way id="104" version="1">
    <nd ref="5"/><nd ref="3"/>
    <tag k="highway" v="tertiary"/><tag k="name" v="Pollock St"/>
  </way>
  <way id="105" version="1">
    <nd ref="3"/><nd ref="6"/>
    <tag k="highway" v="residential"/><tag k="name" v="Union St"/>
  </way>
  <way id="106" version="1">
    <nd ref="6"/><nd ref="4"/>
    <tag k="highway" v="residential"/><tag k="name" v="East St"/>
  </way>
  <way id="107" version="1">
    <nd ref="5"/><nd ref="1"/>
    <tag k="highway" v="residential"/><tag k="name" v="Diag St"/>
  </way>
</osm>"""
        pre_file = Path(__file__).resolve().parent.parent / "static" / "samples" / "test_fixture_T0_pre_disaster.png"
        post_file = Path(__file__).resolve().parent.parent / "static" / "samples" / "test_fixture_T1_post_disaster.png"
        pre_uri = "data:image/png;base64," + base64.b64encode(pre_file.read_bytes()).decode("ascii")
        post_uri = "data:image/png;base64," + base64.b64encode(post_file.read_bytes()).decode("ascii")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_file = str(tmp_path / "upload_osm.db")
            export_dir = tmp_path / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            db.init_db(db_file)
            client = TestClient(main.app)

            with mock.patch.object(assess_mod, "DEFAULT_EXPORT_DIR", export_dir):
                resp = client.post(
                    f"/api/assessment/upload?db_path={db_file}",
                    json={
                        "pre_disaster_image": pre_uri,
                        "post_disaster_image": post_uri,
                        "sector_id": "new-bern-nc",
                        "osm_xml": osm_xml,
                        "use_osm_graph": True,
                    },
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            state_resp = client.get(f"/api/state?db_path={db_file}")
            self.assertEqual(state_resp.status_code, 200)
            state = state_resp.json()
            roads = state["roads_geojson"]["features"]
            self.assertTrue(roads)
            for feat in roads:
                for lon, lat in feat["geometry"]["coordinates"]:
                    self.assertTrue(-77.06 <= lon <= -77.03, f"Road lon not OSM: {lon}")
                    self.assertTrue(35.09 <= lat <= 35.12, f"Road lat not OSM: {lat}")
            active_plan = state.get("active_plan")
            self.assertIsNotNone(active_plan, "Expected active_plan after OSM assessment upload")
            unit_routes = active_plan.get("routes") or []
            self.assertTrue(unit_routes, "Expected multi-unit routes in active_plan")
            for u_route in unit_routes:
                coords = u_route["geometry"]["coordinates"]
                self.assertGreaterEqual(len(coords), 1)
                for lon, lat in coords:
                    self.assertTrue(-77.06 <= lon <= -77.03, f"Route lon not OSM: {lon}")
                    self.assertTrue(35.09 <= lat <= 35.12, f"Route lat not OSM: {lat}")
            self.assertTrue(
                any(len(u["geometry"]["coordinates"]) >= 2 for u in unit_routes),
                "Expected at least one multi-coordinate OSM route",
            )



if __name__ == "__main__":
    unittest.main()

