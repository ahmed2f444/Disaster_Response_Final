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


if __name__ == "__main__":
    unittest.main()

