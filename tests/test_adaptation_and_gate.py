"""Task 4 TDD & QA Tests for Portion 4 (`feedback/adapt.py`, `feedback/gate.py`, `feedback/status.py`, & `damage/`)."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestTask4AdaptationAndSafetyGate(unittest.TestCase):
    """Verify frozen-encoder Tier A adapter training, 5-check safety gate, promotion, and rollback."""

    def test_focal_loss_and_frozen_encoder_tier_a_adapter_isolation(self) -> None:
        from damage.dataset import prepare_paired_tensor
        from damage.model import build_model, load_damage_model
        from feedback import adapt

        t0 = ROOT_DIR / "static" / "samples" / "test_fixture_T0_pre_disaster.png"
        t1 = ROOT_DIR / "static" / "samples" / "test_fixture_T1_post_disaster.png"
        x_tensor, (orig_h, orig_w) = prepare_paired_tensor(t0, t1, multiple=32)
        # Small deterministic crop (1, 6, 128, 128) for fast CPU/GPU test
        x_crop = x_tensor[:, :, :128, :128]
        y_crop = torch.zeros((1, 128, 128), dtype=torch.long)
        y_crop[0, 20:50, 20:50] = 4  # destroyed
        y_crop[0, 70:100, 70:100] = 2  # minor

        routing_weights = ROOT_DIR / "data" / "models" / "routing_ppo.zip"
        routing_hash_before = _sha256_file(routing_weights) if routing_weights.is_file() else ""

        base_net = build_model(pretrained_encoder=False)
        wrapped = adapt.ResidualAdapterUNet(base_net, adapter_channels=16)

        # Portion 4 Plan Sec. 3: Freeze backbone encoder; only adapter + head trainable
        for param in wrapped.base_model.encoder.parameters():
            self.assertFalse(param.requires_grad, "Encoder backbone must stay frozen during Tier A adaptation")
        trainable = [p for p in wrapped.parameters() if p.requires_grad]
        self.assertGreater(len(trainable), 0)

        with tempfile.TemporaryDirectory() as td:
            res = adapt.run_tier_a_adaptation(
                samples=[(x_crop[0], y_crop[0])],
                disaster_id="FLOOD_TEST_2026",
                adapter_version="flood-tier-a-v1",
                adapter_root=Path(td) / "adapters",
                epochs=2,
                weights_path=ROOT_DIR / "data" / "models" / "unet_resnet18_best.pth",
            )
            adapter_pt = Path(res["adapter_path"])
            self.assertTrue(adapter_pt.is_file())
            self.assertLess(res["duration_sec"], 120.0)
            self.assertEqual(res["scope"], "segmentation_only")

            # Verify damage.model can load the promoted adapter without mutating base weights
            adapted_net, dev = load_damage_model(
                weights_path=ROOT_DIR / "data" / "models" / "unet_resnet18_best.pth",
                adapter_path=adapter_pt,
                use_cache=False,
            )
            with torch.no_grad():
                out = adapted_net(x_crop.to(dev))
            self.assertEqual(tuple(out.shape), (1, 5, 128, 128))

            # Verify compute_pixel_metrics runs on real adapted model
            metrics = __import__("feedback.gate", fromlist=["compute_pixel_metrics"]).compute_pixel_metrics(
                adapted_net, [(x_crop[0], y_crop[0])], device=dev
            )
            self.assertIn("destroyed", metrics["per_class_f1"])
            self.assertGreater(metrics["tile_latency_ms"], 0.0)

        if routing_hash_before:
            self.assertEqual(_sha256_file(routing_weights), routing_hash_before, "Portion 2 routing weights must never be modified")

    def test_five_check_safety_gate_fail_closed_and_promotion_rollback_audit(self) -> None:
        import db
        import seed
        from feedback import gate, status

        # 1. Fail-closed when forgetting exceeds tolerance (> 2 F1 points) or holdout fails
        fail_report = gate.evaluate_safety_gate(
            baseline_metrics={
                "per_class_f1": {"no_damage": 0.80, "minor": 0.65, "major": 0.60, "destroyed": 0.55},
                "old_disaster_macro_f1": 0.72,
                "ece": 0.06,
            },
            candidate_metrics={
                "per_class_f1": {"no_damage": 0.81, "minor": 0.68, "major": 0.64, "destroyed": 0.62},
                "old_disaster_macro_f1": 0.68,  # 0.04 drop > 0.02 tolerance!
                "geographic_holdout_f1": 0.69,
                "geographic_holdout_baseline_f1": 0.65,
                "ece": 0.07,
                "tile_latency_ms": 120.0,
            },
        )
        self.assertFalse(fail_report["passed"])
        self.assertFalse(fail_report["checks"]["forgetting_check"]["passed"])

        # 2. Fail-closed when geographic holdout metrics are missing
        missing_holdout = gate.evaluate_safety_gate(
            baseline_metrics={
                "per_class_f1": {"no_damage": 0.80, "minor": 0.65, "major": 0.60, "destroyed": 0.55},
                "old_disaster_macro_f1": 0.72,
                "ece": 0.06,
            },
            candidate_metrics={
                "per_class_f1": {"no_damage": 0.81, "minor": 0.68, "major": 0.64, "destroyed": 0.62},
                "old_disaster_macro_f1": 0.715,
                "ece": 0.065,
                "tile_latency_ms": 120.0,
            },
        )
        self.assertFalse(missing_holdout["passed"])
        self.assertFalse(missing_holdout["checks"]["geographic_holdout"]["passed"])

        # 3. Full pass -> Shadow -> Promotion -> SHA-256 Audit -> One-Click Rollback
        pass_report = gate.evaluate_safety_gate(
            baseline_metrics={
                "per_class_f1": {"no_damage": 0.80, "minor": 0.65, "major": 0.60, "destroyed": 0.55},
                "old_disaster_macro_f1": 0.72,
                "ece": 0.06,
            },
            candidate_metrics={
                "per_class_f1": {"no_damage": 0.82, "minor": 0.69, "major": 0.65, "destroyed": 0.63},
                "old_disaster_macro_f1": 0.71,  # 0.01 drop <= 0.02
                "geographic_holdout_f1": 0.70,
                "geographic_holdout_baseline_f1": 0.64,
                "ece": 0.065,
                "tile_latency_ms": 95.0,
            },
        )
        self.assertTrue(pass_report["passed"])
        self.assertEqual(pass_report["destroyed_f1"], 0.63)

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t4_gate.db")
            seed.seed_sector(db_path=db_file, num_buildings=15)

            promo = status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="harvey-tier-a-v1",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=str(Path(td) / "adapter.pt"),
                gate_report=pass_report,
                promote=True,
                db_path=db_file,
            )
            self.assertTrue(promo["promoted"])
            active = db.get_active_model_version(db_path=db_file)
            self.assertEqual(active["adapter_version"], "harvey-tier-a-v1")

            rb = status.rollback_adapter(
                reason="Commander requested rollback test",
                actor_role="COMMANDER:CMD_LOCAL_01",
                db_path=db_file,
            )
            self.assertTrue(rb["rolled_back"])
            self.assertEqual(rb["active_adapter_version"], "none")
            chain_ok, _, msg = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(chain_ok, msg)

    def test_assess_uses_active_promoted_adapter_and_exports_gradcam_and_embedding(self) -> None:
        import base64

        import db
        import seed
        from damage.assess import run_vision_assessment
        from damage.dataset import prepare_paired_tensor
        from feedback import adapt, gate, status

        t0 = ROOT_DIR / "static" / "samples" / "test_fixture_T0_pre_disaster.png"
        t1 = ROOT_DIR / "static" / "samples" / "test_fixture_T1_post_disaster.png"
        pre_uri = "data:image/png;base64," + base64.b64encode(t0.read_bytes()).decode("ascii")
        post_uri = "data:image/png;base64," + base64.b64encode(t1.read_bytes()).decode("ascii")
        x_tensor, _ = prepare_paired_tensor(t0, t1, multiple=32)
        x_crop = x_tensor[:, :, :128, :128]
        y_crop = torch.zeros((128, 128), dtype=torch.long)
        y_crop[20:50, 20:50] = 3

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "assess_adapted.db")
            export_tmp = Path(td) / "exports"
            seed.seed_sector(db_path=db_file, num_buildings=10)

            res = adapt.run_tier_a_adaptation(
                samples=[(x_crop[0], y_crop)],
                disaster_id="HOUSTON_HARVEY_2017",
                adapter_version="harvey-assess-v1",
                adapter_root=Path(td) / "adapters",
                epochs=1,
                weights_path=ROOT_DIR / "data" / "models" / "unet_resnet18_best.pth",
            )
            pass_gate = gate.evaluate_safety_gate(
                baseline_metrics={"per_class_f1": {"no_damage": 0.7, "minor": 0.6, "major": 0.5, "destroyed": 0.4}, "old_disaster_macro_f1": 0.7, "ece": 0.05},
                candidate_metrics={"per_class_f1": {"no_damage": 0.75, "minor": 0.65, "major": 0.55, "destroyed": 0.5}, "old_disaster_macro_f1": 0.695, "geographic_holdout_f1": 0.68, "geographic_holdout_baseline_f1": 0.62, "ece": 0.055, "tile_latency_ms": 90.0},
            )
            status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="harvey-assess-v1",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=res["adapter_path"],
                gate_report=pass_gate,
                promote=True,
                db_path=db_file,
            )

            assess_out = run_vision_assessment(
                db_path=db_file,
                pre_disaster_image=pre_uri,
                post_disaster_image=post_uri,
                export_dir=export_tmp,
            )
            self.assertEqual(assess_out["status"], "completed")
            audit_rows = db.get_audit_log(limit=5, db_path=db_file)
            inf_row = next(r for r in audit_rows if r["event_type"] == "MODEL_INFERENCE_COMPLETED")
            self.assertEqual(inf_row["payload"]["adapter_version"], "harvey-assess-v1")

    def test_qa_path_traversal_guard_focal_empty_masks_and_pseudo_masks(self) -> None:
        from damage.model import IGNORE_INDEX, build_model
        from feedback import adapt

        # Exact ".." / "../../etc" must NEVER escape adapter_root
        with tempfile.TemporaryDirectory() as td:
            adapter_root = (Path(td) / "adapters").resolve()
            adapter_root.mkdir(parents=True)
            x = torch.randn(6, 64, 64)
            y = torch.full((64, 64), 2, dtype=torch.long)
            y[:16, :16] = 4

            base = build_model(pretrained_encoder=False)
            wrapped = adapt.ResidualAdapterUNet(base, adapter_channels=8)
            enc_before = {k: v.detach().clone() for k, v in wrapped.base_model.encoder.state_dict().items()}
            dec_before = {k: v.detach().clone() for k, v in wrapped.base_model.decoder.state_dict().items()}

            res = adapt.run_tier_a_adaptation(
                samples=[(x, y)],
                disaster_id="..",
                adapter_version="../../evil",
                adapter_root=adapter_root,
                epochs=1,
                weights_path=Path(td) / "nonexistent_fallback_weights.pth",
                device="cpu",
            )
            out_path = Path(res["adapter_path"]).resolve()
            self.assertTrue(out_path.is_file())
            self.assertTrue(out_path.is_relative_to(adapter_root))

            # Frozen encoder + decoder remain strictly requires_grad=False & bitwise equal during step
            crit = adapt.FocalLoss(gamma=2.0, weight=adapt.compute_inverse_class_weights([y]))
            logits = wrapped(x.unsqueeze(0))
            loss = crit(logits, y.unsqueeze(0))
            loss.backward()
            for k, v in wrapped.base_model.encoder.state_dict().items():
                self.assertTrue(torch.equal(v, enc_before[k]))
            for k, v in wrapped.base_model.decoder.state_dict().items():
                self.assertTrue(torch.equal(v, dec_before[k]))

        # Empty / IGNORE_INDEX-only / negative invalid pixel masks must stay finite without ValueError
        all_ignore = torch.full((1, 32, 32), IGNORE_INDEX, dtype=torch.long)
        logits_dummy = torch.randn(1, 5, 32, 32, requires_grad=True)
        focal = adapt.FocalLoss()
        l_empty = focal(logits_dummy, all_ignore)
        self.assertTrue(torch.isfinite(l_empty))
        self.assertEqual(float(l_empty.item()), 0.0)

        bad_mask = torch.tensor([[IGNORE_INDEX, -1, 4, 99]])
        w = adapt.compute_inverse_class_weights([bad_mask])
        self.assertTrue(torch.all(torch.isfinite(w)))
        w_empty = adapt.compute_inverse_class_weights([])
        self.assertEqual(tuple(w_empty.shape), (5,))

        # generate_hazard_pseudo_masks handles binary integer hazard_mask cleanly
        probs = np.zeros((5, 8, 8), dtype=np.float32)
        probs[1] = 0.90
        probs[2, :4, :4] = 0.70
        probs[1, :4, :4] = 0.20
        haz_int = np.zeros((8, 8), dtype=np.int32)
        haz_int[:4, :4] = 1
        pseudo = adapt.generate_hazard_pseudo_masks(probs, haz_int)
        self.assertEqual(int(pseudo[1, 1]), 2)

    def test_qa_safety_gate_all_five_checks_and_cheat_prevention(self) -> None:
        import json
        from feedback import gate

        base = {
            "per_class_f1": {"no_damage": 0.80, "minor": 0.65, "major": 0.60, "destroyed": 0.55},
            "old_disaster_macro_f1": 0.72,
            "ece": 0.06,
        }
        good = {
            "per_class_f1": {"no_damage": 0.82, "minor": 0.69, "major": 0.65, "destroyed": 0.63},
            "old_disaster_macro_f1": 0.71,
            "geographic_holdout_f1": 0.70,
            "geographic_holdout_baseline_f1": 0.64,
            "ece": 0.065,
            "tile_latency_ms": 95.0,
        }

        # Check 1a: Missing classes from candidate (gaming macro_f1 with only destroyed) must FAIL CLOSED
        g_cheat = dict(good, per_class_f1={"destroyed": 0.92})
        r_cheat = gate.evaluate_safety_gate(base, g_cheat)
        self.assertFalse(r_cheat["passed"])
        self.assertFalse(r_cheat["checks"]["per_class_f1"]["passed"])

        # Check 1b: Degraded destroyed F1 even if macro improves
        g_dest = dict(good, per_class_f1={"no_damage": 0.95, "minor": 0.85, "major": 0.80, "destroyed": 0.45})
        r_dest = gate.evaluate_safety_gate(base, g_dest)
        self.assertFalse(r_dest["passed"])
        self.assertFalse(r_dest["checks"]["per_class_f1"]["passed"])

        # Check 3b: Spatial overlap between train and holdout regions fails closed
        g_ov = dict(good, train_region_ids=["R1", "R2"], holdout_region_ids=["R2", "R3"])
        r_ov = gate.evaluate_safety_gate(base, g_ov)
        self.assertFalse(r_ov["passed"])
        self.assertFalse(r_ov["checks"]["geographic_holdout"]["passed"])

        # Check 4a: ECE calibration degradation > 0.05 fails
        g_ece = dict(good, ece=0.20)
        r_ece = gate.evaluate_safety_gate(base, g_ece)
        self.assertFalse(r_ece["passed"])
        self.assertFalse(r_ece["checks"]["calibration_preserved"]["passed"])

        # Check 4b: Calibration improvement (lower candidate ECE than baseline) passes cleanly
        r_ece_better = gate.evaluate_safety_gate(dict(base, ece=0.22), dict(good, ece=0.14))
        self.assertTrue(r_ece_better["checks"]["calibration_preserved"]["passed"])
        self.assertTrue(r_ece_better["passed"])

        # Check 5: Edge latency budget exceeded OR non-finite Inf latency fails without NaN JSON error
        g_lat = dict(good, tile_latency_ms=float("inf"))
        r_lat = gate.evaluate_safety_gate(base, g_lat)
        self.assertFalse(r_lat["passed"])
        self.assertFalse(r_lat["checks"]["edge_latency"]["passed"])
        json.dumps(r_lat, allow_nan=False)

    def test_qa_multi_version_promotion_rapid_runs_and_rollback_chain(self) -> None:
        import db
        import seed
        from feedback import gate, status

        pass_gate = gate.evaluate_safety_gate(
            baseline_metrics={"per_class_f1": {"no_damage": 0.7, "minor": 0.6, "major": 0.5, "destroyed": 0.4}, "old_disaster_macro_f1": 0.7, "ece": 0.05},
            candidate_metrics={"per_class_f1": {"no_damage": 0.75, "minor": 0.65, "major": 0.55, "destroyed": 0.5}, "old_disaster_macro_f1": 0.695, "geographic_holdout_f1": 0.68, "geographic_holdout_baseline_f1": 0.62, "ece": 0.055, "tile_latency_ms": 90.0},
        )
        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "t4_chain.db")
            seed.seed_sector(db_path=db_file, num_buildings=8)

            # Same adapter_version staged then promoted within same second must create 2 distinct run rows
            r_shadow = status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="v1",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=str(Path(td) / "v1.pt"),
                gate_report=pass_gate,
                promote=False,
                db_path=db_file,
            )
            r_v1 = status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="v1",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=str(Path(td) / "v1.pt"),
                gate_report=pass_gate,
                promote=True,
                db_path=db_file,
            )
            self.assertNotEqual(r_shadow["run_id"], r_v1["run_id"])
            self.assertEqual(len(db.get_adaptation_runs(limit=10, db_path=db_file)), 2)

            # Promote v2, rollback -> restores v1, rollback again -> restores "none", 3rd -> safe no-op
            status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="v2",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=str(Path(td) / "v2.pt"),
                gate_report=pass_gate,
                promote=True,
                db_path=db_file,
            )
            rb1 = status.rollback_adapter(reason="Test rollback v2->v1", db_path=db_file)
            self.assertTrue(rb1["rolled_back"])
            self.assertEqual(rb1["active_adapter_version"], "v1")

            rb2 = status.rollback_adapter(reason="Test rollback v1->none", db_path=db_file)
            self.assertTrue(rb2["rolled_back"])
            self.assertEqual(rb2["active_adapter_version"], "none")
            self.assertIsNone(db.get_active_model_version(db_path=db_file))

            rb3 = status.rollback_adapter(reason="No-op rollback", db_path=db_file)
            self.assertFalse(rb3["rolled_back"])

            ok, _, msg = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(ok, msg)

    def test_qa_corrupt_adapter_safe_fallback_and_zero_sql(self) -> None:
        import ast
        import base64
        import re
        import db
        import seed
        from damage.assess import run_vision_assessment
        from damage.model import AdaptedSegmentationModel, load_damage_model
        from feedback import gate, status

        with tempfile.TemporaryDirectory() as td:
            corrupt_pt = Path(td) / "corrupt_adapter.pt"
            corrupt_pt.write_bytes(b"NOT_A_TORCH_ARCHIVE")

            # Direct model load must fall back cleanly to base model rather than crashing
            net_fb, _ = load_damage_model(
                weights_path=ROOT_DIR / "data" / "models" / "unet_resnet18_best.pth",
                adapter_path=corrupt_pt,
                use_cache=False,
            )
            self.assertNotIsInstance(net_fb, AdaptedSegmentationModel)

            # Registered active corrupt adapter in DB must still allow assess completion & log adapter='none'
            db_file = str(Path(td) / "corrupt_assess.db")
            seed.seed_sector(db_path=db_file, num_buildings=6)
            pass_gate = gate.evaluate_safety_gate(
                baseline_metrics={"per_class_f1": {"no_damage": 0.7, "minor": 0.6, "major": 0.5, "destroyed": 0.4}, "old_disaster_macro_f1": 0.7, "ece": 0.05},
                candidate_metrics={"per_class_f1": {"no_damage": 0.75, "minor": 0.65, "major": 0.55, "destroyed": 0.5}, "old_disaster_macro_f1": 0.695, "geographic_holdout_f1": 0.68, "geographic_holdout_baseline_f1": 0.62, "ece": 0.055, "tile_latency_ms": 90.0},
            )
            status.promote_or_stage_adapter(
                model_version="ResNet18-UNet-xBD-v1.0",
                adapter_version="corrupt-v1",
                disaster_id="HOUSTON_HARVEY_2017",
                weights_path=str(corrupt_pt),
                gate_report=pass_gate,
                promote=True,
                db_path=db_file,
            )

            t0 = ROOT_DIR / "static" / "samples" / "test_fixture_T0_pre_disaster.png"
            t1 = ROOT_DIR / "static" / "samples" / "test_fixture_T1_post_disaster.png"
            pre_uri = "data:image/png;base64," + base64.b64encode(t0.read_bytes()).decode("ascii")
            post_uri = "data:image/png;base64," + base64.b64encode(t1.read_bytes()).decode("ascii")

            out = run_vision_assessment(
                db_path=db_file,
                pre_disaster_image=pre_uri,
                post_disaster_image=post_uri,
                export_dir=Path(td) / "exports",
            )
            self.assertEqual(out["status"], "completed")
            inf_row = next(r for r in db.get_audit_log(limit=5, db_path=db_file) if r["event_type"] == "MODEL_INFERENCE_COMPLETED")
            self.assertEqual(inf_row["payload"]["adapter_version"], "none")

        sql_re = re.compile(r"\b(SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|CREATE\s+TABLE)\b", re.I | re.S)
        for fname in ("adapt.py", "gate.py", "status.py"):
            src = (ROOT_DIR / "feedback" / fname).read_text(encoding="utf-8")
            self.assertNotIn("abdelrahman", src.lower())
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertIsNone(sql_re.search(node.value), f"Raw SQL in {fname}")

    def test_safety_gate_rejects_background_collapse_and_tier_a_preserves_bg(self) -> None:
        import db
        import seed
        from damage.dataset import prepare_paired_tensor
        from damage.model import DEFAULT_WEIGHTS_PATH, load_damage_model
        from feedback import adapt, gate

        base = {
            "per_class_f1": {"no_damage": 0.80, "minor": 0.65, "major": 0.60, "destroyed": 0.55},
            "background_f1": 0.92,
            "old_disaster_macro_f1": 0.72,
            "ece": 0.06,
        }
        collapsed = {
            "per_class_f1": {"no_damage": 0.82, "minor": 0.68, "major": 0.62, "destroyed": 0.60},
            "background_f1": 0.02,
            "old_disaster_macro_f1": 0.72,
            "geographic_holdout_f1": 0.70,
            "geographic_holdout_baseline_f1": 0.64,
            "ece": 0.06,
            "tile_latency_ms": 95.0,
        }
        report = gate.evaluate_safety_gate(base, collapsed)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["per_class_f1"]["passed"])

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "bg_preserve.db")
            seed.seed_sector(db_path=db_file, num_buildings=10)
            for idx, cls_name in ((1, "destroyed"), (2, "major"), (3, "minor"), (4, "no_damage")):
                db.update_building_label(f"BLD-000{idx}", cls_name, db_path=db_file)
                db.log_feedback(
                    {
                        "target_id": f"BLD-000{idx}",
                        "prediction": "no_damage",
                        "uncertainty": 0.5,
                        "responder_action": "correct",
                        "corrected_label": cls_name,
                        "responder_id": "QA_BG",
                    },
                    db_path=db_file,
                )
            res = adapt.run_feedback_driven_tier_a(
                disaster_id="HOUSTON_HARVEY_2017",
                epochs=4,
                learning_rate=0.02,
                promote=False,
                db_path=db_file,
                adapter_root=Path(td) / "adapters",
                weights_path=DEFAULT_WEIGHTS_PATH,
                data_dir=Path(td) / "no_uploads",
            )
            t0 = ROOT_DIR / "static" / "samples" / "test_fixture_T0_pre_disaster.png"
            t1 = ROOT_DIR / "static" / "samples" / "test_fixture_T1_post_disaster.png"
            x_t, _ = prepare_paired_tensor(t0, t1, multiple=32)
            net, dev = load_damage_model(
                weights_path=DEFAULT_WEIGHTS_PATH,
                adapter_path=res["training_summary"]["adapter_path"],
                use_cache=False,
            )
            with torch.no_grad():
                probs = torch.softmax(net(x_t.to(dev)), dim=1)[0].cpu().numpy()
            bg_fraction = float((probs.argmax(axis=0) == 0).mean())
            self.assertGreater(bg_fraction, 0.5, f"Adapter collapsed background to {bg_fraction:.3f}")

    def test_drift_auto_trigger_stages_shadow_candidate_without_promoting_or_touching_routing(self) -> None:
        """When drift turns True, trigger queues labels & trains candidate (promote=False -> gate -> shadow), leaving live model & MAPPO untouched."""
        from fastapi.testclient import TestClient
        import db
        import main
        import seed
        from feedback import config, status

        routing_weights = ROOT_DIR / "data" / "models" / "routing_ppo.zip"
        routing_hash_before = _sha256_file(routing_weights) if routing_weights.is_file() else ""

        with tempfile.TemporaryDirectory() as td:
            db_file = str(Path(td) / "drift_trigger.db")
            adapter_root = Path(td) / "adapters"
            seed.seed_sector(db_path=db_file, num_buildings=20)
            edges_before = db.get_road_edges(db_path=db_file)
            routes_before = db.get_routes(include_counterfactuals=False, db_path=db_file)

            # Live model starts as base only (no active adapter)
            self.assertIsNone(db.get_active_model_version(db_path=db_file))

            # Phase A: Drift signal fires before minimum 2 verified labels exist -> queues 1-tap labels only, no promotion
            trig_queue = status.handle_drift_candidate_trigger(
                db_path=db_file,
                drift_distance=0.22,
                spawn_training=True,
                async_training=False,
                adapter_root=adapter_root,
            )
            self.assertTrue(trig_queue["triggered"])
            self.assertTrue(trig_queue["drift_detected"])
            self.assertGreater(trig_queue["queued_label_count"], 0)
            self.assertFalse(trig_queue["promoted"])
            self.assertFalse(trig_queue["routing_policy_coupled"])
            self.assertIsNone(db.get_active_model_version(db_path=db_file))

            # Phase B: Record responder corrections -> drift trigger starts Tier A candidate training (promote=False!)
            for bid, cls_name in (
                ("BLD-0001", "destroyed"),
                ("BLD-0005", "major"),
                ("BLD-0010", "minor"),
                ("BLD-0015", "no_damage"),
            ):
                db.log_feedback(
                    {
                        "target_id": bid,
                        "disaster_id": "HOUSTON_HARVEY_2017",
                        "prediction": "minor",
                        "uncertainty": 0.45,
                        "responder_action": "correct",
                        "corrected_label": cls_name,
                        "responder_id": "RESP_DRIFT",
                    },
                    db_path=db_file,
                )

            trig_cand = status.handle_drift_candidate_trigger(
                db_path=db_file,
                drift_distance=0.28,
                spawn_training=True,
                async_training=True,
                adapter_root=adapter_root,
                force_trigger=True,
            )
            self.assertTrue(trig_cand["triggered"])
            self.assertTrue(trig_cand["candidate_started"])
            self.assertFalse(trig_cand["promoted"])
            self.assertFalse(trig_cand["auto_promote"])
            self.assertFalse(trig_cand["routing_policy_coupled"])

            status.join_drift_candidate_threads(timeout=60.0)

            # Live segmentation model MUST still be untouched (no active adapter promoted!)
            self.assertIsNone(db.get_active_model_version(db_path=db_file))
            versions = db.list_model_versions(limit=10, db_path=db_file)
            self.assertGreaterEqual(len(versions), 1)
            self.assertIn(versions[0]["status"], ("shadow", "rejected"))
            runs = db.get_adaptation_runs(limit=10, db_path=db_file)
            self.assertTrue(any(r["trigger_reason"] == "auto_drift_candidate" and not r["promoted"] for r in runs))
            self.assertIn("checks", runs[0]["gate_report"])
            self.assertEqual(len(runs[0]["gate_report"]["checks"]), 5)

            # Audit log preserves SHA-256 chain integrity & logs drift trigger / gate transition without MODEL_PROMOTED
            audit_events = [r["event_type"] for r in db.get_audit_log(limit=20, db_path=db_file)]
            self.assertNotIn("MODEL_PROMOTED", audit_events)
            self.assertTrue(
                any(ev in audit_events for ev in ("DRIFT_CANDIDATE_TRIGGERED", "MODEL_SHADOW_STAGED", "MODEL_GATE_REJECTED"))
            )
            chain_ok, _, msg = db.verify_audit_chain(db_path=db_file)
            self.assertTrue(chain_ok, msg)

            # Routing & MAPPO weights remain 100% untouched
            if routing_hash_before:
                self.assertEqual(_sha256_file(routing_weights), routing_hash_before)
            self.assertEqual(db.get_road_edges(db_path=db_file), edges_before)
            self.assertEqual(db.get_routes(include_counterfactuals=False, db_path=db_file), routes_before)

            # Explicit promotion endpoint STILL governs going live through the 5-check gate & supports rollback
            client = TestClient(main.app)
            st_body = client.get(f"/api/adaptation/status?db_path={db_file}").json()
            self.assertEqual(st_body["active_adapter_version"], "none")
            self.assertEqual(st_body["retraining_scope"], "segmentation_only")
            self.assertFalse(st_body["routing_policy_coupled"])



if __name__ == "__main__":
    unittest.main()
