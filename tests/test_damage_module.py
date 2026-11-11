"""Unit tests for Portion 1 (Ammar) integrated `damage/` modules.

Covers offline pretrained ResNet-18 U-Net loading, 6-channel preprocessing with
mod-32 padding, connected-component polygon extraction, WGS84 projection,
Grad-CAM spatial region export, encoder embedding export, and Zero-SQL parity.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _png_data_uri(width: int, height: int, rgb: tuple[int, int, int]) -> str:
    img = Image.new("RGB", (width, height), color=rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class TestDamageModelAndPreprocessing(unittest.TestCase):
    """Phase 2: model loading, label mapping, tensor prep, and Grad-CAM/embedding exports."""

    def test_label_normalization_and_class_index_mapping(self) -> None:
        from damage.model import model_class_to_canonical, normalize_damage_label

        self.assertEqual(model_class_to_canonical(1), "no_damage")
        self.assertEqual(model_class_to_canonical(2), "minor")
        self.assertEqual(model_class_to_canonical(3), "major")
        self.assertEqual(model_class_to_canonical(4), "destroyed")
        with self.assertRaises(ValueError):
            model_class_to_canonical(0)
        with self.assertRaises(ValueError):
            model_class_to_canonical(255)

        self.assertEqual(normalize_damage_label("no-damage"), "no_damage")
        self.assertEqual(normalize_damage_label("minor-damage"), "minor")
        self.assertEqual(normalize_damage_label("major-damage"), "major")
        self.assertEqual(normalize_damage_label("destroyed"), "destroyed")
        self.assertEqual(normalize_damage_label("MINOR"), "minor")
        with self.assertRaises(ValueError):
            normalize_damage_label("unknown-tier")

    def test_load_pretrained_checkpoint_and_forward_pass(self) -> None:
        from damage.model import DEFAULT_WEIGHTS_PATH, load_damage_model

        self.assertTrue(DEFAULT_WEIGHTS_PATH.is_file(), f"Missing weights: {DEFAULT_WEIGHTS_PATH}")
        net, device = load_damage_model(DEFAULT_WEIGHTS_PATH)
        x = torch.randn(1, 6, 128, 128, device=device)
        with torch.no_grad():
            logits = net(x)
        self.assertEqual(tuple(logits.shape), (1, 5, 128, 128))

    def test_image_decoding_preprocessing_and_mod32_padding(self) -> None:
        from damage.dataset import prepare_paired_tensor, unpad_spatial

        pre_uri = _png_data_uri(130, 100, (120, 140, 160))
        with tempfile.TemporaryDirectory() as td:
            post_path = Path(td) / "post.png"
            Image.new("RGB", (130, 100), color=(90, 100, 110)).save(post_path)

            tensor, orig_hw = prepare_paired_tensor(pre_uri, post_path)
            self.assertEqual(orig_hw, (100, 130))
            self.assertEqual(tuple(tensor.shape), (1, 6, 128, 160))
            self.assertEqual(tensor.shape[2] % 32, 0)
            self.assertEqual(tensor.shape[3] % 32, 0)

            dummy_probs = np.zeros((5, 128, 160), dtype=np.float32)
            unpadded = unpad_spatial(dummy_probs, orig_hw[0], orig_hw[1])
            self.assertEqual(unpadded.shape, (5, 100, 130))

            with self.assertRaises(PermissionError):
                prepare_paired_tensor(
                    str(post_path),
                    str(post_path),
                    allowed_roots=(ROOT_DIR / "data",),
                )

    def test_connected_component_polygon_extraction_and_wgs84_projection(self) -> None:
        from damage.dataset import (
            HOUSTON_EAST_BOUNDS,
            extract_building_masks,
            pixel_polygon_to_wgs84,
        )

        probs = np.zeros((5, 128, 128), dtype=np.float32)
        probs[0] = 0.95
        # Building 1: 12x12 major damage block
        probs[0, 10:22, 12:24] = 0.05
        probs[3, 10:22, 12:24] = 0.85
        # Building 2: 14x14 destroyed block
        probs[0, 60:74, 70:84] = 0.05
        probs[4, 60:74, 70:84] = 0.90

        components = extract_building_masks(probs, fg_threshold=0.4, min_area_px=25)
        self.assertEqual(len(components), 2)
        for comp in components:
            self.assertIn("mask", comp)
            self.assertIn("coords_px", comp)
            self.assertGreaterEqual(comp["area_px"], 25)
            geojson, cent_lat, cent_lon = pixel_polygon_to_wgs84(
                comp["coords_px"], img_width=128, img_height=128, bounds=HOUSTON_EAST_BOUNDS
            )
            self.assertEqual(geojson["type"], "Polygon")
            ring = geojson["coordinates"][0]
            self.assertEqual(ring[0], ring[-1])
            min_lon, min_lat, max_lon, max_lat = HOUSTON_EAST_BOUNDS
            self.assertTrue(min_lon <= cent_lon <= max_lon)
            self.assertTrue(min_lat <= cent_lat <= max_lat)

    def test_gradcam_export_wgs84_regions_and_embedding_save(self) -> None:
        from damage.dataset import HOUSTON_EAST_BOUNDS
        from damage.exports import (
            FeatureHook,
            building_gradcam,
            gradcam_crop_to_wgs84_regions,
            save_building_exports,
            save_tile_embedding,
        )
        from damage.model import DEFAULT_WEIGHTS_PATH, load_damage_model

        net, device = load_damage_model(DEFAULT_WEIGHTS_PATH)
        hook = FeatureHook(net.decoder)
        x = torch.randn(1, 6, 128, 128, device=device)
        with torch.enable_grad():
            enc_feats = net.encoder(x)
            logits = net(x)
            feat = hook.features
            mask = np.zeros((128, 128), dtype=bool)
            mask[30:56, 40:68] = True
            acts_crop, grad_crop, bbox = building_gradcam(feat, logits[0], mask, cls=3)
        hook.remove()

        self.assertEqual(acts_crop.shape[0], 16)
        self.assertEqual(grad_crop.shape, acts_crop.shape)
        regions = gradcam_crop_to_wgs84_regions(
            acts_crop, grad_crop, bbox, img_shape=(128, 128), bounds=HOUSTON_EAST_BOUNDS, top_k=3
        )
        self.assertGreaterEqual(len(regions), 1)
        self.assertLessEqual(len(regions), 3)
        for item in regions:
            self.assertEqual(item["region"]["type"], "Polygon")
            self.assertTrue(0.0 <= item["weight"] <= 1.0)

        with tempfile.TemporaryDirectory() as td:
            act_p, grad_p, shape = save_building_exports("TILE-1", "BLD-1", acts_crop, grad_crop, out_dir=td)
            self.assertTrue(Path(act_p).is_file())
            self.assertTrue(Path(grad_p).is_file())
            self.assertEqual(shape, list(acts_crop.shape))

            emb_ref, emb_path = save_tile_embedding("TILE-1", enc_feats[-1], out_dir=td)
            self.assertTrue(emb_ref.startswith("emb://resnet18-unet/"))
            self.assertTrue(Path(emb_path).is_file())
            vec = np.load(emb_path)
            self.assertEqual(vec.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_zero_sql_and_no_ammar_part_imports_in_damage_modules(self) -> None:
        sql_keywords = ("SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE")
        damage_dir = ROOT_DIR / "damage"
        for py_file in damage_dir.glob("*.py"):
            if py_file.stat().st_size == 0:
                continue
            src = py_file.read_text(encoding="utf-8")
            self.assertNotIn("ammar part", src.lower(), f"Stale ammar part reference in {py_file.name}")
            for kw in sql_keywords:
                self.assertNotIn(kw, src, f"Raw SQL '{kw}' found in {py_file.name}")


if __name__ == "__main__":
    unittest.main()
