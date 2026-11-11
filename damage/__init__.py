"""Portion 1 — Satellite Building Damage Assessment Package.

Provides pretrained ResNet-18 U-Net model loading, bi-temporal preprocessing,
connected-component polygon extraction, WGS84 projection, Grad-CAM exports,
and encoder embedding persistence integrated with Portion 5 (`db.py`).
"""

from damage.model import (
    CLASS_INDEX_TO_CANONICAL,
    DEFAULT_WEIGHTS_PATH,
    IGNORE_INDEX,
    NUM_CLASSES,
    build_model,
    load_damage_model,
    model_class_to_canonical,
    normalize_damage_label,
)
from damage.dataset import (
    HOUSTON_EAST_BOUNDS,
    extract_building_masks,
    pixel_polygon_to_wgs84,
    prepare_paired_tensor,
    unpad_spatial,
)
from damage.exports import (
    DEFAULT_EXPORT_DIR,
    FeatureHook,
    building_gradcam,
    gradcam_crop_to_wgs84_regions,
    save_building_exports,
    save_tile_embedding,
)

__all__ = [
    "CLASS_INDEX_TO_CANONICAL",
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_WEIGHTS_PATH",
    "HOUSTON_EAST_BOUNDS",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "FeatureHook",
    "build_model",
    "building_gradcam",
    "extract_building_masks",
    "gradcam_crop_to_wgs84_regions",
    "load_damage_model",
    "model_class_to_canonical",
    "normalize_damage_label",
    "pixel_polygon_to_wgs84",
    "prepare_paired_tensor",
    "save_building_exports",
    "save_tile_embedding",
    "unpad_spatial",
]

