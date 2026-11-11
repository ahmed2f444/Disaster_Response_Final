"""Portion 1 — Bi-Temporal Imagery Preprocessing, Mask Extraction, and WGS84 Projection."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import MultiPoint, Polygon
from torch.utils.data import Dataset

from damage.model import IGNORE_INDEX, normalize_damage_label

# ImageNet statistics repeated across pre-disaster (RGB) and post-disaster (RGB) channels
MEAN = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)

# Default Houston-East operational sector bbox (min_lon, min_lat, max_lon, max_lat) aligned with seed.py/app.js
HOUSTON_EAST_BOUNDS: tuple[float, float, float, float] = (-95.202, 29.768, -95.078, 29.858)

SUBTYPE_TO_CLASS: dict[str, int] = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": IGNORE_INDEX,
}


def _is_within_allowed_roots(candidate: Path, allowed_roots: tuple[Path, ...] | None) -> bool:
    if allowed_roots is None:
        return True
    resolved = candidate.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in allowed_roots)


def decode_image_input(
    source: str | bytes | bytearray | Path | Image.Image,
    allowed_roots: tuple[Path, ...] | None = None,
) -> tuple[Image.Image, tuple[float, float, float, float] | None]:
    """Decode PIL Image, raw bytes, base64 data URI, or file path into RGB PIL Image + optional GeoTIFF bounds."""
    if isinstance(source, Image.Image):
        return source.convert("RGB"), _extract_geotiff_bounds(source)

    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(source)))
        bounds = _extract_geotiff_bounds(img)
        return img.convert("RGB"), bounds

    if isinstance(source, Path):
        if not _is_within_allowed_roots(source, allowed_roots):
            raise PermissionError(f"Path outside allowed directories: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"Satellite image file not found: {source}")
        img = Image.open(source)
        bounds = _extract_geotiff_bounds(img)
        return img.convert("RGB"), bounds

    if isinstance(source, str):
        text = source.strip()
        if not text:
            raise ValueError("Empty image input string")
        if text.startswith("data:") and ";base64," in text:
            b64_payload = text.split(";base64,", 1)[1]
            return _decode_base64_image(b64_payload)
        candidate = Path(text)
        if candidate.is_file():
            if not _is_within_allowed_roots(candidate, allowed_roots):
                raise PermissionError(f"Path outside allowed directories: {text}")
            img = Image.open(candidate)
            bounds = _extract_geotiff_bounds(img)
            return img.convert("RGB"), bounds
        return _decode_base64_image(text)

    raise TypeError(f"Unsupported image input type: {type(source).__name__}")


def _decode_base64_image(b64_payload: str) -> tuple[Image.Image, tuple[float, float, float, float] | None]:
    try:
        raw_bytes = base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image string is neither an existing file path nor valid base64 data") from exc
    img = Image.open(io.BytesIO(raw_bytes))
    bounds = _extract_geotiff_bounds(img)
    return img.convert("RGB"), bounds


def _extract_geotiff_bounds(img: Image.Image) -> tuple[float, float, float, float] | None:
    """Extract geographic WGS84 (min_lon, min_lat, max_lon, max_lat) from GeoTIFF tags when present."""
    tags = getattr(img, "tag_v2", None)
    if not tags:
        return None
    scale = tags.get(33550)  # ModelPixelScaleTag: (sx, sy, sz)
    tie = tags.get(33922)    # ModelTiepointTag: (i, j, k, x, y, z)
    if not scale or not tie or len(scale) < 2 or len(tie) < 6:
        return None
    sx, sy = float(scale[0]), float(scale[1])
    px_i, px_j, _, lon0, lat0 = float(tie[0]), float(tie[1]), float(tie[2]), float(tie[3]), float(tie[4])
    if sx <= 0.0 or sy <= 0.0:
        return None
    w, h = img.size
    min_lon = lon0 - px_i * sx
    max_lat = lat0 + px_j * sy
    max_lon = min_lon + w * sx
    min_lat = max_lat - h * sy
    if not (-180.0 <= min_lon < max_lon <= 180.0 and -90.0 <= min_lat < max_lat <= 90.0):
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def stack_pre_post_arrays(pre_img: Image.Image, post_img: Image.Image) -> np.ndarray:
    """Normalize aligned pre/post RGB images into a (6, H, W) float32 tensor array."""
    if post_img.size != pre_img.size:
        post_img = post_img.resize(pre_img.size, Image.Resampling.BILINEAR)
    pre_arr = np.asarray(pre_img, dtype=np.float32) / 255.0
    post_arr = np.asarray(post_img, dtype=np.float32) / 255.0
    stacked = np.concatenate([pre_arr, post_arr], axis=2)  # (H, W, 6)
    normalized = (stacked - MEAN) / STD
    return normalized.transpose(2, 0, 1).astype(np.float32)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicate-pad spatial dims (H, W) of a (B, C, H, W) tensor to multiples of `multiple`."""
    orig_h, orig_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_h = (multiple - (orig_h % multiple)) % multiple
    pad_w = (multiple - (orig_w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (orig_h, orig_w)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (orig_h, orig_w)


def unpad_spatial(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """Slice spatial dimensions of `(..., H_pad, W_pad)` back to `(..., orig_h, orig_w)`."""
    return arr[..., :orig_h, :orig_w]


def prepare_paired_input(
    pre_source: str | bytes | bytearray | Path | Image.Image,
    post_source: str | bytes | bytearray | Path | Image.Image,
    multiple: int = 32,
    allowed_roots: tuple[Path, ...] | None = None,
) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float, float, float] | None]:
    """Decode, normalize, stack, and mod-32 pad a pre/post pair, also returning GeoTIFF bounds if present."""
    pre_img, pre_bounds = decode_image_input(pre_source, allowed_roots=allowed_roots)
    post_img, post_bounds = decode_image_input(post_source, allowed_roots=allowed_roots)
    stacked = stack_pre_post_arrays(pre_img, post_img)
    tensor = torch.from_numpy(stacked).unsqueeze(0)
    padded, orig_hw = pad_to_multiple(tensor, multiple=multiple)
    return padded, orig_hw, pre_bounds or post_bounds


def prepare_paired_tensor(
    pre_source: str | bytes | bytearray | Path | Image.Image,
    post_source: str | bytes | bytearray | Path | Image.Image,
    multiple: int = 32,
    allowed_roots: tuple[Path, ...] | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Decode, normalize, stack, and mod-32 pad a pre/post imagery pair into `(1, 6, H_pad, W_pad)`."""
    padded, orig_hw, _ = prepare_paired_input(
        pre_source, post_source, multiple=multiple, allowed_roots=allowed_roots
    )
    return padded, orig_hw


def _find_root(parent: list[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def _union_roots(parent: list[int], a: int, b: int) -> int:
    ra, rb = _find_root(parent, a), _find_root(parent, b)
    if ra == rb:
        return ra
    if ra < rb:
        parent[rb] = ra
        return ra
    parent[ra] = rb
    return rb


def extract_building_masks(
    probs: np.ndarray,
    fg_threshold: float = 0.35,
    min_area_px: int = 24,
    max_buildings: int = 120,
    max_area_fraction: float = 0.45,
) -> list[dict[str, Any]]:
    """Extract connected building footprint masks and pixel polygons from `(5, H, W)` probabilities."""
    if probs.ndim != 3 or probs.shape[0] < 5:
        raise ValueError(f"Expected probs shape (5, H, W), got {probs.shape}")

    fg_prob = probs[1:5].sum(axis=0)
    fg_mask = fg_prob >= fg_threshold
    h, w = fg_mask.shape

    parent: list[int] = []
    row_runs: list[list[tuple[int, int, int]]] = []
    run_rows: list[tuple[int, int, int]] = []  # (y, x0, x1)

    for y in range(h):
        row = fg_mask[y]
        if not np.any(row):
            row_runs.append([])
            continue
        padded = np.empty(w + 2, dtype=np.uint8)
        padded[0] = 0
        padded[1:-1] = row.view(np.uint8)
        padded[-1] = 0
        diffs = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)

        cur_runs: list[tuple[int, int, int]] = []
        prev_runs = row_runs[y - 1] if y > 0 else []
        p_idx = 0
        for s_val, e_val in zip(starts.tolist(), ends.tolist()):
            run_id = len(parent)
            parent.append(run_id)
            run_rows.append((y, s_val, e_val))
            while p_idx < len(prev_runs) and prev_runs[p_idx][1] <= s_val:
                p_idx += 1
            scan = p_idx
            while scan < len(prev_runs) and prev_runs[scan][0] < e_val:
                _union_roots(parent, run_id, prev_runs[scan][2])
                scan += 1
            cur_runs.append((s_val, e_val, run_id))
        row_runs.append(cur_runs)

    if not parent:
        return []

    comp_runs: dict[int, list[tuple[int, int, int]]] = {}
    comp_areas: dict[int, int] = {}
    for run_id, seg in enumerate(run_rows):
        root = _find_root(parent, run_id)
        comp_runs.setdefault(root, []).append(seg)
        comp_areas[root] = comp_areas.get(root, 0) + (seg[2] - seg[1])

    max_area_px = (
        int(max_area_fraction * h * w)
        if (0.0 < max_area_fraction < 1.0 and (h * w) >= 64)
        else (h * w + 1)
    )
    sorted_roots = sorted(
        (root for root, area in comp_areas.items() if min_area_px <= area <= max_area_px),
        key=lambda r: comp_areas[r],
        reverse=True,
    )[:max_buildings]

    results: list[dict[str, Any]] = []
    for root in sorted_roots:
        segs = comp_runs[root]
        comp_mask = np.zeros((h, w), dtype=bool)
        hull_points: list[tuple[int, int]] = []
        y0, x0, y1, x1 = h, w, 0, 0
        for y, sx0, sx1 in segs:
            comp_mask[y, sx0:sx1] = True
            y0 = min(y0, y)
            y1 = max(y1, y + 1)
            x0 = min(x0, sx0)
            x1 = max(x1, sx1)
            hull_points.append((sx0, y))
            if sx1 - 1 > sx0:
                hull_points.append((sx1 - 1, y))

        coords_px = _hull_coords_from_points(hull_points, x0, y0, x1, y1)
        results.append(
            {
                "mask": comp_mask,
                "area_px": comp_areas[root],
                "bbox": (y0, y1, x0, x1),
                "coords_px": coords_px,
            }
        )
    return results


def _hull_coords_from_points(
    points: list[tuple[int, int]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[tuple[float, float]]:
    if len(points) >= 3:
        geom = MultiPoint(points).convex_hull.simplify(1.5, preserve_topology=True)
        if geom.geom_type == "Polygon" and len(geom.exterior.coords) >= 4:
            return [(float(px), float(py)) for px, py in geom.exterior.coords]
    return [
        (float(x0), float(y0)),
        (float(x1), float(y0)),
        (float(x1), float(y1)),
        (float(x0), float(y1)),
        (float(x0), float(y0)),
    ]


def pixel_polygon_to_wgs84(
    coords_px: list[tuple[float, float]],
    img_width: int,
    img_height: int,
    bounds: tuple[float, float, float, float] = HOUSTON_EAST_BOUNDS,
) -> tuple[dict[str, Any], float, float]:
    """Project pixel polygon `(x, y)` coordinates onto WGS84 bounds and compute centroid `(lat, lon)`."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {img_width}x{img_height}")
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    ring: list[list[float]] = []
    for px, py in coords_px:
        u = min(1.0, max(0.0, float(px) / float(img_width)))
        v = min(1.0, max(0.0, float(py) / float(img_height)))
        lon = round(min_lon + u * lon_span, 6)
        lat = round(max_lat - v * lat_span, 6)
        ring.append([lon, lat])

    if not ring:
        raise ValueError("Cannot project empty polygon coordinate list")
    if ring[0] != ring[-1]:
        ring.append(list(ring[0]))

    poly = Polygon(ring)
    if poly.is_valid and not poly.is_empty:
        cent = poly.centroid
        cent_lon, cent_lat = round(float(cent.x), 6), round(float(cent.y), 6)
    else:
        cent_lon = round(sum(pt[0] for pt in ring[:-1]) / max(1, len(ring) - 1), 6)
        cent_lat = round(sum(pt[1] for pt in ring[:-1]) / max(1, len(ring) - 1), 6)

    return {"type": "Polygon", "coordinates": [ring]}, cent_lat, cent_lon


def load_xbd_footprints(label_json: str | Path) -> list[dict[str, Any]]:
    """Load building polygons and damage labels from an xBD `*_post_disaster.json` file."""
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    lng_lat_by_uid: dict[str, Any] = {}
    for feat in data.get("features", {}).get("lng_lat", []):
        props = feat.get("properties", {})
        uid = props.get("uid")
        if uid and feat.get("wkt"):
            lng_lat_by_uid[str(uid)] = wkt.loads(feat["wkt"])

    buildings: list[dict[str, Any]] = []
    for feat in data.get("features", {}).get("xy", []):
        props = feat.get("properties", {})
        if props.get("feature_type", "building") != "building":
            continue
        subtype = props.get("subtype", "un-classified")
        uid = str(props["uid"])
        cls_idx = SUBTYPE_TO_CLASS.get(subtype, IGNORE_INDEX)
        buildings.append(
            {
                "uid": uid,
                "polygon": wkt.loads(feat["wkt"]),
                "lng_lat_polygon": lng_lat_by_uid.get(uid),
                "subtype": subtype,
                "canonical_label": normalize_damage_label(subtype) if cls_idx != IGNORE_INDEX else None,
                "class": cls_idx,
            }
        )
    return buildings


def rasterize(geom: Any, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a Shapely Polygon/MultiPolygon into a boolean `(H, W)` mask."""
    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if not poly.is_empty:
            draw.polygon(list(poly.exterior.coords), fill=1)
    return np.asarray(img, dtype=bool)


def build_label_mask(buildings: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    """Render per-building damage class indices into a `(H, W)` uint8 target mask."""
    label = np.zeros(shape, dtype=np.uint8)
    for b in buildings:
        label[rasterize(b["polygon"], shape)] = b["class"]
    return label


def list_tiles(disaster: str, n: int, split_dir: Path) -> list[str]:
    """List the first `n` tile IDs for a disaster prefix inside an xBD split directory."""
    label_files = sorted((split_dir / "labels").glob(f"{disaster}_*_post_disaster.json"))
    if len(label_files) < n:
        raise ValueError(f"Only {len(label_files)} tiles for '{disaster}' in {split_dir}, need {n}")
    return [f.name.replace("_post_disaster.json", "") for f in label_files[:n]]


def tile_paths(tile_id: str, split_dir: Path) -> tuple[Path, Path, Path]:
    """Return `(pre_png, post_png, post_label_json)` paths for an xBD tile ID."""
    img_dir = split_dir / "images"
    return (
        img_dir / f"{tile_id}_pre_disaster.png",
        img_dir / f"{tile_id}_post_disaster.png",
        split_dir / "labels" / f"{tile_id}_post_disaster.json",
    )


def load_stacked(pre_path: str | Path, post_path: str | Path) -> np.ndarray:
    """Load pre and post image files into a normalized `(6, H, W)` float32 array."""
    pre_img, _ = decode_image_input(Path(pre_path))
    post_img, _ = decode_image_input(Path(post_path))
    return stack_pre_post_arrays(pre_img, post_img)


class XBDTileDataset(Dataset):
    """PyTorch Dataset yielding normalized 6-channel xBD tile pairs and 5-class target masks."""

    def __init__(
        self,
        tile_ids: list[str],
        split_dir: Path,
        crop: int | None = None,
        augment: bool = False,
    ) -> None:
        self.tile_ids = tile_ids
        self.split_dir = split_dir
        self.crop = crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_id = self.tile_ids[idx]
        pre, post, label_json = tile_paths(tile_id, self.split_dir)
        x = load_stacked(pre, post)
        y = build_label_mask(load_xbd_footprints(label_json), x.shape[1:]).astype(np.int64)

        if self.crop:
            h, w = y.shape
            top = random.randint(0, h - self.crop)
            left = random.randint(0, w - self.crop)
            x = x[:, top : top + self.crop, left : left + self.crop]
            y = y[top : top + self.crop, left : left + self.crop]

        if self.augment and random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, ::-1].copy()

        return torch.from_numpy(x), torch.from_numpy(y)
