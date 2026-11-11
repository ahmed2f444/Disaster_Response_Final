"""Portion 3 (Yousef) — Pixel-Level Grad-CAM Saliency, WGS84 Regions & Overlay Renderer.

Consumes Portion 1's exported final decoder activations and target-class gradients
(`data/exports/{tile_id}_{building_id}_activations.npy` and `_gradients.npy`) as well
as legacy `{building_id}_act.npy` / `_grad.npy` files, producing normalized CAM maps,
WGS84 attribution polygons (`gradcam_attribution`), spatial summaries, and PNG overlays.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = ROOT_DIR / "data" / "exports"
DEFAULT_UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
_REAL_SAMPLE_POST = ROOT_DIR / "static" / "samples" / "hurricane-matthew_00000030_post_disaster.png"
DEFAULT_SAMPLE_POST = (
    _REAL_SAMPLE_POST
    if _REAL_SAMPLE_POST.is_file()
    else (ROOT_DIR / "static" / "samples" / "test_fixture_T1_post_disaster.png")
)
HOUSTON_EAST_BOUNDS: tuple[float, float, float, float] = (-95.202, 29.768, -95.078, 29.858)


def _safe_token(raw: str) -> str:
    """Sanitize an identifier for safe filename matching without path traversal."""
    text = str(raw or "").strip()
    if not text or "\x00" in text or ".." in text or "/" in text or "\\" in text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")


def _allowed_roots(extra_dir: Path | None = None) -> tuple[Path, ...]:
    roots = [
        (ROOT_DIR / "data").resolve(),
        (ROOT_DIR / "static" / "samples").resolve(),
    ]
    if extra_dir is not None:
        roots.append(extra_dir.resolve())
    if "pytest" in sys.modules:
        roots.append(Path(tempfile.gettempdir()).resolve())
    return tuple(roots)


def _is_within_roots(candidate: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def find_building_export_pair(
    building_id: str,
    export_dir: str | Path | None = None,
) -> tuple[Path, Path] | None:
    """Locate the newest Portion 1 (`*_activations.npy`/`*_gradients.npy`) or legacy pair."""
    safe_uid = _safe_token(building_id)
    if not safe_uid:
        return None
    dest = Path(export_dir if export_dir is not None else DEFAULT_EXPORT_DIR).resolve()
    if not dest.is_dir():
        return None

    candidates: list[Path] = []
    for pattern in (f"*_{safe_uid}_activations.npy", f"{safe_uid}_activations.npy"):
        for act_path in dest.glob(pattern):
            if act_path.is_file() and _is_within_roots(act_path, (dest,)):
                candidates.append(act_path)

    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for act_path in candidates:
        grad_name = act_path.name[: -len("_activations.npy")] + "_gradients.npy"
        grad_path = act_path.with_name(grad_name)
        if grad_path.is_file() and _is_within_roots(grad_path, (dest,)):
            return act_path, grad_path

    legacy_act = dest / f"{safe_uid}_act.npy"
    legacy_grad = dest / f"{safe_uid}_grad.npy"
    if legacy_act.is_file() and legacy_grad.is_file():
        return legacy_act, legacy_grad
    return None


def compute_cam_map(activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Compute ReLU-gated, max-normalized 2D Grad-CAM map from `(C, H, W)` tensors."""
    acts = np.asarray(activations, dtype=np.float32)
    grads = np.asarray(gradients, dtype=np.float32)
    if acts.ndim != 3 or grads.shape != acts.shape or acts.size == 0:
        raise ValueError(f"Expected matching non-empty 3D (C, H, W) arrays, got {acts.shape} and {grads.shape}")

    acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
    grads = np.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)

    weights = np.mean(grads, axis=(1, 2))
    cam = np.maximum(np.sum(weights[:, None, None] * acts, axis=0), 0.0)
    max_val = float(cam.max()) if cam.size else 0.0

    # Segmentation crops mix negative background grads with positive target grads;
    # fall back to positive-gradient channel weighting when global mean cancels out.
    if max_val <= 1e-8:
        pos_weights = np.mean(np.maximum(grads, 0.0), axis=(1, 2))
        cam_pos = np.maximum(np.sum(pos_weights[:, None, None] * acts, axis=0), 0.0)
        if float(cam_pos.max()) > 1e-12:
            cam = cam_pos
            max_val = float(cam.max())
        else:
            cam_elem = np.maximum(np.sum(grads * acts, axis=0), 0.0)
            if float(cam_elem.max()) > 1e-12:
                cam = cam_elem
                max_val = float(cam.max())

    if max_val > 1e-12:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam.astype(np.float32)


def summarize_cam_hotspot(cam: np.ndarray) -> dict[str, Any]:
    """Extract deterministic spatial hotspot statistics from a normalized 2D CAM map."""
    if cam.ndim != 2 or cam.size == 0:
        return {
            "peak_saliency": 0.0,
            "mean_saliency": 0.0,
            "active_fraction": 0.0,
            "quadrant": "central",
            "summary": "No positive decoder saliency detected.",
        }

    h, w = cam.shape
    peak = float(cam.max())
    mean = float(cam.mean())
    active_frac = float(np.mean(cam >= 0.5))

    mask = cam >= max(0.5 * peak, 1e-6) if peak > 1e-6 else np.ones_like(cam, dtype=bool)
    ys, xs = np.where(mask)
    cy = float(ys.mean()) / max(1.0, float(h - 1)) if h > 1 else 0.5
    cx = float(xs.mean()) / max(1.0, float(w - 1)) if w > 1 else 0.5

    v_pos = "top" if cy < 0.38 else ("bottom" if cy > 0.62 else "middle")
    h_pos = "left" if cx < 0.38 else ("right" if cx > 0.62 else "center")
    quadrant = "central" if (v_pos == "middle" and h_pos == "center") else f"{v_pos}-{h_pos}"

    summary = (
        f"Decoder Grad-CAM concentrates in the {quadrant} building region "
        f"(peak intensity {peak:.2f}, active area ratio {active_frac:.2f})."
    )
    return {
        "peak_saliency": round(peak, 4),
        "mean_saliency": round(mean, 4),
        "active_fraction": round(active_frac, 4),
        "quadrant": quadrant,
        "summary": summary,
    }


def cam_to_wgs84_regions(
    cam: np.ndarray,
    geometry: dict[str, Any] | None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Partition a 2D CAM map into top-K WGS84 GeoJSON attribution sub-polygons."""
    ring = (
        geometry.get("coordinates", [[]])[0]
        if isinstance(geometry, dict) and isinstance(geometry.get("coordinates"), list)
        else []
    )
    if len(ring) < 3 or cam.ndim != 2 or cam.size == 0:
        return []

    lons = [float(pt[0]) for pt in ring]
    lats = [float(pt[1]) for pt in ring]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    lon_span = max(1e-7, max_lon - min_lon)
    lat_span = max(1e-7, max_lat - min_lat)

    h, w = cam.shape
    rows = 3 if h >= 6 else (2 if h >= 2 else 1)
    cols = 3 if w >= 6 else (2 if w >= 2 else 1)
    y_edges = np.linspace(0, h, rows + 1, dtype=int)
    x_edges = np.linspace(0, w, cols + 1, dtype=int)

    regions: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            if y1 <= y0 or x1 <= x0:
                continue
            cell = cam[y0:y1, x0:x1]
            weight = round(min(1.0, max(0.05, float(cell.mean()))), 3)

            sub_min_lon = round(min_lon + (x0 / w) * lon_span, 6)
            sub_max_lon = round(min_lon + (x1 / w) * lon_span, 6)
            sub_max_lat = round(max_lat - (y0 / h) * lat_span, 6)
            sub_min_lat = round(max_lat - (y1 / h) * lat_span, 6)
            poly = {
                "type": "Polygon",
                "coordinates": [[
                    [sub_min_lon, sub_max_lat],
                    [sub_max_lon, sub_max_lat],
                    [sub_max_lon, sub_min_lat],
                    [sub_min_lon, sub_min_lat],
                    [sub_min_lon, sub_max_lat],
                ]],
            }
            regions.append({"region": poly, "weight": weight})

    regions.sort(key=lambda item: item["weight"], reverse=True)
    return regions[: max(1, int(top_k))]


def _load_active_tile_context(
    export_dir: Path,
    tile_image_path: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Find active post-disaster tile image and optional bbox metadata cleanly."""
    roots = _allowed_roots(export_dir)
    meta: dict[str, Any] | None = None

    for meta_candidate in (
        export_dir / "_active_pair" / "latest_meta.json",
        DEFAULT_UPLOADS_DIR / "latest_meta.json",
        DEFAULT_EXPORT_DIR / "_active_pair" / "latest_meta.json",
    ):
        if meta_candidate.is_file() and _is_within_roots(meta_candidate, roots):
            try:
                loaded = json.loads(meta_candidate.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
                    break
            except Exception:
                pass

    if tile_image_path is not None:
        raw_text = str(tile_image_path).strip()
        if raw_text and ".." not in Path(raw_text).parts:
            cand = Path(raw_text)
            if not cand.is_absolute():
                cand = (ROOT_DIR / cand) if cand.parts and cand.parts[0].lower() in {"data", "static"} else (export_dir / cand)
            if cand.is_file() and _is_within_roots(cand, roots):
                return cand.resolve(), meta

    candidates: list[Path] = []
    if meta and meta.get("post_path"):
        candidates.append(Path(str(meta["post_path"])))
    candidates.extend([
        export_dir / "_active_pair" / "latest_post.png",
        DEFAULT_UPLOADS_DIR / "latest_post.png",
        DEFAULT_EXPORT_DIR / "_active_pair" / "latest_post.png",
        DEFAULT_SAMPLE_POST,
    ])
    for cand in candidates:
        if cand.is_file() and _is_within_roots(cand, roots):
            return cand.resolve(), meta
    return None, meta


def _crop_image_to_building(
    img: Image.Image,
    building_geometry: dict[str, Any] | None,
    meta: dict[str, Any] | None,
) -> Image.Image:
    """Crop full satellite tile to the building WGS84 bbox when metadata aligns."""
    if not isinstance(building_geometry, dict) or not isinstance(meta, dict):
        return img
    bounds = meta.get("bounds")
    ring = building_geometry.get("coordinates", [[]])[0]
    if not (isinstance(bounds, list) and len(bounds) == 4 and len(ring) >= 3):
        return img

    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bounds]
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    if lon_span <= 1e-9 or lat_span <= 1e-9:
        return img

    b_lons = [float(p[0]) for p in ring]
    b_lats = [float(p[1]) for p in ring]
    w, h = img.size

    x0 = int(max(0, np.floor(((min(b_lons) - min_lon) / lon_span) * w) - 4))
    x1 = int(min(w, np.ceil(((max(b_lons) - min_lon) / lon_span) * w) + 4))
    y0 = int(max(0, np.floor(((max_lat - max(b_lats)) / lat_span) * h) - 4))
    y1 = int(min(h, np.ceil(((max_lat - min(b_lats)) / lat_span) * h) + 4))

    if (x1 - x0) >= 8 and (y1 - y0) >= 8:
        return img.crop((x0, y0, x1, y1))
    return img


def render_gradcam_overlay(
    cam: np.ndarray,
    out_path: Path,
    tile_image_path: Path | None = None,
    building_geometry: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Render a jet-colormap Grad-CAM overlay PNG and persist it to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cam_h, cam_w = cam.shape
    target_w = max(192, cam_w)
    target_h = max(192, int(round(target_w * (cam_h / max(1, cam_w)))))

    if tile_image_path is not None and tile_image_path.is_file():
        with Image.open(tile_image_path) as src:
            base = _crop_image_to_building(src.convert("RGB"), building_geometry, meta)
            base_resized = base.resize((target_w, target_h), Image.Resampling.BILINEAR)
    else:
        base_resized = Image.new("RGB", (target_w, target_h), color=(12, 20, 36))

    cam_img = Image.fromarray(np.clip(cam * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    cam_up = np.asarray(cam_img.resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    img_arr = np.asarray(base_resized, dtype=np.float32) / 255.0
    heatmap = cm.jet(cam_up)[:, :, :3].astype(np.float32)
    overlay = np.clip((0.5 * img_arr + 0.5 * heatmap) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(overlay).save(out_path, format="PNG")
    return out_path


def compute_gradcam(
    building_id: str,
    tile_image_path: str | Path | None = None,
    export_dir: str | Path | None = None,
    building_geometry: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Load `.npy` activations/gradients, compute Grad-CAM, and save PNG overlay."""
    dest = Path(export_dir if export_dir is not None else DEFAULT_EXPORT_DIR).resolve()
    pair = find_building_export_pair(building_id, export_dir=dest)
    if pair is None:
        return None, "Missing activation or gradient files."

    act_path, grad_path = pair
    try:
        activations = np.load(act_path)
        gradients = np.load(grad_path)
        cam = compute_cam_map(activations, gradients)
    except Exception as exc:
        return None, f"Failed to compute Grad-CAM: {exc}"

    safe_uid = _safe_token(building_id) or "building"
    resolved_tile, meta = _load_active_tile_context(dest, tile_image_path=tile_image_path)
    out_path = dest / f"{safe_uid}_gradcam.png"
    try:
        render_gradcam_overlay(
            cam=cam,
            out_path=out_path,
            tile_image_path=resolved_tile,
            building_geometry=building_geometry,
            meta=meta,
        )
        return str(out_path), "Success"
    except Exception as exc:
        return None, f"Failed to process image: {exc}"


def compute_building_gradcam(
    building: dict[str, Any],
    tile_image_path: str | Path | None = None,
    export_dir: str | Path | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Unified Portion 3 Grad-CAM pipeline returning WGS84 regions, summary & PNG path."""
    building_id = str(building.get("building_id") or "")
    geometry = building.get("geometry") if isinstance(building.get("geometry"), dict) else None
    cached_regions = list(building.get("gradcam_attribution") or [])
    dest = Path(export_dir if export_dir is not None else DEFAULT_EXPORT_DIR).resolve()

    pair = find_building_export_pair(building_id, export_dir=dest)
    if pair is not None:
        act_path, grad_path = pair
        activations = np.load(act_path)
        gradients = np.load(grad_path)
        cam = compute_cam_map(activations, gradients)
        hotspot = summarize_cam_hotspot(cam)

        safe_uid = _safe_token(building_id) or "building"
        resolved_tile, meta = _load_active_tile_context(dest, tile_image_path=tile_image_path)
        overlay_path = render_gradcam_overlay(
            cam=cam,
            out_path=dest / f"{safe_uid}_gradcam.png",
            tile_image_path=resolved_tile,
            building_geometry=geometry,
            meta=meta,
        )
        regions = cached_regions or cam_to_wgs84_regions(cam, geometry=geometry, top_k=top_k)
        return {
            "building_id": building_id,
            "available": True,
            "source": "npy_exports",
            "regions": regions,
            "overlay_path": str(overlay_path),
            "hotspot_summary": hotspot["summary"],
            "quadrant": hotspot["quadrant"],
            "peak_saliency": hotspot["peak_saliency"],
            "mean_saliency": hotspot["mean_saliency"],
            "active_fraction": hotspot["active_fraction"],
        }

    if cached_regions:
        top_weight = max(float(r.get("weight", 0.0)) for r in cached_regions)
        return {
            "building_id": building_id,
            "available": True,
            "source": "cached_regions",
            "regions": cached_regions[:top_k],
            "overlay_path": None,
            "hotspot_summary": (
                f"Cached Grad-CAM spatial attribution highlights {len(cached_regions[:top_k])} "
                f"building region(s) (peak weight {top_weight:.2f})."
            ),
            "quadrant": "central",
            "peak_saliency": round(top_weight, 4),
            "mean_saliency": round(float(np.mean([float(r.get("weight", 0.0)) for r in cached_regions])), 4),
            "active_fraction": 0.0,
        }

    return {
        "building_id": building_id,
        "available": False,
        "source": "none",
        "regions": [],
        "overlay_path": None,
        "hotspot_summary": "No Grad-CAM activations or attribution regions recorded for this building.",
        "quadrant": "unknown",
        "peak_saliency": 0.0,
        "mean_saliency": 0.0,
        "active_fraction": 0.0,
    }
