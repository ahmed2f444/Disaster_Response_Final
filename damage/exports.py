"""Portion 1 — Grad-CAM Activation/Gradient Exports and Encoder Embedding Persistence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from damage.dataset import HOUSTON_EAST_BOUNDS, pixel_polygon_to_wgs84

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = ROOT_DIR / "data" / "exports"
GRADCAM_PAD = 4


class FeatureHook:
    """Forward hook capturing final decoder feature maps prior to segmentation head."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.features: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self.features = output

    def remove(self) -> None:
        self.handle.remove()


def _safe_filename_token(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("_") or "item"


def building_gradcam(
    feat: torch.Tensor,
    logits: torch.Tensor,
    mask: np.ndarray,
    cls: int,
    pad: int = GRADCAM_PAD,
    retain_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Compute decoder activations and target-class gradient crop around a building mask."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Cannot compute Grad-CAM on an empty building mask")

    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)

    mask_t = torch.from_numpy(mask).to(logits.device)
    score = logits[cls][mask_t].mean()
    grad = torch.autograd.grad(score, feat, retain_graph=retain_graph)[0][0]

    acts = feat.detach()[0]
    acts_crop = acts[:, y0:y1, x0:x1].cpu().numpy().astype(np.float32)
    grad_crop = grad[:, y0:y1, x0:x1].detach().cpu().numpy().astype(np.float32)
    return acts_crop, grad_crop, (y0, y1, x0, x1)


def gradcam_crop_to_wgs84_regions(
    acts_crop: np.ndarray,
    grad_crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, int],
    bounds: tuple[float, float, float, float] = HOUSTON_EAST_BOUNDS,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Convert Grad-CAM activation/gradient crops into WGS84 GeoJSON attribution regions."""
    y0, y1, x0, x1 = bbox
    img_h, img_w = img_shape

    alpha = grad_crop.mean(axis=(1, 2), keepdims=True)
    cam = np.maximum(0.0, np.sum(alpha * acts_crop, axis=0))
    max_val = float(cam.max()) if cam.size else 0.0
    if max_val > 1e-8:
        cam_norm = cam / max_val
    else:
        cam_norm = np.ones_like(cam, dtype=np.float32) * 0.5

    crop_h, crop_w = cam_norm.shape
    rows = 3 if crop_h >= 6 else (2 if crop_h >= 2 else 1)
    cols = 3 if crop_w >= 6 else (2 if crop_w >= 2 else 1)

    y_edges = np.linspace(0, crop_h, rows + 1, dtype=int)
    x_edges = np.linspace(0, crop_w, cols + 1, dtype=int)

    candidates: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            cy0, cy1 = int(y_edges[r]), int(y_edges[r + 1])
            cx0, cx1 = int(x_edges[c]), int(x_edges[c + 1])
            if cy1 <= cy0 or cx1 <= cx0:
                continue
            cell = cam_norm[cy0:cy1, cx0:cx1]
            weight = round(min(1.0, max(0.05, float(cell.mean()))), 3)
            px_coords = [
                (float(x0 + cx0), float(y0 + cy0)),
                (float(x0 + cx1), float(y0 + cy0)),
                (float(x0 + cx1), float(y0 + cy1)),
                (float(x0 + cx0), float(y0 + cy1)),
                (float(x0 + cx0), float(y0 + cy0)),
            ]
            region_poly, _, _ = pixel_polygon_to_wgs84(
                px_coords, img_width=img_w, img_height=img_h, bounds=bounds
            )
            candidates.append({"region": region_poly, "weight": weight})

    candidates.sort(key=lambda item: item["weight"], reverse=True)
    return candidates[:top_k]


def save_building_exports(
    tile_id: str,
    building_uid: str,
    acts: np.ndarray,
    grads: np.ndarray,
    out_dir: str | Path = DEFAULT_EXPORT_DIR,
) -> tuple[str, str, list[int]]:
    """Save per-building decoder activations and gradients as `.npy` files under `out_dir`."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    safe_tile = _safe_filename_token(tile_id)
    safe_uid = _safe_filename_token(building_uid)
    act_path = dest / f"{safe_tile}_{safe_uid}_activations.npy"
    grad_path = dest / f"{safe_tile}_{safe_uid}_gradients.npy"
    np.save(act_path, acts)
    np.save(grad_path, grads)
    return str(act_path), str(grad_path), list(acts.shape)


def save_tile_embedding(
    tile_id: str,
    encoder_feat: torch.Tensor,
    out_dir: str | Path = DEFAULT_EXPORT_DIR,
) -> tuple[str, str]:
    """Global-average-pool encoder bottleneck features, L2-normalize, save `.npy`, and return ref URI."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    vec = encoder_feat.detach().float().mean(dim=(0, 2, 3)).cpu().numpy()
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    safe_tile = _safe_filename_token(tile_id)
    emb_path = dest / f"{safe_tile}_embedding.npy"
    np.save(emb_path, vec.astype(np.float32))
    return f"emb://resnet18-unet/{safe_tile}", str(emb_path)

