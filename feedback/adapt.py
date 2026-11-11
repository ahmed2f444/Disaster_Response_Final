"""Portion 4 — Continual Adaptation Engine (Tier A / B / C, Segmentation Only).

Implements Portion 4 Plan Sec. 3 & Unified System Doc Sec. 8 / Stage 9:
- Tier A (field, <30m target / <1h): Freeze ResNet-18 encoder & decoder; train
  lightweight per-disaster residual adapter (`ResidualAdapterUNet`) with
  class-weighted Focal Loss (`P10` rare "destroyed" class focus).
- Tier B (2-3h): Generate conservative pseudo-label masks from hazard footprints.
- Tier C (weekly): Fine-tune using `DualReplayBuffer` + logged feedback.
- Strictly scoped to Portion 1 segmentation; never modifies Portion 2 routing.
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon
from torch import nn

from damage.assess import _wgs84_to_pixel_coords
from damage.dataset import HOUSTON_EAST_BOUNDS, pad_to_multiple, prepare_paired_input, rasterize, unpad_spatial
from damage.model import (
    DEFAULT_WEIGHTS_PATH,
    IGNORE_INDEX,
    NUM_CLASSES,
    AdaptedSegmentationModel,
    build_model,
    load_damage_model,
    resolve_device,
)
from db import DEFAULT_DB_PATH, get_buildings, get_feedback_stats, get_feedback_logs
from feedback import config
from feedback.drift import compute_drift_distance
from feedback.gate import compute_pixel_metrics, evaluate_safety_gate
from feedback.replay import DualReplayBuffer
from feedback.status import promote_or_stage_adapter


class ResidualAdapterUNet(AdaptedSegmentationModel):
    """Frozen-backbone ResNet-18 U-Net adapter wrapper for parameter-isolated continual learning."""

    def __init__(self, base_model: nn.Module, adapter_channels: int = 16) -> None:
        super().__init__(base_model, bottleneck=adapter_channels)
        self.freeze_backbone()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional per-class weights and ignore_index support."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0
        ce = F.cross_entropy(
            logits.float(),
            targets.long(),
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        probs = torch.softmax(logits.float(), dim=1)
        safe_targets = torch.where(valid_mask, targets.long(), torch.zeros_like(targets.long()))
        pt = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal[valid_mask].mean()


def compute_inverse_class_weights(targets: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute bounded inverse-frequency class weights emphasizing rare Destroyed pixels."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for y in targets:
        arr = y.detach().cpu().numpy().ravel()
        valid = arr[(arr != IGNORE_INDEX) & (arr >= 0) & (arr < NUM_CLASSES)]
        if valid.size > 0:
            counts += np.bincount(valid.astype(np.int64), minlength=NUM_CLASSES)[:NUM_CLASSES]
    total = max(float(counts.sum()), 1.0)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls_idx in range(NUM_CLASSES):
        if counts[cls_idx] > 0:
            raw = total / (NUM_CLASSES * counts[cls_idx])
            weights[cls_idx] = float(np.clip(raw, 0.25, 8.0))
        else:
            weights[cls_idx] = 1.0
    # Keep background anchored and boost severe damage classes relative to background
    weights[0] = max(weights[0], 0.5)
    weights[4] = max(weights[4], 2.0)
    return torch.from_numpy(weights)


def _anchor_background_pixels(
    target_mask: torch.Tensor,
    base_probs: torch.Tensor,
    min_bg_prob: float = 0.60,
) -> torch.Tensor:
    """Supervise confident base-model background pixels outside labeled polygons as class 0."""
    anchored = target_mask.clone()
    bg_confident = (anchored == IGNORE_INDEX) & (base_probs.argmax(dim=0) == 0) & (base_probs[0] >= min_bg_prob)
    anchored[bg_confident] = 0
    return anchored


def _safe_token(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw))
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def run_tier_a_adaptation(
    samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    disaster_id: str = "HOUSTON_HARVEY_2017",
    adapter_version: str | None = None,
    adapter_root: str | Path | None = None,
    weights_path: str | Path | None = None,
    epochs: int = 4,
    learning_rate: float = 5e-3,
    bottleneck: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    """Run Tier A frozen-backbone adapter fine-tuning on labeled disaster crops."""
    if not samples:
        raise ValueError("Tier A adaptation requires at least one labeled (image_6ch, mask) sample")

    t0 = time.perf_counter()
    eff_bottleneck = max(1, int(bottleneck))
    eff_epochs = max(1, int(epochs))
    target_device = resolve_device(device)
    ckpt = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if ckpt.is_file():
        base_net, target_device = load_damage_model(weights_path=ckpt, device=target_device, use_cache=False)
    else:
        base_net = build_model(pretrained_encoder=False).to(target_device)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = ResidualAdapterUNet(base_net, adapter_channels=eff_bottleneck).to(target_device)
    model.train()
    # Keep BatchNorm statistics in base backbone frozen during small-batch field adaptation
    model.base_model.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    class_weights = compute_inverse_class_weights([y for _, y in samples]).to(target_device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)

    losses: list[float] = []
    for _ in range(eff_epochs):
        epoch_loss = 0.0
        for x_sample, y_sample in samples:
            xb = x_sample.unsqueeze(0).to(target_device) if x_sample.ndim == 3 else x_sample.to(target_device)
            yb = y_sample.unsqueeze(0).to(target_device) if y_sample.ndim == 2 else y_sample.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        mean_loss = epoch_loss / len(samples)
        if not math.isfinite(mean_loss):
            raise RuntimeError("Tier A adaptation diverged with non-finite loss; aborting adapter save")
        losses.append(round(mean_loss, 6))

    ver = adapter_version or f"{_safe_token(disaster_id).lower()}-tier-a-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    safe_disaster = _safe_token(disaster_id)
    safe_ver = _safe_token(ver)
    root = Path(adapter_root or config.ADAPTER_DIR).resolve()
    out_dir = (root / safe_disaster / safe_ver).resolve()
    if not out_dir.is_relative_to(root):
        raise ValueError("Resolved adapter directory escapes configured adapter root")
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = out_dir / "adapter.pt"
    torch.save(
        {
            "disaster_id": disaster_id,
            "adapter_version": ver,
            "bottleneck": eff_bottleneck,
            "adapter_state_dict": model.adapter_block.state_dict(),
            "adapter_scale": model.adapter_scale.detach().cpu(),
            "losses": losses,
        },
        adapter_file,
    )

    elapsed = round(time.perf_counter() - t0, 4)
    return {
        "tier": "tier_a",
        "scope": "segmentation_only",
        "disaster_id": disaster_id,
        "adapter_version": ver,
        "adapter_path": str(adapter_file),
        "samples_used": len(samples),
        "epochs": eff_epochs,
        "final_loss": losses[-1],
        "duration_sec": elapsed,
        "trainable_param_count": sum(p.numel() for p in trainable_params),
    }


def generate_hazard_pseudo_masks(
    probs: np.ndarray,
    hazard_mask: np.ndarray,
    high_conf_threshold: float = 0.85,
) -> np.ndarray:
    """Tier B (2-3h): combine high-confidence predictions with hazard footprint masks."""
    if probs.ndim != 3 or probs.shape[0] < NUM_CLASSES:
        raise ValueError(f"Expected probs shape ({NUM_CLASSES}, H, W), got {probs.shape}")
    pred_cls = probs.argmax(axis=0)
    max_p = probs.max(axis=0)
    pseudo = np.full(pred_cls.shape, IGNORE_INDEX, dtype=np.int64)
    confident = max_p >= high_conf_threshold
    pseudo[confident] = pred_cls[confident]
    # Inside hazard extent, conservative minor/major bump only where model already sees a building
    haz_bool = np.asarray(hazard_mask, dtype=bool)
    in_hazard_bld = haz_bool & (pred_cls >= 1) & (max_p >= 0.55)
    pseudo[in_hazard_bld] = np.maximum(pred_cls[in_hazard_bld], 2)
    return pseudo


def run_tier_c_scheduled_finetune(
    sample_loader: dict[str, tuple[torch.Tensor, torch.Tensor]],
    disaster_id: str = "HOUSTON_HARVEY_2017",
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = 16,
    epochs: int = 6,
    adapter_root: str | Path | None = None,
) -> dict[str, Any]:
    """Tier C (weekly): draw prioritized non-FIFO replay batch + train segmentation adapter."""
    buf = DualReplayBuffer(db_path=db_path)
    replay_items = buf.sample_batch(batch_size=batch_size, prioritized=True)
    matched = [sample_loader[item["target_id"]] for item in replay_items if item["target_id"] in sample_loader]
    if not matched:
        raise ValueError("No labeled tensor crops matched replay buffer sample IDs for Tier C adaptation")
    ver = f"{_safe_token(disaster_id).lower()}-tier-c-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    res = run_tier_a_adaptation(
        samples=matched,
        disaster_id=disaster_id,
        adapter_version=ver,
        adapter_root=adapter_root,
        epochs=epochs,
        learning_rate=3e-3,
    )
    res["tier"] = "tier_c"
    res["replay_batch_size"] = len(matched)
    return res


_LABEL_TO_CLASS_IDX: dict[str, int] = {
    "no_damage": 1,
    "no-damage": 1,
    "minor": 2,
    "minor-damage": 2,
    "major": 3,
    "major-damage": 3,
    "destroyed": 4,
}


def _resolve_active_imagery_for_adaptation(
    data_dir: Path | None = None,
    root_dir: Path | None = None,
) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float, float, float], str]:
    """Resolve the genuine pre/post satellite pair from uploads or test fixture."""
    root = (root_dir or config.ROOT_DIR).resolve()
    data_root = (data_dir or (root / "data")).resolve()
    uploads_dir = (data_root / "uploads").resolve()
    meta_file = uploads_dir / "latest_meta.json"

    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        pre_p = Path(str(meta.get("pre_path") or (uploads_dir / "latest_pre.png"))).resolve()
        post_p = Path(str(meta.get("post_path") or (uploads_dir / "latest_post.png"))).resolve()
        if pre_p.is_relative_to(data_root) and post_p.is_relative_to(data_root) and pre_p.is_file() and post_p.is_file():
            raw_b = meta.get("bounds")
            bounds = (
                (float(raw_b[0]), float(raw_b[1]), float(raw_b[2]), float(raw_b[3]))
                if isinstance(raw_b, list) and len(raw_b) == 4
                else HOUSTON_EAST_BOUNDS
            )
            padded, (orig_h, orig_w), geo_b = prepare_paired_input(pre_p, post_p, multiple=32, allowed_roots=(data_root,))
            return padded[0, :, :orig_h, :orig_w], (orig_h, orig_w), geo_b or bounds, "data/uploads/latest_pair"

    fixture_dir = (root / "test images").resolve()
    t0_path = fixture_dir / "test_fixture_T0_pre_disaster.png"
    t1_path = fixture_dir / "test_fixture_T1_post_disaster.png"
    if t0_path.is_file() and t1_path.is_file():
        padded, (orig_h, orig_w), geo_b = prepare_paired_input(t0_path, t1_path, multiple=32, allowed_roots=(root,))
        return padded[0, :, :orig_h, :orig_w], (orig_h, orig_w), geo_b or HOUSTON_EAST_BOUNDS, "test_images_fixture"

    raise ValueError("No real pre/post satellite imagery pair available on disk for Tier A adaptation.")


def _rasterize_building_mask(
    buildings_map: dict[str, dict[str, Any]],
    building_ids: Sequence[str],
    class_lookup: Callable[[str, dict[str, Any]], int | None],
    img_h: int,
    img_w: int,
    bounds: tuple[float, float, float, float],
) -> tuple[torch.Tensor, list[str]]:
    """Rasterize labeled WGS84 footprints onto an IGNORE_INDEX-initialized tensor mask."""
    mask = np.full((img_h, img_w), IGNORE_INDEX, dtype=np.int64)
    covered_ids: list[str] = []
    for bid in building_ids:
        bld = buildings_map.get(bid)
        if not bld:
            continue
        geom = bld.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if not coords or not coords[0]:
            continue
        ring = _wgs84_to_pixel_coords(coords[0], img_w, img_h, bounds)
        if len(ring) < 3:
            continue
        poly = Polygon(ring)
        if poly.is_empty:
            continue
        cls_idx = class_lookup(bid, bld)
        if cls_idx is None:
            continue
        px_mask = rasterize(poly, (img_h, img_w))
        if np.any(px_mask):
            mask[px_mask] = cls_idx
            covered_ids.append(bid)
    return torch.from_numpy(mask), covered_ids


def run_feedback_driven_tier_a(
    disaster_id: str = "HOUSTON_HARVEY_2017",
    adapter_version: str | None = None,
    epochs: int = 8,
    learning_rate: float = 2e-2,
    promote: bool = True,
    tier: str = "tier_a",
    trigger_reason: str = "manual_ui_tier_a",
    db_path: str = DEFAULT_DB_PATH,
    adapter_root: str | Path | None = None,
    weights_path: str | Path | None = None,
    data_dir: Path | None = None,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """Train a Tier A adapter on real imagery + responder feedback, evaluate 5 gate checks, and promote if passed."""
    x_full, (orig_h, orig_w), bounds, imagery_source = _resolve_active_imagery_for_adaptation(
        data_dir=data_dir, root_dir=root_dir
    )
    buildings_list = get_buildings(db_path=db_path)
    buildings_map = {str(b["building_id"]): b for b in buildings_list}
    logs = get_feedback_logs(limit=5000, db_path=db_path)

    verified: dict[str, int] = {}
    for row in logs:
        tid = str(row.get("target_id") or "")
        act = str(row.get("responder_action") or "").lower()
        if tid in verified or tid not in buildings_map or act not in {"confirm", "correct"}:
            continue
        lbl_raw = str(row.get("corrected_label") or row.get("prediction") or "").strip().lower()
        if lbl_raw in _LABEL_TO_CLASS_IDX:
            verified[tid] = _LABEL_TO_CLASS_IDX[lbl_raw]

    _, covered_verified = _rasterize_building_mask(
        buildings_map, list(verified.keys()), lambda tid, _: verified[tid], orig_h, orig_w, bounds
    )
    if len(covered_verified) < 2:
        raise ValueError(
            "Need at least 2 verified (Confirm/Correct) building labels overlapping active imagery "
            "to split training vs geographic holdout."
        )

    sorted_ver = sorted(covered_verified, key=lambda bid: float(buildings_map[bid].get("centroid_lon") or 0.0))
    split_idx = max(1, min(len(sorted_ver) - 1, len(sorted_ver) // 2))
    train_ids = sorted_ver[:split_idx]
    holdout_ids = sorted_ver[split_idx:]

    old_candidates = [
        str(b["building_id"])
        for b in buildings_list
        if str(b["building_id"]) not in verified and b.get("status_flag") == "CALIBRATED"
    ][:40]
    if not old_candidates:
        old_candidates = [bid for bid in train_ids]

    y_train, _ = _rasterize_building_mask(
        buildings_map, train_ids, lambda tid, _: verified[tid], orig_h, orig_w, bounds
    )
    y_hold, _ = _rasterize_building_mask(
        buildings_map, holdout_ids, lambda tid, _: verified[tid], orig_h, orig_w, bounds
    )
    y_all_fb, _ = _rasterize_building_mask(
        buildings_map, sorted_ver, lambda tid, _: verified[tid], orig_h, orig_w, bounds
    )
    y_old, covered_old = _rasterize_building_mask(
        buildings_map,
        old_candidates,
        lambda _, b: _LABEL_TO_CLASS_IDX.get(str(b.get("damage_class") or "").strip().lower()),
        orig_h,
        orig_w,
        bounds,
    )
    if not covered_old:
        y_old = y_train

    ckpt = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    base_net, device = load_damage_model(weights_path=ckpt, use_cache=False)
    with torch.no_grad():
        xb_pad, _ = pad_to_multiple(x_full.unsqueeze(0).to(device), multiple=32)
        base_probs_pad = torch.softmax(base_net(xb_pad).float(), dim=1)[0].cpu()
        base_probs = torch.from_numpy(unpad_spatial(base_probs_pad.numpy(), orig_h, orig_w))

    y_train_bg = _anchor_background_pixels(y_train, base_probs)
    y_old_bg = _anchor_background_pixels(y_old, base_probs)
    y_all_bg = _anchor_background_pixels(y_all_fb, base_probs)

    base_fb_metrics = compute_pixel_metrics(base_net, [(x_full, y_all_bg)], device=device)
    base_hold_metrics = compute_pixel_metrics(base_net, [(x_full, y_hold)], device=device)
    base_old_metrics = compute_pixel_metrics(base_net, [(x_full, y_old_bg)], device=device)

    train_res = run_tier_a_adaptation(
        samples=[(x_full, y_train_bg), (x_full, y_old_bg)],
        disaster_id=disaster_id,
        adapter_version=adapter_version,
        adapter_root=adapter_root,
        weights_path=ckpt,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )

    adapted_net, cand_device = load_damage_model(
        weights_path=ckpt,
        adapter_path=train_res["adapter_path"],
        use_cache=False,
    )
    cand_fb_metrics = compute_pixel_metrics(adapted_net, [(x_full, y_all_bg)], device=cand_device)
    cand_hold_metrics = compute_pixel_metrics(adapted_net, [(x_full, y_hold)], device=cand_device)
    cand_old_metrics = compute_pixel_metrics(adapted_net, [(x_full, y_old_bg)], device=cand_device)

    gate_report = evaluate_safety_gate(
        baseline_metrics={
            "per_class_f1": base_fb_metrics["per_class_f1"],
            "background_f1": base_fb_metrics.get("background_f1"),
            "old_disaster_macro_f1": base_old_metrics["macro_f1"],
            "ece": base_fb_metrics["ece"],
        },
        candidate_metrics={
            "per_class_f1": cand_fb_metrics["per_class_f1"],
            "background_f1": cand_fb_metrics.get("background_f1"),
            "old_disaster_macro_f1": cand_old_metrics["macro_f1"],
            "geographic_holdout_f1": cand_hold_metrics["macro_f1"],
            "geographic_holdout_baseline_f1": base_hold_metrics["macro_f1"],
            "train_building_ids": train_ids,
            "holdout_building_ids": holdout_ids,
            "ece": cand_fb_metrics["ece"],
            "tile_latency_ms": cand_fb_metrics["tile_latency_ms"],
        },
    )

    training_summary = {
        "imagery_source": imagery_source,
        "verified_buildings_used": len(sorted_ver),
        "train_buildings": len(train_ids),
        "holdout_buildings": len(holdout_ids),
        "epochs": train_res["epochs"],
        "final_loss": train_res["final_loss"],
        "duration_sec": train_res["duration_sec"],
        "trainable_param_count": train_res["trainable_param_count"],
        "baseline_macro_f1": base_fb_metrics["macro_f1"],
        "candidate_macro_f1": cand_fb_metrics["macro_f1"],
        "baseline_destroyed_f1": base_fb_metrics["destroyed_f1"],
        "candidate_destroyed_f1": cand_fb_metrics["destroyed_f1"],
        "baseline_holdout_f1": base_hold_metrics["macro_f1"],
        "candidate_holdout_f1": cand_hold_metrics["macro_f1"],
        "baseline_ece": base_fb_metrics["ece"],
        "candidate_ece": cand_fb_metrics["ece"],
        "tile_latency_ms": cand_fb_metrics["tile_latency_ms"],
        "adapter_path": train_res["adapter_path"],
    }
    gate_report["training_summary"] = training_summary

    stats = get_feedback_stats(db_path=db_path)
    drift_dist = compute_drift_distance(stats, db_path=db_path)
    promo_res = promote_or_stage_adapter(
        model_version="ResNet18-UNet-xBD-v1.0",
        adapter_version=train_res["adapter_version"],
        disaster_id=disaster_id,
        weights_path=train_res["adapter_path"],
        gate_report=gate_report,
        promote=promote,
        tier=tier,
        trigger_reason=trigger_reason,
        drift_distance=drift_dist,
        db_path=db_path,
    )
    promo_res["training_summary"] = training_summary
    return promo_res


