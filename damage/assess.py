"""Portion 1 — Vision Damage Assessment Plug-In for Portion 5 Backbone."""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException, status
from shapely import wkt
from shapely.geometry import Polygon

from db import (
    DEFAULT_DB_PATH,
    clear_stale_assessed_buildings,
    clear_stale_satellite_incidents,
    get_active_model_version,
    get_buildings,
    get_feedback_logs,
    get_incidents,
    get_road_edges,
    insert_building,
    insert_incident,
    record_model_transition,
)
from damage.dataset import (
    HOUSTON_EAST_BOUNDS,
    decode_image_input,
    extract_building_masks,
    pixel_polygon_to_wgs84,
    prepare_paired_input,
    rasterize,
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
from damage.model import (
    DEFAULT_WEIGHTS_PATH,
    NUM_CLASSES,
    AdaptedSegmentationModel,
    load_damage_model,
    model_class_to_canonical,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT_DIR / "data").resolve()
LOW_CONFIDENCE_THRESHOLD = 0.65
MODEL_VERSION_LABEL = "ResNet18-UNet-xBD-v1.0"

_TASK_LOCK = threading.Lock()
_TASK_STORE: dict[str, dict[str, Any]] = {}


def vote_building(probs: np.ndarray, mask: np.ndarray) -> tuple[int, float] | None:
    """Confidence-weighted damage class vote across building pixels (classes 1..4)."""
    pixels = probs[:, mask]
    if pixels.shape[1] == 0:
        return None
    building_probs = pixels[1:5]
    winners = building_probs.argmax(axis=0)
    best_scores = building_probs.max(axis=0)
    votes = np.zeros(NUM_CLASSES - 1, dtype=np.float64)
    for k in range(NUM_CLASSES - 1):
        votes[k] = float(best_scores[winners == k].sum())
    cls_idx = int(votes.argmax()) + 1
    confidence = float(pixels[cls_idx].mean())
    return cls_idx, max(0.0, min(1.0, confidence))


def _is_path_like(text: str) -> bool:
    return ("/" in text) or ("\\" in text) or (".." in text) or (len(text) > 1 and text[1] == ":")


def _resolve_ref_inside_data(ref: str) -> Path | None:
    candidate = Path(ref)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif len(candidate.parts) > 0 and candidate.parts[0].lower() == "data":
        resolved = (ROOT_DIR / candidate).resolve()
    else:
        resolved = (DATA_DIR / candidate).resolve()
    if not resolved.is_relative_to(DATA_DIR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tile reference escapes allowed data directory: {ref}",
        )
    return resolved if resolved.is_file() else None


def _resolve_imagery_pair(
    pre_disaster_image: str | None,
    post_disaster_image: str | None,
    pre_tile_ref: str | None,
    post_tile_ref: str | None,
) -> tuple[str, str] | None:
    """Resolve real pre/post imagery sources or return None for metadata-only pending requests."""
    if pre_disaster_image or post_disaster_image:
        if not (pre_disaster_image and post_disaster_image):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Both pre_disaster_image and post_disaster_image are required when uploading imagery.",
            )
        return pre_disaster_image, post_disaster_image

    if pre_tile_ref or post_tile_ref:
        resolved_paths: list[str] = []
        for ref in (pre_tile_ref, post_tile_ref):
            if not ref:
                continue
            found = _resolve_ref_inside_data(ref)
            if found is not None:
                resolved_paths.append(str(found))
        if len(resolved_paths) == 2:
            return resolved_paths[0], resolved_paths[1]

    return None


def _wgs84_to_pixel_coords(
    ring: list[list[float]],
    img_w: int,
    img_h: int,
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_span = max(1e-9, max_lon - min_lon)
    lat_span = max(1e-9, max_lat - min_lat)
    pts: list[tuple[float, float]] = []
    for lon, lat in ring:
        px = ((float(lon) - min_lon) / lon_span) * float(img_w)
        py = ((max_lat - float(lat)) / lat_span) * float(img_h)
        pts.append((max(0.0, min(float(img_w - 1), px)), max(0.0, min(float(img_h - 1), py))))
    return pts


def _masks_from_supplied_footprints(
    footprints: list[dict[str, Any]],
    img_h: int,
    img_w: int,
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx, fp in enumerate(footprints, start=1):
        uid = str(fp.get("uid") or fp.get("building_id") or f"SUP-{idx:04d}")
        coords_px: list[tuple[float, float]] = []
        poly_obj: Polygon | None = None
        supplied_geojson: dict[str, Any] | None = None

        if fp.get("polygon_px"):
            coords_px = [(float(p[0]), float(p[1])) for p in fp["polygon_px"]]
            poly_obj = Polygon(coords_px)
        elif fp.get("wkt"):
            poly_obj = wkt.loads(fp["wkt"])
            if poly_obj.geom_type == "Polygon":
                coords_px = [(float(x), float(y)) for x, y in poly_obj.exterior.coords]
        elif isinstance(fp.get("geometry"), dict) and fp["geometry"].get("coordinates"):
            supplied_geojson = fp["geometry"]
            ring = supplied_geojson["coordinates"][0]
            coords_px = _wgs84_to_pixel_coords(ring, img_w, img_h, bounds)
            poly_obj = Polygon(coords_px)

        if poly_obj is None or poly_obj.is_empty or len(coords_px) < 3:
            continue
        mask = rasterize(poly_obj, (img_h, img_w))
        if not np.any(mask):
            continue
        candidates.append(
            {
                "uid": uid,
                "mask": mask,
                "coords_px": coords_px,
                "supplied_geojson": supplied_geojson,
            }
        )
    return candidates


def _save_active_imagery_snapshot(
    pair: tuple[str, str],
    snap_dir: Path,
    bounds: tuple[float, float, float, float],
    sector_id: str,
    pair_digest: str,
) -> None:
    """Persist normalized active T0/T1 PNGs and bbox metadata for Portion 4 Tier A adaptation."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    pre_img, _ = decode_image_input(pair[0], allowed_roots=(DATA_DIR,))
    post_img, _ = decode_image_input(pair[1], allowed_roots=(DATA_DIR,))
    if post_img.size != pre_img.size:
        post_img = post_img.resize(pre_img.size)
    pre_path = snap_dir / "latest_pre.png"
    post_path = snap_dir / "latest_post.png"
    pre_img.save(pre_path, format="PNG")
    post_img.save(post_path, format="PNG")
    meta = {
        "sector_id": sector_id,
        "pair_digest": pair_digest,
        "bounds": [float(v) for v in bounds],
        "width": pre_img.size[0],
        "height": pre_img.size[1],
        "pre_path": str(pre_path),
        "post_path": str(post_path),
    }
    (snap_dir / "latest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _record_task_state(task_id: str, payload: dict[str, Any]) -> None:
    with _TASK_LOCK:
        _TASK_STORE[task_id] = dict(payload)


def run_vision_assessment(
    db_path: str = DEFAULT_DB_PATH,
    sector_id: str = "houston-east",
    task_id: str | None = None,
    pre_disaster_image: str | None = None,
    post_disaster_image: str | None = None,
    pre_tile_ref: str | None = None,
    post_tile_ref: str | None = None,
    bounds: list[float] | tuple[float, float, float, float] | None = None,
    footprints: list[dict[str, Any]] | None = None,
    weights_path: str | Path | None = None,
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run pretrained ResNet-18 U-Net inference + Grad-CAM + encoder embedding export."""
    active_task_id = task_id or f"TASK_ASSESS_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    existing = get_buildings(db_path=db_path)
    existing_low = sum(1 for b in existing if b.get("status_flag") == "LOW_CONFIDENCE")

    pair = _resolve_imagery_pair(pre_disaster_image, post_disaster_image, pre_tile_ref, post_tile_ref)
    if pair is None:
        state = {
            "status": "pending",
            "progress_pct": 0.0,
            "buildings_detected": len(existing),
            "low_confidence_count": existing_low,
            "duration_sec": 0.0,
            "message": (
                f"Assessment {active_task_id} queued for sector {sector_id}; "
                "upload pre/post satellite tiles to run pretrained model inference."
            ),
        }
        _record_task_state(active_task_id, state)
        return state

    t_start = time.perf_counter()
    try:
        tensor, (orig_h, orig_w), geotiff_bounds = prepare_paired_input(
            pair[0], pair[1], multiple=32, allowed_roots=(DATA_DIR,)
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid imagery input: {exc}") from exc

    active_bounds: tuple[float, float, float, float] = (
        tuple(float(v) for v in bounds)  # type: ignore[assignment]
        if bounds and len(bounds) == 4
        else (geotiff_bounds or HOUSTON_EAST_BOUNDS)
    )

    active_reg = get_active_model_version(db_path=db_path)
    active_adapter_version = "none"
    adapter_weights: Path | None = None
    if active_reg and active_reg.get("adapter_version") not in (None, "", "none"):
        candidate_path = Path(str(active_reg.get("weights_path") or ""))
        if not candidate_path.is_absolute():
            candidate_path = ROOT_DIR / candidate_path
        if candidate_path.is_file():
            adapter_weights = candidate_path
            active_adapter_version = str(active_reg["adapter_version"])

    net, device = load_damage_model(weights_path=weights_path or DEFAULT_WEIGHTS_PATH, adapter_path=adapter_weights)
    if adapter_weights is not None and not isinstance(net, AdaptedSegmentationModel):
        active_adapter_version = "none"
    pair_digest = hashlib.sha256(
        struct.pack("!ii", orig_h, orig_w) + tensor.detach().cpu().numpy().tobytes()
    ).hexdigest()[:8].upper()

    # Preserve authoritative responder label corrections across identical satellite captures
    responder_corrected: dict[str, str] = {}
    for fb in get_feedback_logs(limit=500, db_path=db_path):
        tid = str(fb.get("target_id") or "")
        if tid and tid not in responder_corrected and str(fb.get("responder_action") or "").lower() == "correct":
            corr_lbl = str(fb.get("corrected_label") or "").strip().lower()
            if corr_lbl:
                responder_corrected[tid] = corr_lbl

    enc_hook = FeatureHook(net.encoder)
    dec_hook = FeatureHook(net.decoder)
    assessed_records: list[dict[str, Any]] = []
    try:
        with torch.enable_grad():
            logits_padded = net(tensor.to(device))
            enc_feats = enc_hook.features
            dec_feat_full = dec_hook.features
            if enc_feats is None or dec_feat_full is None:
                raise RuntimeError("Model feature hooks failed to capture encoder/decoder activations")

            probs_padded = torch.softmax(logits_padded.detach(), dim=1)[0].cpu().numpy()
            probs = unpad_spatial(probs_padded, orig_h, orig_w)
            logits_unpad = logits_padded[0, :, :orig_h, :orig_w]

            if footprints:
                candidates = _masks_from_supplied_footprints(footprints, orig_h, orig_w, active_bounds)
            else:
                candidates = extract_building_masks(probs, fg_threshold=0.35, min_area_px=24, max_buildings=80)

            tile_id = f"{sector_id}_{pair_digest}"
            dest_exports = Path(export_dir) if export_dir else DEFAULT_EXPORT_DIR
            embedding_ref, _ = save_tile_embedding(tile_id, enc_feats[-1], out_dir=dest_exports)
            is_live_db = Path(db_path).resolve() == Path(DEFAULT_DB_PATH).resolve()
            snap_dir = (DATA_DIR / "uploads") if (export_dir is None and is_live_db) else (dest_exports / "_active_pair")
            _save_active_imagery_snapshot(pair, snap_dir, active_bounds, sector_id, pair_digest)

            now_utc = datetime.now(timezone.utc).isoformat()
            if not footprints:
                clear_stale_assessed_buildings(
                    current_prefix=f"BLD-P1-{pair_digest}-",
                    exclude_ids=set(responder_corrected.keys()),
                    db_path=db_path,
                )
                clear_stale_satellite_incidents(
                    current_prefix=f"INC-SAT-{pair_digest}-",
                    db_path=db_path,
                )
            assessed_count = 0
            low_conf_count = 0

            for idx, cand in enumerate(candidates):
                mask = cand["mask"]
                vote = vote_building(probs, mask)
                if vote is None:
                    continue
                cls_idx, conf = vote
                dmg_label = model_class_to_canonical(cls_idx)
                conf_rounded = round(conf, 4)
                status_flag = "CALIBRATED" if conf_rounded >= LOW_CONFIDENCE_THRESHOLD else "LOW_CONFIDENCE"

                bld_id = str(cand.get("uid") or f"BLD-P1-{pair_digest}-{idx + 1:03d}")
                if bld_id in responder_corrected:
                    dmg_label = responder_corrected[bld_id]
                    conf_rounded = 1.0
                    status_flag = "CALIBRATED"
                if status_flag == "LOW_CONFIDENCE":
                    low_conf_count += 1

                keep_graph = idx < (len(candidates) - 1)
                acts_crop, grad_crop, bbox = building_gradcam(
                    dec_feat_full, logits_unpad, mask, cls_idx, retain_graph=keep_graph
                )
                save_building_exports(tile_id, bld_id, acts_crop, grad_crop, out_dir=dest_exports)
                gradcam_regions = gradcam_crop_to_wgs84_regions(
                    acts_crop,
                    grad_crop,
                    bbox,
                    img_shape=(orig_h, orig_w),
                    bounds=active_bounds,
                    top_k=3,
                )

                if cand.get("supplied_geojson"):
                    geom_geojson = cand["supplied_geojson"]
                    ring = geom_geojson["coordinates"][0]
                    cent_lon = round(sum(float(p[0]) for p in ring[:-1]) / max(1, len(ring) - 1), 6)
                    cent_lat = round(sum(float(p[1]) for p in ring[:-1]) / max(1, len(ring) - 1), 6)
                else:
                    geom_geojson, cent_lat, cent_lon = pixel_polygon_to_wgs84(
                        cand["coords_px"], img_width=orig_w, img_height=orig_h, bounds=active_bounds
                    )

                rec = {
                    "building_id": bld_id,
                    "geometry": geom_geojson,
                    "centroid_lat": cent_lat,
                    "centroid_lon": cent_lon,
                    "damage_class": dmg_label,
                    "class_label": dmg_label,
                    "confidence": conf_rounded,
                    "max_prob": conf_rounded,
                    "status_flag": status_flag,
                    "gradcam_attribution": gradcam_regions,
                    "shap_top3": [],
                    "embedding_ref": embedding_ref,
                    "last_updated": now_utc,
                    "data_source": "resnet18_unet_pretrained",
                }
                insert_building(rec, db_path=db_path)
                assessed_records.append(rec)
                assessed_count += 1
    finally:
        enc_hook.remove()
        dec_hook.remove()

    road_changed = False
    incident_changed = False
    if get_road_edges(db_path=db_path):
        severity_rank = {"destroyed": (3, "critical", "rescue"), "major": (2, "high", "medical"), "minor": (1, "medium", "transport")}
        damaged_sorted = sorted(
            [r for r in assessed_records if r["damage_class"] in severity_rank],
            key=lambda r: (-severity_rank[r["damage_class"]][0], -float(r["confidence"]), r["building_id"]),
        )
        existing_incs = {i["incident_id"]: i for i in get_incidents(db_path=db_path)}
        for k, bld in enumerate(damaged_sorted[:2], start=1):
            inc_id = f"INC-SAT-{pair_digest}-{k:02d}"
            _, sev_str, req_cap = severity_rank[bld["damage_class"]]
            prev = existing_incs.get(inc_id)
            if (
                prev is not None
                and prev.get("severity") == sev_str
                and prev.get("required_capability") == req_cap
                and round(float(prev["location"]["lat"]), 5) == round(float(bld["centroid_lat"]), 5)
                and round(float(prev["location"]["lon"]), 5) == round(float(bld["centroid_lon"]), 5)
            ):
                continue
            insert_incident(
                {
                    "incident_id": inc_id,
                    "location": {"lat": bld["centroid_lat"], "lon": bld["centroid_lon"]},
                    "severity": sev_str,
                    "required_capability": req_cap,
                    "time_reported": prev["time_reported"] if prev else now_utc,
                    "status": prev["status"] if prev else "open",
                },
                db_path=db_path,
            )
            incident_changed = True

        try:
            from routing.graph import apply_satellite_damage_to_roads

            _, road_changed = apply_satellite_damage_to_roads(db_path=db_path, assessed_buildings=assessed_records)
        except ImportError:
            road_changed = False

    elapsed = round(time.perf_counter() - t_start, 3)
    record_model_transition(
        model_version=MODEL_VERSION_LABEL,
        adapter_version=active_adapter_version,
        action="MODEL_INFERENCE_COMPLETED",
        gate_checks={
            "task_id": active_task_id,
            "pair_digest": pair_digest,
            "sector_id": sector_id,
            "buildings_detected": assessed_count,
            "low_confidence_count": low_conf_count,
            "road_changed": road_changed,
            "incident_changed": incident_changed,
            "device": device,
            "duration_sec": elapsed,
        },
        actor_role="PORTION1:DAMAGE_ASSESSMENT",
        db_path=db_path,
    )

    state = {
        "status": "completed",
        "progress_pct": 100.0,
        "buildings_detected": assessed_count,
        "low_confidence_count": low_conf_count,
        "pair_digest": pair_digest,
        "road_changed": road_changed,
        "incident_changed": incident_changed,
        "embedding_ref": embedding_ref,
        "duration_sec": elapsed,
        "message": (
            f"Assessed {assessed_count} building(s) for sector {sector_id} "
            f"using {MODEL_VERSION_LABEL} ({elapsed:.2f}s)."
        ),
    }
    _record_task_state(active_task_id, state)
    return state


def get_assessment_status(task_id: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Retrieve live progress, building count, and low-confidence metrics for an assessment task."""
    with _TASK_LOCK:
        cached = _TASK_STORE.get(task_id)
    if cached is not None:
        return dict(cached)

    buildings = get_buildings(db_path=db_path)
    low_conf = sum(1 for b in buildings if b.get("status_flag") == "LOW_CONFIDENCE")
    return {
        "status": "pending",
        "progress_pct": 0.0,
        "buildings_detected": len(buildings),
        "low_confidence_count": low_conf,
    }
