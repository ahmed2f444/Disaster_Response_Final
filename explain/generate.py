"""Portion 3 (Yousef) — Unified XAI Explanation Orchestrator (Zero-SQL Plug-In Hooks).

Implements all Portion 3 interfaces expected by Portion 5 (`main.py`, `docs/api.md`)
and the Unified System Document (Sections 3.4, 4.3, 5, 6, 7, 9 & Stages 3, 6, 7):
- `generate_narrative(plan_or_diff, use_llm_stretch=False)` -> Endpoint 13 (What-If)
- `explain_route(route=..., route_id=..., db_path=...)` -> Endpoint 14 & Stage 6 async
- `explain_building(building=..., building_id=..., db_path=...)` -> Endpoint 15 & Stage 3
- `schedule_async_route_explanation(route, db_path=...)` -> Stage 6 background worker
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db import DEFAULT_DB_PATH, get_building, get_route, insert_explanation
from explain.gradcam import DEFAULT_EXPORT_DIR, compute_building_gradcam
from explain.llm import generate_building_explanation, generate_explanation, is_llm_enabled
from explain.shap_explainer import compute_shap_attributions
from explain.validate import coerce_shap_dict, template_fallback, validate_llm_output

_LOGGER = logging.getLogger(__name__)


def _format_top_shap_clause(shap_items: list[dict[str, Any]], top_k: int = 3) -> str:
    ranked = sorted(
        (item for item in shap_items if isinstance(item, dict) and "feature" in item),
        key=lambda item: abs(float(item.get("value", item.get("impact", 0.0)))),
        reverse=True,
    )[:top_k]
    if not ranked:
        return "routing factors"
    return ", ".join(
        f"{item['feature']} ({float(item.get('value', 0.0)):+.2f})" for item in ranked
    )


def _deterministic_route_brief(
    route: dict[str, Any],
    shap_items: list[dict[str, Any]],
) -> str:
    """Build a crisp, GTF-grounded deterministic route narrative without LLM overhead."""
    route_id = str(route.get("route_id") or route.get("plan_id") or "PLAN")
    eta = float(route.get("eta_seconds") or route.get("total_eta_sec") or 0.0)
    altered = bool(route.get("mask_altered", False))
    reason = str(route.get("mask_reason") or "").strip()

    edge_ids: list[str] = []
    for sub in route.get("routes", []) or []:
        if isinstance(sub, dict):
            edge_ids.extend(str(e) for e in sub.get("edge_ids", []))
    corridor_str = ", ".join(list(dict.fromkeys(edge_ids))[:6])
    corridor_clause = f" via [{corridor_str}]" if corridor_str else ""

    top_factors = _format_top_shap_clause(shap_items, top_k=3)
    if altered:
        mask_clause = (
            f"Safety mask altered the PPO proposal ({reason})."
            if reason
            else "Safety mask altered the PPO proposal to bypass blocked or restricted edges."
        )
    else:
        mask_clause = "Safety mask did not alter the PPO proposal; corridor verified passable."

    return (
        f"[P3 SHAP] Route {route_id}{corridor_clause} has ETA {eta:.1f}s. "
        f"Top PPO factors: {top_factors}. {mask_clause}"
    )


def _deterministic_building_brief(
    building: dict[str, Any],
    gradcam_info: dict[str, Any],
) -> str:
    """Build a crisp, GTF-grounded deterministic Grad-CAM building assessment briefing."""
    bld_id = str(building.get("building_id") or "BLD")
    dmg_lbl = str(building.get("damage_class", building.get("class_label", "minor"))).lower()
    conf = float(building.get("confidence", 0.85))
    hotspot = str(gradcam_info.get("hotspot_summary") or "").strip()
    focus_clause = f" {hotspot}" if hotspot else ""
    return (
        f"[P3 GRAD-CAM] Building {bld_id} is classified as {dmg_lbl.upper()} "
        f"(calibrated confidence: {conf * 100:.1f}%).{focus_clause}"
    )


def generate_narrative(
    plan_or_diff: dict[str, Any],
    use_llm_stretch: bool = False,
) -> tuple[str, str]:
    """Generate GTF-constrained counterfactual What-If narrative (`Endpoint 13` hook)."""
    plan_id = str(plan_or_diff.get("plan_id") or "WHATIF")
    whatif_eta = float(plan_or_diff.get("whatif_eta_sec") or plan_or_diff.get("eta_seconds") or 0.0)
    baseline_eta = float(plan_or_diff.get("baseline_eta_sec") or plan_or_diff.get("total_eta_sec") or whatif_eta)
    delta_sec = float(plan_or_diff.get("eta_delta_sec", whatif_eta - baseline_eta))
    flood_delta = float(plan_or_diff.get("flood_delta_m", plan_or_diff.get("flood_delta", 0.0)))
    newly_blocked = [str(e) for e in (plan_or_diff.get("newly_blocked_edges") or [])]
    mask_altered = bool(plan_or_diff.get("mask_altered", bool(newly_blocked)))

    shap_items = compute_shap_attributions(plan_or_diff, mask_altered=mask_altered)
    top_factors = _format_top_shap_clause(shap_items, top_k=2)

    if use_llm_stretch and is_llm_enabled():
        shap_dict = coerce_shap_dict(shap_items)
        mask_reason = (
            f"Newly blocked edges ({', '.join(newly_blocked[:4])}) under flood delta {flood_delta:+.2f}m"
            if newly_blocked
            else f"Flood delta {flood_delta:+.2f}m"
        )
        raw_llm = generate_explanation(
            shap_values=shap_dict,
            mask_altered=mask_altered,
            mask_reason=mask_reason,
            uncertainty=0.08,
            route={**plan_or_diff, "eta_seconds": whatif_eta, "mask_altered": mask_altered},
        )
        if raw_llm:
            ok, _ = validate_llm_output(
                raw_llm,
                shap_values=shap_dict,
                mask_reason=mask_reason,
                uncertainty=0.08,
                route={**plan_or_diff, "eta_seconds": whatif_eta, "mask_altered": mask_altered},
                llm_used=True,
            )
            if ok:
                return raw_llm, "LLM_ADVISORY"

    blocked_clause = (
        f" Newly blocked: {', '.join(newly_blocked[:4])}."
        if newly_blocked
        else " No additional corridors blocked."
    )
    mask_clause = (
        "Safety mask altered the detour."
        if mask_altered
        else "Safety mask did not alter the primary corridor."
    )
    text = (
        f"[P3 WHAT-IF] Dry-run {plan_id} under flood delta {flood_delta:+.2f}m yields "
        f"ETA {whatif_eta:.1f}s (delta {delta_sec:+.1f}s vs baseline {baseline_eta:.1f}s)."
        f"{blocked_clause} Key SHAP drivers: {top_factors}. {mask_clause}"
    )
    return text, "JINJA2_TEMPLATE"


def explain_route(
    route: dict[str, Any] | None = None,
    route_id: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    *,
    use_llm: bool | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Compute KernelSHAP attributions and GTF-validated narrative for a Route (`Endpoint 14`)."""
    resolved_route = dict(route) if isinstance(route, dict) else None
    if resolved_route is None and route_id:
        resolved_route = get_route(route_id, db_path=db_path)
    if resolved_route is None:
        raise ValueError(f"Route not found for explanation: {route_id}")

    target_id = str(route_id or resolved_route.get("route_id") or resolved_route.get("plan_id") or "PLAN")
    uncertainty = float(resolved_route.get("uncertainty") if resolved_route.get("uncertainty") is not None else 0.08)
    mask_altered = bool(resolved_route.get("mask_altered", False))
    mask_reason = str(resolved_route.get("mask_reason") or "Safety mask active")

    # Compute 7-feature KernelSHAP on PPO pre-mask proposal (`ppo_proposed_path`)
    shap_items = compute_shap_attributions(resolved_route, mask_altered=mask_altered)
    shap_dict = coerce_shap_dict(shap_items)
    route_ctx = {**resolved_route, "route_id": target_id, "shap_values": shap_items}

    should_try_llm = is_llm_enabled() and os.getenv("AEGIS_SYNC_LLM") == "1" if use_llm is None else bool(use_llm)
    llm_used = False
    validation_passed = True

    if should_try_llm:
        raw_llm = generate_explanation(
            shap_values=shap_dict,
            mask_altered=mask_altered,
            mask_reason=mask_reason,
            uncertainty=uncertainty,
            route=route_ctx,
        )
        if raw_llm:
            llm_used = True
            ok, _ = validate_llm_output(
                raw_llm,
                shap_values=shap_dict,
                mask_reason=mask_reason,
                uncertainty=uncertainty,
                route=route_ctx,
                llm_used=True,
            )
            if ok:
                natural_language = raw_llm
            else:
                validation_passed = False
                natural_language = template_fallback(
                    shap_values=shap_dict,
                    mask_altered=mask_altered,
                    mask_reason=mask_reason,
                    uncertainty=uncertainty,
                    route=route_ctx,
                )
        else:
            natural_language = _deterministic_route_brief(route_ctx, shap_items)
    else:
        natural_language = _deterministic_route_brief(route_ctx, shap_items)

    now_utc = datetime.now(timezone.utc).isoformat()
    explanation_id = f"EXP-{target_id}-{uuid.uuid4().hex[:6].upper()}"
    record = {
        "explanation_id": explanation_id,
        "target_type": "route",
        "target_id": target_id,
        "natural_language": natural_language,
        "attribution_method": "shap",
        "shap_values": shap_items,
        "system_uncertainty": round(uncertainty, 4),
        "llm_used": llm_used,
        "validation_passed": validation_passed,
        "mask_altered": mask_altered,
        "mask_reason": mask_reason,
        "timestamp": now_utc,
    }
    if persist:
        insert_explanation(record, db_path=db_path)
    return record


def explain_building(
    building: dict[str, Any] | None = None,
    building_id: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
    *,
    export_dir: str | Path | None = None,
    use_llm: bool | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Compute Grad-CAM spatial attribution and GTF-validated narrative for a Building (`Endpoint 15`)."""
    resolved_bld = dict(building) if isinstance(building, dict) else None
    if resolved_bld is None and building_id:
        resolved_bld = get_building(building_id, db_path=db_path)
    if resolved_bld is None:
        raise ValueError(f"Building not found for explanation: {building_id}")

    target_id = str(building_id or resolved_bld.get("building_id") or "BLD")
    conf = float(resolved_bld.get("confidence", 0.85))
    uncertainty = round(max(0.0, min(1.0, 1.0 - conf)), 4)

    gradcam_info = compute_building_gradcam(resolved_bld, export_dir=export_dir)
    regions = list(gradcam_info.get("regions") or resolved_bld.get("gradcam_attribution") or [])

    should_try_llm = is_llm_enabled() and os.getenv("AEGIS_SYNC_LLM") == "1" if use_llm is None else bool(use_llm)
    llm_used = False
    validation_passed = True

    if should_try_llm:
        raw_llm = generate_building_explanation(building=resolved_bld, gradcam_info=gradcam_info)
        if raw_llm:
            llm_used = True
            ok, _ = validate_llm_output(
                raw_llm,
                uncertainty=uncertainty,
                building=resolved_bld,
                gradcam_info=gradcam_info,
                llm_used=True,
            )
            if ok:
                natural_language = raw_llm
            else:
                validation_passed = False
                natural_language = template_fallback(
                    uncertainty=uncertainty,
                    building=resolved_bld,
                    gradcam_summary=str(gradcam_info.get("hotspot_summary") or ""),
                )
        else:
            natural_language = _deterministic_building_brief(resolved_bld, gradcam_info)
    else:
        natural_language = _deterministic_building_brief(resolved_bld, gradcam_info)

    now_utc = datetime.now(timezone.utc).isoformat()
    explanation_id = f"EXP-{target_id}-{uuid.uuid4().hex[:6].upper()}"
    record = {
        "explanation_id": explanation_id,
        "target_type": "building",
        "target_id": target_id,
        "natural_language": natural_language,
        "attribution_method": "gradcam",
        "gradcam_regions": regions,
        "overlay_path": gradcam_info.get("overlay_path"),
        "hotspot_summary": gradcam_info.get("hotspot_summary"),
        "system_uncertainty": uncertainty,
        "llm_used": llm_used,
        "validation_passed": validation_passed,
        "timestamp": now_utc,
    }
    if persist:
        insert_explanation(record, db_path=db_path)
    return record


def schedule_async_route_explanation(
    route: dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Stage 6: Schedule asynchronous GTF LLM route explanation outside the <500ms reroute loop."""
    if not isinstance(route, dict) or not (route.get("route_id") or route.get("plan_id")):
        return False
    route_snapshot = dict(route)
    route_id = str(route_snapshot.get("route_id") or route_snapshot.get("plan_id"))

    # Persist fast deterministic KernelSHAP explanation immediately unless LLM async worker is enabled
    if not is_llm_enabled():
        explain_route(route=route_snapshot, route_id=route_id, db_path=db_path, use_llm=False, persist=True)
        return True

    def _worker() -> None:
        try:
            explain_route(route=route_snapshot, route_id=route_id, db_path=db_path, use_llm=True, persist=True)
        except Exception as exc:
            _LOGGER.warning("Async route explanation failed for %s: %s", route_id, exc)

    t = threading.Thread(target=_worker, name=f"p3-explain-{route_id}", daemon=True)
    t.start()
    return True
