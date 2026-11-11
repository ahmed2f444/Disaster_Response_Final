"""Portion 5 (Ahmed) — System Integration & Edge Web Architecture.

FastAPI Tactical Command Backbone, Explanation Validation & ETA Gate,
Commander Approval/Rejection Gate, and SHA-256 Merkle Audit Chain.

Exposes the full 18 HTTP system contract URLs connecting all 5 portions:
- Portion 1 (Ammar): Satellite damage assessment & Building endpoints (`damage/`)
- Portion 2 (Negm): MARL routing, hazards, Incident & What-If endpoints (`routing/`)
- Portion 3 (Yousef): Explainable AI (Grad-CAM & SHAP) endpoints (`explain/`)
- Portion 4 (Abdulrahman): Active learning feedback & drift endpoints (`feedback/`)
- Portion 5 (Ahmed): Validation gate, Commander approval/rejection, audit & Web UI
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import re
import socket
import sys
import tempfile
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Path as FPath, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db import (
    DEFAULT_DB_PATH,
    append_audit_event,
    decide_route_with_audit,
    get_active_model_version,
    get_audit_log,
    get_building,
    get_buildings,
    get_explanation_by_target,
    get_feedback_logs,
    get_feedback_stats,
    get_incident,
    get_incidents,
    get_road_edges,
    get_route,
    get_routes,
    init_db,
    insert_explanation,
    insert_incident,
    insert_route,
    log_feedback,
    normalize_damage_class,
    update_building_label,
    update_incident_status,
    verify_audit_chain,
)
from models import (
    AdaptationRollbackRequest,
    AdaptationRunRequest,
    AssessmentStatusResponse,
    AttributionMethodEnum,
    AuditLogEntry,
    AuditLogReportResponse,
    Building,
    BuildingCorrectionRequest,
    BuildingCorrectionResponse,
    BuildingDetailResponse,
    BuildingExplanationResponse,
    BuildingListResponse,
    DispatchComputeRequest,
    DispatchComputeResponse,
    DriftReportResponse,
    FeedbackItem,
    FeedbackSubmitResponse,
    HealthResponse,
    Incident,
    IncidentCreateRequest,
    IncidentListResponse,
    IncidentStatusEnum,
    Route,
    RouteDecisionRequest,
    RouteDecisionResponse,
    RouteExplanationResponse,
    RouteListResponse,
    TacticalStateResponse,
    UploadAssessmentRequest,
    UploadAssessmentResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from seed import seed_sector

MAX_GEOJSON_FEATURES = 10_000
MAX_GEOJSON_BYTES = 100 * 1024  # Unified System Document Section 4.5 (< 100 KB)

DEFAULT_ROUTING_SHAP = [
    {"feature": "hazard_distance", "value": 0.48, "impact": 0.48, "attribution": 0.48},
    {"feature": "travel_time", "value": 0.35, "impact": 0.35, "attribution": 0.35},
    {"feature": "road_capacity", "value": 0.17, "impact": 0.17, "attribution": 0.17},
    {"feature": "closure_state", "value": 0.14, "impact": 0.14, "attribution": 0.14},
    {"feature": "vehicle_restriction", "value": 0.09, "impact": 0.09, "attribution": 0.09},
    {"feature": "incident_severity", "value": 0.07, "impact": 0.07, "attribution": 0.07},
    {"feature": "unit_capability", "value": 0.05, "impact": 0.05, "attribution": 0.05},
]

_ETA_PATTERN = re.compile(
    r"(?:eta\s*(?:is|of|[:=])?\s*)(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
    re.IGNORECASE,
)
_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_EDGE_PATTERN = re.compile(r"\bE-\d+\b")
_HEDGE_PATTERN = re.compile(r"\b(?:maybe|perhaps|possibly|probably|might\s+be|could\s+be|likely)\b", re.IGNORECASE)
_CONFIDENCE_WORD_PATTERN = re.compile(r"\b(?:confidence|certainty|confident)\b", re.IGNORECASE)
_OVERRIDE_PATTERN = re.compile(r"\b(?:override|disregard|ignore\s+(?:the\s+)?ppo)\b", re.IGNORECASE)
_MASK_NO_CHANGE_PATTERN = re.compile(
    r"\b(?:mask\s+did\s+not\s+alter|not\s+altered\s+by\s+(?:the\s+)?mask|unchanged\s+by\s+(?:the\s+)?mask)\b",
    re.IGNORECASE,
)
_MASK_CHANGED_PATTERN = re.compile(
    r"\b(?:mask\s+altered\s+the|diverted\s+by\s+the\s+safety\s+mask)\b",
    re.IGNORECASE,
)

_ASSESSMENT_TASKS_LOCK = threading.Lock()
_ASSESSMENT_TASKS: dict[str, dict[str, Any]] = {}


def _resolve_db_path(db_path: str = DEFAULT_DB_PATH) -> str:
    """Resolve and strictly confine SQLite paths to data/ (plus tempdir during pytest runs)."""
    if not db_path or "\x00" in str(db_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid db_path")
    raw_path = Path(db_path)
    if ".." in raw_path.parts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid db_path: parent traversal segments are not allowed.",
        )
    candidate = raw_path if raw_path.is_absolute() else (ROOT_DIR / raw_path)
    resolved = candidate.resolve()
    if resolved.suffix.lower() not in {".db", ".sqlite"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid db_path: database file must end with .db or .sqlite.",
        )
    allowed_roots = [(ROOT_DIR / "data").resolve()]
    if "pytest" in sys.modules or os.getenv("AEGIS_ALLOW_TEMP_DB") == "1":
        allowed_roots.append(Path(tempfile.gettempdir()).resolve())
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid db_path: path must stay inside data/ directory.",
        )
    return str(resolved)


def _ensure_db_ready(resolved_db: str, allow_default_seed: bool = False) -> str:
    """Ensure SQLite DB exists and is initialized, returning HTTP 404 for unknown custom paths."""
    db_file = Path(resolved_db)
    default_resolved = (ROOT_DIR / DEFAULT_DB_PATH).resolve()
    if allow_default_seed and db_file.resolve() == default_resolved and not db_file.exists():
        seed_sector(db_path=resolved_db)
        return resolved_db
    if not db_file.exists() or db_file.stat().st_size == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Database '{db_file.name}' is not initialized. Seed the sector or POST /api/reset-sector.",
        )
    return resolved_db


def _load_plugin_callable(module_name: str, attr_name: str) -> Any | None:
    """Safely load an optional teammate hook without failing when stubs are 0 B."""
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    fn = getattr(mod, attr_name, None)
    return fn if callable(fn) else None


def _invoke_plugin(fn: Any, module_label: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a teammate plugin callable with signature filtering and HTTP 502 error boundary."""
    try:
        call_kwargs = dict(kwargs)
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not has_var_kw and call_kwargs:
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            pass
        return fn(*args, **call_kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{module_label} plugin error: {exc}",
        ) from exc


def _compute_drift_distance(
    feedback_stats: dict[str, Any],
    db_path: str | None = None,
) -> float:
    """Delegate drift computation to Portion 4 if plugged in; returns 0.0 when stubbed."""
    plugin_fn = _load_plugin_callable("feedback.drift", "compute_drift_distance")
    if plugin_fn is not None:
        return float(_invoke_plugin(plugin_fn, "Portion 4 drift", feedback_stats, db_path=db_path))
    return 0.0


def _persist_feedback_and_replay(
    item_dict: dict[str, Any],
    resolved_db: str,
    allow_identical_correction: bool = False,
) -> dict[str, Any]:
    """Persist feedback through Portion 4 validation & ingest into dual-buffer replay."""
    fb_mod = None
    try:
        fb_mod = importlib.import_module("feedback.feedback")
    except Exception:
        fb_mod = None

    if fb_mod is not None and callable(getattr(fb_mod, "normalize_and_validate_feedback", None)):
        try:
            validated = fb_mod.normalize_and_validate_feedback(
                item_dict,
                allow_identical_correction=allow_identical_correction,
            )
            item_dict.update(validated)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        rate_warn = False
        if callable(getattr(fb_mod, "check_responder_rate_limit", None)):
            rate_warn, _ = fb_mod.check_responder_rate_limit(
                str(item_dict.get("responder_id") or "FIELD_RESPONDER_01"),
                db_path=resolved_db,
            )
    else:
        rate_warn = False

    active = get_active_model_version(db_path=resolved_db)
    if active and item_dict.get("adapter_version") in (None, "", "none"):
        item_dict["model_version"] = active.get("model_version", item_dict.get("model_version", "ResNet18-UNet-xBD-v1.0"))
        item_dict["adapter_version"] = active.get("adapter_version", "none")

    saved = log_feedback(item_dict, db_path=resolved_db)
    saved["rate_limited_warning"] = rate_warn

    # Ingest verified confirm/correct actions into DualReplayBuffer (Portion 4)
    action_lower = str(saved.get("responder_action") or "").lower()
    if action_lower in {"confirm", "correct"}:
        try:
            replay_mod = importlib.import_module("feedback.replay")
            buf_cls = getattr(replay_mod, "DualReplayBuffer", None)
            if buf_cls is not None:
                lbl = str(item_dict.get("corrected_label") or item_dict.get("prediction") or "").strip()
                if lbl:
                    try:
                        _, canonical_lbl = normalize_damage_class(lbl)
                        bld_row = get_building(str(item_dict["target_id"]), db_path=resolved_db)
                        buf_cls(db_path=resolved_db).add_sample(
                            target_id=str(item_dict["target_id"]),
                            disaster_id=str(item_dict.get("disaster_id") or "HOUSTON_HARVEY_2017"),
                            damage_class=canonical_lbl,
                            uncertainty=float(item_dict.get("uncertainty") or 0.0),
                            priority=1.0 + float(item_dict.get("uncertainty") or 0.0) * 2.0 + float(saved.get("severity_jump") or 0) * 0.5,
                            embedding_ref=str((bld_row or {}).get("embedding_ref") or ""),
                            metadata={"source": action_lower, "responder_id": item_dict.get("responder_id", "")},
                        )
                    except ValueError:
                        pass
        except Exception:
            pass

    return saved


def _enforce_geojson_budget(
    building_features: list[dict[str, Any]],
    edge_features: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enforce Unified System Document Section 4.5 (< 100 KB and < 10,000 features)."""
    prioritized_buildings = sorted(
        building_features,
        key=lambda f: 0 if (f.get("properties") or {}).get("data_source") == "resnet18_unet_pretrained" else 1,
    )
    bounded_edges = list(edge_features[: MAX_GEOJSON_FEATURES - 1])
    max_buildings = max(0, (MAX_GEOJSON_FEATURES - 1) - len(bounded_edges))
    bounded_buildings = list(prioritized_buildings[:max_buildings])

    def _byte_size(b_feats: list[dict[str, Any]], e_feats: list[dict[str, Any]]) -> int:
        payload = {
            "buildings_geojson": {"type": "FeatureCollection", "features": b_feats},
            "roads_geojson": {"type": "FeatureCollection", "features": e_feats},
        }
        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    if _byte_size(bounded_buildings, bounded_edges) < MAX_GEOJSON_BYTES:
        return bounded_buildings, bounded_edges

    # Strip non-essential telemetry strings first if close to the 100 KB ceiling
    stripped = [
        {
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                k: feat["properties"][k]
                for k in ("building_id", "damage_class", "class_label", "confidence", "max_prob", "status_flag", "data_source")
                if k in feat["properties"]
            },
        }
        for feat in bounded_buildings
    ]
    while stripped and _byte_size(stripped, bounded_edges) >= MAX_GEOJSON_BYTES:
        keep_count = max(0, int(len(stripped) * 0.9))
        if keep_count == len(stripped):
            stripped.pop()
        else:
            stripped = stripped[:keep_count]

    while bounded_edges and _byte_size(stripped, bounded_edges) >= MAX_GEOJSON_BYTES:
        keep_count = max(1, int(len(bounded_edges) * 0.9))
        if keep_count == len(bounded_edges):
            bounded_edges.pop()
        else:
            bounded_edges = bounded_edges[:keep_count]

    return stripped, bounded_edges


def _deterministic_route_template(route: dict[str, Any]) -> str:
    route_id = str(route.get("route_id") or route.get("plan_id") or "PLAN")
    eta = float(route.get("eta_seconds") or route.get("total_eta_sec") or 0.0)
    mask_reason = str(route.get("mask_reason") or "Safety mask active")
    return (
        f"[TEMPLATE FALLBACK] Route {route_id} follows safety-filtered corridor "
        f"(ETA {eta:.1f}s). {mask_reason}"
    )


def _validate_route_explanation(
    candidate_text: str,
    route: dict[str, Any],
    llm_used: bool,
    stored_validation_passed: bool | None = None,
) -> tuple[str, bool]:
    """Portion 5 Validation & ETA Gate (Unified System Document Sec. 7, Mismatch 3, Stage 7)."""
    fallback = _deterministic_route_template(route)
    if stored_validation_passed is False:
        return fallback, False

    text = (candidate_text or "").strip()
    if len(text) < 10:
        return fallback, False

    expected_eta = float(route.get("eta_seconds") or route.get("total_eta_sec") or 0.0)
    if expected_eta > 0.0:
        stated_etas = [float(m) for m in _ETA_PATTERN.findall(text)]
        if stated_etas and not all(abs(val - expected_eta) <= 0.5 for val in stated_etas):
            return fallback, False

    # Check cited road edge IDs against known route/sector edges
    known_edges: set[str] = {str(rs.get("edge_id")) for rs in route.get("road_states", []) if rs.get("edge_id")}
    for sub in route.get("routes", []):
        known_edges.update(str(e) for e in sub.get("edge_ids", []))
        known_edges.update(str(e) for e in sub.get("ppo_edge_ids", []))
    if known_edges:
        cited_edges = set(_EDGE_PATTERN.findall(text))
        if cited_edges and not cited_edges.issubset(known_edges):
            return fallback, False

    # Enforce Unified Document Section 7 GTF prohibitions for LLM-authored narratives
    if llm_used:
        if (
            _PCT_PATTERN.search(text)
            or _CONFIDENCE_WORD_PATTERN.search(text)
            or _HEDGE_PATTERN.search(text)
            or _OVERRIDE_PATTERN.search(text)
        ):
            return fallback, False
        if "mask_altered" in route:
            if route["mask_altered"] and _MASK_NO_CHANGE_PATTERN.search(text):
                return fallback, False
            if (not route["mask_altered"]) and _MASK_CHANGED_PATTERN.search(text):
                return fallback, False

    return text, True


def _validate_building_explanation(
    candidate_text: str,
    building: dict[str, Any],
    llm_used: bool,
    stored_validation_passed: bool | None = None,
) -> tuple[str, bool]:
    """Portion 5 Validation Gate for Building Grad-CAM explanations (Sec. 7)."""
    dmg_lbl = str(building.get("damage_class", building.get("class_label", "minor"))).lower()
    conf = float(building.get("confidence", 0.85))
    fallback = (
        f"[TEMPLATE FALLBACK] Structural assessment ({dmg_lbl.upper()}). "
        f"System calibrated confidence: {conf * 100:.1f}%."
    )
    if stored_validation_passed is False:
        return fallback, False

    text = (candidate_text or "").strip()
    if len(text) < 10:
        return fallback, False

    lowered = text.lower()
    label_patterns = {
        "no_damage": r"\bno(?:[\s_]+damage)\b",
        "minor": r"\bminor\b",
        "major": r"\bmajor\b",
        "destroyed": r"\bdestroyed\b",
    }
    other_labels = set(label_patterns) - {dmg_lbl}
    for other in other_labels:
        for match in re.finditer(label_patterns[other], lowered):
            prefix = lowered[max(0, match.start() - 64) : match.start()]
            if not re.search(r"\b(?:not|no|neither|nor|unlike|without|non)\b[^.!?;]*$", prefix):
                return fallback, False

    stated_pcts = [float(m) for m in _PCT_PATTERN.findall(text)]
    if llm_used and stated_pcts:
        return fallback, False
    if stated_pcts:
        expected_pct = conf * 100.0
        if not all(abs(p - expected_pct) <= 2.0 for p in stated_pcts):
            return fallback, False
    if llm_used and (_CONFIDENCE_WORD_PATTERN.search(text) or _HEDGE_PATTERN.search(text)):
        return fallback, False

    return text, True


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Pre-warm Portion 2 routing engine on startup so live reroutes stay < 500 ms."""
    try:
        default_db = Path(_resolve_db_path(DEFAULT_DB_PATH))
        if default_db.exists():
            warm_fn = _load_plugin_callable("routing.generate", "warm_engine")
            if warm_fn is not None:
                warm_fn(db_path=str(default_db))
    except Exception:
        pass
    yield


app = FastAPI(
    title="AEGIS Tactical Disaster Response Command API",
    description="Portion 5 (Ahmed) System Integration Backbone & Edge Web Architecture (18 Contract Endpoints)",
    version="2.2.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://{host}:{port}"
        for port in range(8000, 8020)
        for host in ("127.0.0.1", "localhost")
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

WEB_UI_DIR = ROOT_DIR / "static"
INDEX_HTML_PATH = WEB_UI_DIR / "index.html"
WEB_UI_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_UPLOADS_DIR = (ROOT_DIR / "data" / "uploads").resolve()
app.mount("/static", StaticFiles(directory=str(WEB_UI_DIR)), name="static")


# =====================================================================
# Static & Offline Client Asset Endpoints (Leaflet v1.9.4 served via /static)
# =====================================================================

@app.get("/active-upload/{filename}", include_in_schema=False)
async def serve_active_upload_asset(
    filename: str = FPath(..., description="Active satellite upload artifact filename"),
) -> Response:
    """Serve active pre/post satellite tile and metadata for Leaflet image overlay."""
    allowed = {
        "latest_pre.png": "image/png",
        "latest_post.png": "image/png",
        "latest_meta.json": "application/json",
    }
    media_type = allowed.get(filename)
    if media_type is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    uploads_root = ACTIVE_UPLOADS_DIR.resolve()
    target = (uploads_root / filename).resolve()
    if not target.is_relative_to(uploads_root) or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active satellite upload not available")
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "no-store"})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Empty 204 response for browser favicon request."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/", response_class=HTMLResponse, include_in_schema=False, tags=["Web Interface"])
async def serve_web_interface() -> HTMLResponse:
    """Serve the 6-in-1 Tactical Disaster Response Command Web Interface."""
    if INDEX_HTML_PATH.exists():
        content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    else:
        content = "<h1>AEGIS Command Web Interface - static/index.html pending</h1>"
    return HTMLResponse(content=content)


# =====================================================================
# THE 18 SYSTEM HTTP CONTRACT ENDPOINTS
# =====================================================================

# ---------------------------------------------------------------------
# [Endpoint 1] Health Probe & Offline Status (Portion 5 Ahmed)
# ---------------------------------------------------------------------
@app.get("/healthz", response_model=HealthResponse, tags=["System Health"])
async def health_check() -> HealthResponse:
    """[Endpoint 1] Liveness probe, offline readiness indicator, and contract verification."""
    return HealthResponse(
        status="healthy",
        application="aegis-disaster-response-backbone",
        version="2.2.0",
        offline_ready=True,
        active_endpoints=18,
    )


# ---------------------------------------------------------------------
# [Endpoint 2] Combined Tactical State (Portion 5 Ahmed)
# ---------------------------------------------------------------------
@app.get("/api/state", response_model=TacticalStateResponse, tags=["Tactical State"])
async def api_get_state(db_path: str = Query(DEFAULT_DB_PATH)) -> TacticalStateResponse:
    """[Endpoint 2] Retrieve current sector tactical state with Section 4.5 GeoJSON bounds (<100KB, <10k features)."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path), allow_default_seed=True)

    buildings = get_buildings(db_path=resolved_db)
    incidents = get_incidents(db_path=resolved_db)
    routes = get_routes(include_counterfactuals=False, db_path=resolved_db)
    feedback_stats = get_feedback_stats(db_path=resolved_db)
    chain_ok, audit_count, chain_msg = verify_audit_chain(db_path=resolved_db)

    # Keep per-building GeoJSON lean (<100 KB total); Grad-CAM detail is fetched on demand via Endpoint 6/15.
    building_features = [
        {
            "type": "Feature",
            "geometry": b.get("geometry", {}),
            "properties": {
                "building_id": b["building_id"],
                "damage_class": b["damage_class"],
                "class_label": b["class_label"],
                "confidence": b.get("confidence", 0.9),
                "max_prob": b.get("max_prob", 0.9),
                "status_flag": b.get("status_flag", "CALIBRATED"),
                "embedding_ref": b.get("embedding_ref", ""),
                "data_source": b.get("data_source", "xBD_houston_east"),
            },
        }
        for b in buildings
    ]

    e_rows = get_road_edges(db_path=resolved_db)
    edge_features = []
    road_counts = {"OPEN": 0, "HAZARDOUS": 0, "BLOCKED": 0}
    for e in e_rows:
        status_val = str(e["edge_status"]).upper()
        road_counts[status_val] = road_counts.get(status_val, 0) + 1
        edge_features.append({
            "type": "Feature",
            "geometry": e["geometry"],
            "properties": {
                "edge_id": e["edge_id"],
                "u_node": e["u_node"],
                "v_node": e["v_node"],
                "length_m": e["length_m"],
                "flood_depth_m": e["flood_depth_m"],
                "debris_score": e["debris_score"],
                "closure_state": e.get("closure_state", False),
                "hazard_level": e.get("hazard_level", 0.0),
                "capacity": e.get("capacity", 1.0),
                "vehicle_restriction": e.get("vehicle_restriction", "none"),
                "edge_status": status_val,
            },
        })

    bounded_buildings, bounded_edges = _enforce_geojson_budget(building_features, edge_features)
    drift_distance = _compute_drift_distance(feedback_stats, db_path=resolved_db)
    route_objs = [Route(**r) for r in routes]
    incident_objs = [Incident(**inc) for inc in incidents]

    return TacticalStateResponse(
        sector="xBD Hurricane Harvey (houston-east)",
        data_mode="xBD_GEOJSON_GROUNDED",
        buildings_geojson={"type": "FeatureCollection", "features": bounded_buildings},
        roads_geojson={"type": "FeatureCollection", "features": bounded_edges},
        road_counts=road_counts,
        active_plan=route_objs[0] if route_objs else None,
        active_routes=route_objs,
        incidents=incident_objs,
        drift_distance=drift_distance,
        feedback_count=feedback_stats.get("total_count", 0),
        audit_status={"valid": chain_ok, "entries": audit_count, "message": chain_msg},
    )


# ---------------------------------------------------------------------
# [Endpoint 3] Upload Screen: Trigger Damage Assessment (Portion 1 Ammar)
# ---------------------------------------------------------------------
@app.post("/api/assessment/upload", response_model=UploadAssessmentResponse, tags=["Portion 1: Vision Assessment"])
async def upload_satellite_imagery(
    payload: UploadAssessmentRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> UploadAssessmentResponse:
    """[Endpoint 3] Trigger satellite damage assessment (delegates to damage/assess.py if populated, else returns pending)."""
    resolved_db = _resolve_db_path(db_path)
    if not Path(resolved_db).exists():
        init_db(db_path=resolved_db)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    task_id = f"TASK_ASSESS_{now_ts}_{uuid.uuid4().hex[:6]}"

    assess_fn = _load_plugin_callable("damage.assess", "run_vision_assessment")
    assess_status = "pending"
    progress_pct = 0.0
    duration_sec = 12.5
    assess_msg = (
        f"Bi-temporal satellite imagery queued for sector {payload.sector_id}. "
        "Awaiting Portion 1 vision assessment module (damage/assess.py)."
    )
    if assess_fn is not None:
        assess_res = _invoke_plugin(
            assess_fn,
            "Portion 1 vision assessment",
            task_id=task_id,
            db_path=resolved_db,
            sector_id=payload.sector_id,
            pre_disaster_image=payload.pre_disaster_image,
            post_disaster_image=payload.post_disaster_image,
            pre_tile_ref=payload.pre_tile_ref,
            post_tile_ref=payload.post_tile_ref,
            bounds=payload.bounds,
            footprints=payload.footprints,
        )
        if isinstance(assess_res, dict):
            assess_status = str(assess_res.get("status") or "completed")
            raw_pct = assess_res.get("progress_pct")
            progress_pct = float(raw_pct) if raw_pct is not None else (100.0 if assess_status == "completed" else 0.0)
            if assess_res.get("duration_sec") is not None:
                duration_sec = float(assess_res["duration_sec"])
            assess_msg = str(
                assess_res.get("message")
                or f"Bi-temporal satellite imagery processed for sector {payload.sector_id}."
            )
        else:
            assess_status = "completed"
            progress_pct = 100.0
            assess_msg = f"Bi-temporal satellite imagery processed for sector {payload.sector_id}."

    with _ASSESSMENT_TASKS_LOCK:
        _ASSESSMENT_TASKS[task_id] = {
            "status": assess_status,
            "progress_pct": progress_pct,
            "sector_id": payload.sector_id,
        }

    append_audit_event(
        actor_role="OPERATOR",
        event_type="ASSESSMENT_TRIGGERED",
        payload={"task_id": task_id, "sector_id": payload.sector_id, "status": assess_status},
        db_path=resolved_db,
    )

    if assess_status == "completed" and get_road_edges(db_path=resolved_db):
        compute_fn = _load_plugin_callable("routing.generate", "compute_multi_unit_routes")
        if compute_fn is not None:
            road_changed = bool(assess_res.get("road_changed")) if isinstance(assess_res, dict) else False
            inc_changed = bool(assess_res.get("incident_changed")) if isinstance(assess_res, dict) else False
            existing_routes = get_routes(include_counterfactuals=False, db_path=resolved_db)
            has_p2 = any(str(r.get("route_id", "")).startswith("PLAN-P2-") for r in existing_routes)
            if road_changed or inc_changed or not has_p2:
                sat_plan = _persist_marl_dispatch(
                    resolved_db=resolved_db,
                    compute_fn=compute_fn,
                    actor_role="PORTION2:SAT_DISPATCH",
                    extra_payload={
                        "task_id": task_id,
                        "pair_digest": assess_res.get("pair_digest", "") if isinstance(assess_res, dict) else "",
                    },
                )
                assess_msg += f" MARL route {sat_plan.route_id} recalculated (ETA {sat_plan.eta_seconds:.1f}s)."
            else:
                assess_msg += f" Consistent capture: route {existing_routes[0]['route_id']} unchanged."

    return UploadAssessmentResponse(
        task_id=task_id,
        status=assess_status,
        message=assess_msg,
        estimated_duration_sec=duration_sec,
    )


# ---------------------------------------------------------------------
# [Endpoint 4] Damage Assessment Progress & Status (Portion 1 Ammar)
# ---------------------------------------------------------------------
@app.get("/api/assessment/status/{task_id}", response_model=AssessmentStatusResponse, tags=["Portion 1: Vision Assessment"])
async def get_assessment_status(
    task_id: str = FPath(..., description="Assessment task ID"),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> AssessmentStatusResponse:
    """[Endpoint 4] Query background progress and status of a damage assessment job."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    buildings = get_buildings(db_path=resolved_db)
    low_conf = sum(1 for b in buildings if b.get("status_flag") == "LOW_CONFIDENCE")

    status_fn = _load_plugin_callable("damage.assess", "get_assessment_status")
    if status_fn is not None:
        plugin_state = _invoke_plugin(status_fn, "Portion 1 assessment status", task_id=task_id, db_path=resolved_db)
        if isinstance(plugin_state, dict):
            return AssessmentStatusResponse(
                task_id=task_id,
                status=str(plugin_state.get("status", "pending")),
                progress_pct=float(plugin_state.get("progress_pct", 0.0)),
                buildings_detected=int(plugin_state.get("buildings_detected", len(buildings))),
                low_confidence_count=int(plugin_state.get("low_confidence_count", low_conf)),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    with _ASSESSMENT_TASKS_LOCK:
        task_info = _ASSESSMENT_TASKS.get(task_id)

    task_status = str(task_info["status"]) if task_info else "pending"
    task_progress = float(task_info["progress_pct"]) if task_info else 0.0

    return AssessmentStatusResponse(
        task_id=task_id,
        status=task_status,
        progress_pct=task_progress,
        buildings_detected=len(buildings),
        low_confidence_count=low_conf,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------
# [Endpoint 5] List Assessed Buildings (Portion 1 Ammar)
# ---------------------------------------------------------------------
@app.get("/api/buildings", response_model=BuildingListResponse, tags=["Portion 1: Vision Assessment"])
async def list_buildings(
    damage_class: Optional[str] = Query(
        None,
        description="Filter by canonical damage_class (no_damage|minor|major|destroyed) or tier alias 0..3",
    ),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> BuildingListResponse:
    """[Endpoint 5] List all assessed Building objects with damage classes and Bayesian confidence."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    buildings = get_buildings(db_path=resolved_db)

    filtered = buildings
    if damage_class is not None:
        try:
            _, target_label = normalize_damage_class(damage_class)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid damage_class '{damage_class}': expected no_damage|minor|major|destroyed or 0..3",
            ) from exc
        filtered = [b for b in filtered if b.get("damage_class") == target_label]
    if min_confidence is not None:
        filtered = [b for b in filtered if b.get("confidence", 0.0) >= min_confidence]

    return BuildingListResponse(
        total_count=len(filtered),
        buildings=[Building(**b) for b in filtered],
    )


# ---------------------------------------------------------------------
# [Endpoint 6] Building Detail with Grad-CAM (Portion 1 Ammar)
# ---------------------------------------------------------------------
@app.get("/api/buildings/{building_id}", response_model=BuildingDetailResponse, tags=["Portion 1: Vision Assessment"])
async def get_building_detail(
    building_id: str = FPath(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> BuildingDetailResponse:
    """[Endpoint 6] Retrieve a single Building object with its Grad-CAM spatial attributions and embedding_ref."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    building = get_building(building_id, db_path=resolved_db)
    if not building:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Building {building_id} not found")

    return BuildingDetailResponse(
        building=Building(**building),
        gradcam_overlay_url=f"/api/explanations/building/{building_id}",
    )


# ---------------------------------------------------------------------
# [Endpoint 7] Building Correction Dropdown (Portion 3/4 Responder Action)
# ---------------------------------------------------------------------
@app.post("/api/buildings/{building_id}/correct", response_model=BuildingCorrectionResponse, tags=["Portion 4: Active Learning"])
async def correct_building_classification(
    building_id: str = FPath(...),
    payload: BuildingCorrectionRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> BuildingCorrectionResponse:
    """[Endpoint 7] Building Detail Panel correction dropdown: update label and record feedback log."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    building = get_building(building_id, db_path=resolved_db)
    if not building:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Building {building_id} not found")

    prev_class = str(building.get("damage_class", building.get("class_label", "minor"))).lower()
    target_class_label = payload.corrected_damage_class.value
    prev_conf = float(building.get("confidence", 0.5) if building.get("confidence") is not None else 0.5)
    prev_uncertainty = round(max(0.0, min(1.0, 1.0 - prev_conf)), 4)

    update_building_label(
        building_id=building_id,
        damage_class=target_class_label,
        class_label=target_class_label,
        confidence=1.0,
        db_path=resolved_db,
    )

    fb_res = _persist_feedback_and_replay(
        {
            "target_id": building_id,
            "prediction": prev_class,
            "uncertainty": prev_uncertainty,
            "responder_action": "correct",
            "corrected_label": target_class_label,
            "responder_id": payload.responder_id,
        },
        resolved_db=resolved_db,
        allow_identical_correction=True,
    )

    append_audit_event(
        actor_role=f"RESPONDER:{payload.responder_id}",
        event_type="BUILDING_LABEL_CORRECTED",
        payload={
            "building_id": building_id,
            "previous_class": prev_class,
            "corrected_class": target_class_label,
            "note": payload.note,
        },
        db_path=resolved_db,
    )

    stats = get_feedback_stats(db_path=resolved_db)
    drift_distance = _compute_drift_distance(stats, db_path=resolved_db)

    return BuildingCorrectionResponse(
        building_id=building_id,
        previous_class=prev_class,
        updated_class=target_class_label,
        feedback_logged=True,
        audit_logged=True,
        drift_distance=drift_distance,
        feedback_count=stats.get("total_count", 1),
        severity_jump=int(fb_res.get("severity_jump", 0)),
        requires_second_opinion=bool(fb_res.get("requires_second_opinion", False)),
        reliability_weight=float(fb_res.get("reliability_weight", 1.0)),
        rate_limited_warning=bool(fb_res.get("rate_limited_warning", False)),
    )


# ---------------------------------------------------------------------
# [Endpoint 8] Create / Formulate Incident (Portion 2 Negm)
# ---------------------------------------------------------------------
@app.post("/api/incidents", response_model=Incident, tags=["Portion 2: MARL Routing & Hazards"])
async def create_incident(
    req: IncidentCreateRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> Incident:
    """[Endpoint 8] Formulate a new Incident from damage map or field report (Portion 2 contract)."""
    resolved_db = _resolve_db_path(db_path)
    if not Path(resolved_db).exists():
        init_db(db_path=resolved_db)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    incident_id = req.incident_id or f"INC-{now_ts % 10000:04d}"

    incident_dict = {
        "incident_id": incident_id,
        "location": {"lat": req.lat, "lon": req.lon},
        "severity": req.severity.value,
        "required_capability": req.required_capability.value,
        "time_reported": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }
    insert_incident(incident_dict, db_path=resolved_db)

    append_audit_event(
        actor_role="DISPATCHER",
        event_type="INCIDENT_FORMULATED",
        payload=incident_dict,
        db_path=resolved_db,
    )

    return Incident(**incident_dict)


# ---------------------------------------------------------------------
# [Endpoint 9] List Incidents (Portion 2 Negm)
# ---------------------------------------------------------------------
@app.get("/api/incidents", response_model=IncidentListResponse, tags=["Portion 2: MARL Routing & Hazards"])
async def list_incidents(
    status_filter: Optional[IncidentStatusEnum] = Query(None, alias="status"),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> IncidentListResponse:
    """[Endpoint 9] Query all active, assigned, and resolved Incidents."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    raw = get_incidents(status=status_filter.value if status_filter else None, db_path=resolved_db)
    items = [Incident(**item) for item in raw]
    return IncidentListResponse(total_count=len(items), incidents=items)


# ---------------------------------------------------------------------
# [Endpoint 10] Compute MARL Dispatch & Routes (Portion 2 Negm — HTTP 501 Shell)
# ---------------------------------------------------------------------
def _persist_marl_dispatch(
    resolved_db: str,
    compute_fn: Any,
    actor_role: str = "PORTION2:DISPATCH",
    extra_payload: dict[str, Any] | None = None,
) -> DispatchComputeResponse:
    """Invoke Portion 2 MARL dispatch, validate contract/GTF gate, persist Route, assign incidents, and log audit."""
    plan = _invoke_plugin(compute_fn, "Portion 2 dispatch", db_path=resolved_db)
    if not isinstance(plan, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Portion 2 dispatch plugin returned non-dict payload.",
        )
    try:
        validated_plan = DispatchComputeResponse(**plan)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portion 2 dispatch plugin returned invalid Route payload: {exc}",
        ) from exc

    if validated_plan.explanation:
        safe_txt, gate_ok = _validate_route_explanation(
            candidate_text=validated_plan.explanation,
            route=validated_plan.model_dump(mode="json"),
            llm_used=(validated_plan.narrative_mode != "JINJA2_TEMPLATE"),
        )
        if not gate_ok:
            validated_plan = validated_plan.model_copy(
                update={"explanation": safe_txt, "narrative_text": safe_txt, "narrative_mode": "JINJA2_TEMPLATE"}
            )

    if get_route(validated_plan.route_id, db_path=resolved_db) is None:
        insert_route(validated_plan.model_dump(mode="json"), db_path=resolved_db)
        for sub in validated_plan.routes:
            inc_id = str(sub.get("incident_id") or "")
            if inc_id:
                inc_row = get_incident(inc_id, db_path=resolved_db)
                if inc_row and inc_row.get("status") == "open":
                    update_incident_status(inc_id, "assigned", db_path=resolved_db)
        audit_payload: dict[str, Any] = {
            "route_id": validated_plan.route_id,
            "eta_seconds": validated_plan.eta_seconds,
        }
        if extra_payload:
            audit_payload.update(extra_payload)
        append_audit_event(
            actor_role=actor_role,
            event_type="ROUTE_DISPATCHED",
            payload=audit_payload,
            route_id=validated_plan.route_id,
            db_path=resolved_db,
        )
        async_explain_fn = _load_plugin_callable("explain.generate", "schedule_async_route_explanation")
        if async_explain_fn is not None:
            try:
                async_explain_fn(validated_plan.model_dump(mode="json"), db_path=resolved_db)
            except Exception:
                pass
    return validated_plan


@app.post("/api/routes/dispatch", response_model=DispatchComputeResponse, tags=["Portion 2: MARL Routing & Hazards"])
async def compute_dispatch(
    req: DispatchComputeRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> DispatchComputeResponse:
    """[Endpoint 10] Portion 2 contract shell: delegates to routing/ once implemented, else returns HTTP 501."""
    compute_fn = _load_plugin_callable("routing.generate", "compute_multi_unit_routes")
    if compute_fn is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Portion 2 routing compute (routing/generate.py:compute_multi_unit_routes) is not implemented yet.",
        )

    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    effective_blocked = list(req.blocked_edges or [])
    picked_edge: str | None = None
    if req.arbitrary_blockage:
        pick_fn = _load_plugin_callable("routing.graph", "pick_arbitrary_blockage_edge")
        if pick_fn is not None:
            picked_edge = _invoke_plugin(
                pick_fn,
                "Portion 2 arbitrary blockage",
                db_path=resolved_db,
                exclude=set(effective_blocked) | set(req.opened_edges or []),
                seed_val=req.arbitrary_seed,
            )
            if picked_edge and picked_edge not in effective_blocked:
                effective_blocked.append(picked_edge)

    hazard_fn = _load_plugin_callable("routing.graph", "update_road_hazards")
    if hazard_fn is not None:
        _invoke_plugin(
            hazard_fn,
            "Portion 2 road hazards",
            db_path=resolved_db,
            water_level_delta_m=req.water_level_delta_m,
            blocked_edges=(effective_blocked or None),
            opened_edges=(req.opened_edges or None),
        )
    return _persist_marl_dispatch(
        resolved_db=resolved_db,
        compute_fn=compute_fn,
        actor_role="PORTION2:DISPATCH",
        extra_payload=({"arbitrary_blocked_edge": picked_edge} if picked_edge else None),
    )


# ---------------------------------------------------------------------
# [Endpoint 11] List Routes (Portion 2 Negm / Portion 5 Ahmed)
# ---------------------------------------------------------------------
@app.get("/api/routes", response_model=RouteListResponse, tags=["Portion 2: MARL Routing & Hazards"])
async def list_routes(
    include_counterfactuals: bool = Query(False),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> RouteListResponse:
    """[Endpoint 11] List current Route objects, each containing both ppo_proposed_path and masked_path."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    routes = get_routes(include_counterfactuals=include_counterfactuals, db_path=resolved_db)
    return RouteListResponse(total_routes=len(routes), routes=[Route(**r) for r in routes])


# ---------------------------------------------------------------------
# [Endpoint 12] Route Detail (Portion 2 Negm)
# ---------------------------------------------------------------------
@app.get("/api/routes/{route_id}", response_model=Route, tags=["Portion 2: MARL Routing & Hazards"])
async def get_route_detail(
    route_id: str = FPath(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> Route:
    """[Endpoint 12] Retrieve Route object detailing pre-mask proposal, safety-filtered path, and SHAP features."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    route = get_route(route_id, db_path=resolved_db)
    if not route:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Route {route_id} not found")
    return Route(**route)


# ---------------------------------------------------------------------
# [Endpoint 13] What-If Counterfactual Dry-Run (Portion 2/3 — HTTP 501 Shell)
# ---------------------------------------------------------------------
@app.post("/api/routes/what-if", response_model=WhatIfResponse, tags=["Portion 2: MARL Routing & Hazards"])
@app.post("/api/what-if", response_model=WhatIfResponse, include_in_schema=False)
async def run_what_if_endpoint(
    req: WhatIfRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> WhatIfResponse:
    """[Endpoint 13] Counterfactual shell: delegates to routing/ & explain/ once populated, else returns HTTP 501."""
    counterfactual_fn = _load_plugin_callable("routing.generate", "run_counterfactual")
    if counterfactual_fn is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Portion 2/3 what-if compute (routing/generate.py:run_counterfactual) is not implemented yet.",
        )

    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    res = _invoke_plugin(
        counterfactual_fn,
        "Portion 2 counterfactual",
        blocked_edges=req.blocked_edges,
        opened_edges=req.opened_edges,
        flood_delta=req.flood_delta,
        db_path=resolved_db,
    )
    if not isinstance(res, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Portion 2 counterfactual plugin returned non-dict payload.",
        )

    try:
        flood_val = float(res.get("flood_delta_m", res.get("flood_delta", req.flood_delta)))
        total_eta = float(res.get("baseline_eta_sec", res.get("total_eta_sec", 0.0)))
        whatif_eta = float(res.get("whatif_eta_sec", 0.0))
        delta_val = float(res.get("eta_delta_sec", res.get("delta_sec", 0.0)))
        newly_blocked = list(res.get("newly_blocked_edges", []))
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portion 2 counterfactual plugin returned invalid numeric fields: {exc}",
        ) from exc

    narrative_fn = _load_plugin_callable("explain.generate", "generate_narrative")
    if narrative_fn is not None:
        raw_out = _invoke_plugin(
            narrative_fn,
            "Portion 3 counterfactual narrative",
            res,
            use_llm_stretch=False,
        )
        if not isinstance(raw_out, tuple) or len(raw_out) != 2:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Portion 3 generate_narrative plugin must return (text, mode).",
            )
        raw_narrative, raw_mode = raw_out
        known_road_states = [{"edge_id": e["edge_id"]} for e in get_road_edges(db_path=resolved_db)]
        gate_route = {
            "route_id": res.get("plan_id", "WHATIF"),
            "eta_seconds": whatif_eta,
            "road_states": known_road_states,
            "routes": res.get("routes", []),
            "mask_altered": bool(res.get("mask_altered", bool(req.blocked_edges or newly_blocked))),
            "mask_reason": f"What-If delta {delta_val:+.1f}s under flood surge {flood_val:+.2f}m.",
        }
        narrative_text, gate_ok = _validate_route_explanation(
            candidate_text=str(raw_narrative),
            route=gate_route,
            llm_used=(str(raw_mode) != "JINJA2_TEMPLATE"),
        )
        mode = str(raw_mode) if gate_ok else "JINJA2_TEMPLATE"
    else:
        narrative_text = (
            f"[TEMPLATE FALLBACK] What-If dry run completed in {res.get('compute_ms', 0.0)} ms "
            f"(ETA {whatif_eta:.1f}s)."
        )
        mode = "JINJA2_TEMPLATE"

    try:
        return WhatIfResponse(
            plan_id=res["plan_id"],
            compute_ms=res["compute_ms"],
            flood_delta=flood_val,
            newly_blocked_edges=newly_blocked,
            dynamic_hazards=res.get("dynamic_hazards", {}),
            total_eta_sec=total_eta,
            whatif_eta_sec=whatif_eta,
            delta_sec=delta_val,
            routes=res["routes"],
            narrative_text=narrative_text,
            narrative_mode=mode,
        )
    except (KeyError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portion 2 counterfactual plugin returned invalid WhatIfResponse payload: {exc}",
        ) from exc


# ---------------------------------------------------------------------
# [Endpoint 14] Route Explanation & P5 Validation Gate (Portion 3 & Portion 5)
# ---------------------------------------------------------------------
@app.get("/api/explanations/route/{route_id}", response_model=RouteExplanationResponse, tags=["Portion 3: Explainable AI"])
async def get_route_explanation(
    route_id: str = FPath(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> RouteExplanationResponse:
    """[Endpoint 14] Retrieve route explanation, SHAP values, uncertainty, and run Portion 5 Validation Gate."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    route = get_route(route_id, db_path=resolved_db)
    exp_row = get_explanation_by_target("route", route_id, db_path=resolved_db)
    now_utc = datetime.now(timezone.utc).isoformat()

    if not route:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Route {route_id} not found")

    shap_vals = route.get("shap_values") or DEFAULT_ROUTING_SHAP
    llm_used = bool(exp_row["llm_used"]) if exp_row else False
    plugin_valid: bool | None = None
    if exp_row:
        raw_text = exp_row["natural_language"]
        shap_fn = _load_plugin_callable("explain.shap_explainer", "compute_shap_attributions")
        if shap_fn is not None:
            shap_vals = _invoke_plugin(shap_fn, "Portion 3 route SHAP", route)
    else:
        route_cand = str(route.get("explanation") or "").strip()
        if route_cand:
            _, route_cand_valid = _validate_route_explanation(
                candidate_text=route_cand,
                route=route,
                llm_used=False,
            )
            if not route_cand_valid:
                plugin_valid = False
        explain_fn = _load_plugin_callable("explain.generate", "explain_route")
        if explain_fn is not None:
            plugin_out = _invoke_plugin(
                explain_fn,
                "Portion 3 route explanation",
                route=route,
                route_id=route_id,
                db_path=resolved_db,
                persist=False,
            )
            if isinstance(plugin_out, dict):
                raw_text = str(plugin_out.get("natural_language", ""))
                llm_used = bool(plugin_out.get("llm_used", False))
                if plugin_out.get("shap_values"):
                    shap_vals = plugin_out["shap_values"]
                if "validation_passed" in plugin_out and plugin_valid is not False:
                    plugin_valid = bool(plugin_out["validation_passed"])
            else:
                raw_text = str(plugin_out or "")
        else:
            raw_text = route.get("explanation") or _deterministic_route_template(route)

    stored_valid = bool(exp_row["validation_passed"]) if exp_row else plugin_valid
    validated_text, validation_passed = _validate_route_explanation(
        candidate_text=raw_text,
        route=route,
        llm_used=llm_used,
        stored_validation_passed=stored_valid,
    )
    uncertainty_val = (
        float(exp_row["system_uncertainty"])
        if (exp_row and exp_row.get("system_uncertainty") is not None)
        else float(route.get("uncertainty", 0.08))
    )

    if (not validation_passed) and (exp_row is None or stored_valid):
        insert_explanation(
            {
                "explanation_id": exp_row["explanation_id"] if exp_row else f"EXP-{route_id}",
                "target_type": "route",
                "target_id": route_id,
                "natural_language": validated_text,
                "attribution_method": exp_row.get("attribution_method", "shap") if exp_row else "shap",
                "system_uncertainty": uncertainty_val,
                "llm_used": llm_used,
                "validation_passed": False,
                "timestamp": exp_row.get("timestamp", now_utc) if exp_row else now_utc,
            },
            db_path=resolved_db,
        )

    return RouteExplanationResponse(
        explanation_id=exp_row["explanation_id"] if exp_row else f"EXP-{route_id}",
        target_type="route",
        target_id=route_id,
        route_id=route_id,
        natural_language=validated_text,
        attribution_method=AttributionMethodEnum.SHAP,
        shap_values=shap_vals,
        system_uncertainty=uncertainty_val,
        llm_used=llm_used,
        mask_altered=bool(route.get("mask_altered", False)),
        mask_reason=str(route.get("mask_reason", "Safety mask active")),
        validation_passed=validation_passed,
        timestamp=str(exp_row["timestamp"]) if exp_row else now_utc,
    )


# ---------------------------------------------------------------------
# [Endpoint 15] Building Grad-CAM Explanation & P5 Validation Gate (Portion 3 & 5)
# ---------------------------------------------------------------------
@app.get("/api/explanations/building/{building_id}", response_model=BuildingExplanationResponse, tags=["Portion 3: Explainable AI"])
async def get_building_explanation(
    building_id: str = FPath(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> BuildingExplanationResponse:
    """[Endpoint 15] Retrieve Grad-CAM spatial localization, uncertainty, and run Portion 5 Validation Gate."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    building = get_building(building_id, db_path=resolved_db)
    if not building:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Building {building_id} not found")

    exp_row = get_explanation_by_target("building", building_id, db_path=resolved_db)
    gradcam = building.get("gradcam_attribution") or []

    dmg_lbl = str(building.get("damage_class", building.get("class_label", "minor"))).lower()
    conf = float(building.get("confidence", 0.85))
    llm_used = bool(exp_row["llm_used"]) if exp_row else False
    plugin_valid: bool | None = None
    overlay_available = False

    find_pair_fn = _load_plugin_callable("explain.gradcam", "find_building_export_pair")
    if find_pair_fn is not None:
        try:
            overlay_available = find_pair_fn(building_id) is not None
        except Exception:
            overlay_available = False

    if exp_row:
        raw_text = exp_row["natural_language"]
    else:
        explain_fn = _load_plugin_callable("explain.generate", "explain_building")
        if explain_fn is not None:
            plugin_out = _invoke_plugin(
                explain_fn,
                "Portion 3 building explanation",
                building=building,
                building_id=building_id,
                db_path=resolved_db,
                persist=False,
            )
            if isinstance(plugin_out, dict):
                raw_text = str(plugin_out.get("natural_language", ""))
                llm_used = bool(plugin_out.get("llm_used", False))
                if not gradcam and plugin_out.get("gradcam_regions"):
                    gradcam = plugin_out["gradcam_regions"]
                if plugin_out.get("overlay_path"):
                    overlay_available = True
                if "validation_passed" in plugin_out:
                    plugin_valid = bool(plugin_out["validation_passed"])
            else:
                raw_text = str(plugin_out or "")
        else:
            raw_text = (
                f"[TEMPLATE FALLBACK] Structural assessment ({dmg_lbl.upper()}). "
                f"Calibrated confidence: {conf * 100:.1f}%. (Awaiting Portion 3 Grad-CAM narrative)"
            )

    stored_valid = bool(exp_row["validation_passed"]) if exp_row else plugin_valid
    validated_text, validation_passed = _validate_building_explanation(
        candidate_text=raw_text,
        building=building,
        llm_used=llm_used,
        stored_validation_passed=stored_valid,
    )
    now_utc = datetime.now(timezone.utc).isoformat()
    uncertainty_val = (
        float(exp_row["system_uncertainty"])
        if (exp_row and exp_row.get("system_uncertainty") is not None)
        else round(1.0 - conf, 4)
    )

    if (not validation_passed) and (exp_row is None or stored_valid):
        insert_explanation(
            {
                "explanation_id": exp_row["explanation_id"] if exp_row else f"EXP-{building_id}",
                "target_type": "building",
                "target_id": building_id,
                "natural_language": validated_text,
                "attribution_method": exp_row.get("attribution_method", "gradcam") if exp_row else "gradcam",
                "system_uncertainty": uncertainty_val,
                "llm_used": llm_used,
                "validation_passed": False,
                "timestamp": exp_row.get("timestamp", now_utc) if exp_row else now_utc,
            },
            db_path=resolved_db,
        )

    gradcam_image_url = (
        f"/api/explanations/building/{building_id}/overlay.png"
        if overlay_available
        else None
    )

    return BuildingExplanationResponse(
        explanation_id=exp_row["explanation_id"] if exp_row else f"EXP-{building_id}",
        target_type="building",
        target_id=building_id,
        building_id=building_id,
        damage_class=dmg_lbl,
        confidence=conf,
        gradcam_regions=gradcam,
        natural_language=validated_text,
        explanation_text=validated_text,
        attribution_method=AttributionMethodEnum.GRADCAM,
        system_uncertainty=uncertainty_val,
        llm_used=llm_used,
        validation_passed=validation_passed,
        timestamp=str(exp_row["timestamp"]) if exp_row else str(building.get("last_updated", now_utc)),
        gradcam_image_url=gradcam_image_url,
    )


@app.get("/api/explanations/building/{building_id}/overlay.png", include_in_schema=False, tags=["Portion 3: Explainable AI"])
async def get_building_gradcam_overlay_png(
    building_id: str = FPath(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> FileResponse:
    """Serve rendered Portion 3 Grad-CAM heatmap overlay PNG when `.npy` exports exist."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    building = get_building(building_id, db_path=resolved_db)
    if not building:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Building {building_id} not found")

    gradcam_fn = _load_plugin_callable("explain.gradcam", "compute_building_gradcam")
    if gradcam_fn is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Portion 3 Grad-CAM module not integrated.")

    res = _invoke_plugin(gradcam_fn, "Portion 3 Grad-CAM overlay", building)
    overlay_raw = res.get("overlay_path") if isinstance(res, dict) else None
    if not overlay_raw:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No Grad-CAM overlay exports for {building_id}")

    overlay_path = Path(str(overlay_raw)).resolve()
    allowed_roots = [(ROOT_DIR / "data" / "exports").resolve()]
    try:
        import explain.gradcam as _gx_mod

        allowed_roots.append(Path(_gx_mod.DEFAULT_EXPORT_DIR).resolve())
    except Exception:
        pass
    if not overlay_path.is_file() or not any(overlay_path.is_relative_to(r) for r in allowed_roots):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Grad-CAM overlay file unavailable for {building_id}")
    return FileResponse(path=str(overlay_path), media_type="image/png")


def _execute_route_decision(
    route_id: str,
    new_status: str,
    req: RouteDecisionRequest,
    resolved_db: str,
) -> RouteDecisionResponse:
    """Atomically enforce pending_approval gate lock and write SHA-256 audit entry."""
    outcome, audit_entry = decide_route_with_audit(
        route_id=route_id,
        new_status=new_status,
        actor_role=f"COMMANDER:{req.commander_id}",
        notes=req.notes,
        db_path=resolved_db,
    )
    if outcome == "NOT_FOUND":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Route {route_id} not found")
    if outcome.startswith("CONFLICT:") or audit_entry is None:
        current_status = outcome.split(":", 1)[1] if ":" in outcome else "locked"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Route {route_id} is already '{current_status}' and cannot transition to '{new_status}'.",
        )
    action = "APPROVED" if new_status == "approved" else "REJECTED"
    verb = "approved" if new_status == "approved" else "rejected"
    tail = (
        "Tamper-evident audit digest generated."
        if new_status == "approved"
        else "Operational audit trail updated."
    )
    return RouteDecisionResponse(
        route_id=route_id,
        action=action,
        actor=audit_entry["actor_role"],
        commander_id=req.commander_id,
        audit_hash=audit_entry["curr_hash"],
        prev_hash=audit_entry["prev_hash"],
        timestamp=audit_entry["utc_timestamp"],
        message=f"Route {route_id} {verb} by {req.commander_id}. {tail}",
    )


def _prepare_legacy_route_decision(
    payload: dict[str, Any],
    resolved_db: str,
) -> tuple[str, RouteDecisionRequest]:
    """Resolve default route ID fallback and validate commander decision request."""
    plan_id = payload.get("plan_id")
    if not plan_id:
        routes = get_routes(include_counterfactuals=False, db_path=resolved_db)
        plan_id = routes[0]["route_id"] if routes else "PLAN-BASELINE-01"
    try:
        req = RouteDecisionRequest(
            commander_id=str(payload.get("commander_id", "CMD_LOCAL_01")),
            notes=str(payload.get("notes", "")),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return str(plan_id), req


# ---------------------------------------------------------------------
# [Endpoint 16] Commander Route Approval Gate (Portion 5 Ahmed)
# Formula: SHA-256(prev_hash + route_id + action + actor + timestamp)
# ---------------------------------------------------------------------
@app.post("/api/routes/{route_id}/approve", response_model=RouteDecisionResponse, tags=["Portion 5: Commander Gate"])
async def approve_route_endpoint(
    route_id: str = FPath(..., description="Route plan ID to approve"),
    req: RouteDecisionRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> RouteDecisionResponse:
    """[Endpoint 16] Incident Commander Approval Gate: Lock route to APPROVED and record SHA-256 audit entry."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    return _execute_route_decision(route_id, "approved", req, resolved_db)


@app.post("/api/approve", include_in_schema=False)
async def legacy_approve_plan(
    payload: dict[str, Any] = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """Legacy approval endpoint compatibility."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    plan_id, req = _prepare_legacy_route_decision(payload, resolved_db)
    return _execute_route_decision(plan_id, "approved", req, resolved_db).model_dump()


# ---------------------------------------------------------------------
# [Endpoint 17] Commander Route Rejection Gate (Portion 5 Ahmed)
# Formula: SHA-256(prev_hash + route_id + action + actor + timestamp)
# ---------------------------------------------------------------------
@app.post("/api/routes/{route_id}/reject", response_model=RouteDecisionResponse, tags=["Portion 5: Commander Gate"])
async def reject_route_endpoint(
    route_id: str = FPath(..., description="Route plan ID to reject"),
    req: RouteDecisionRequest = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> RouteDecisionResponse:
    """[Endpoint 17] Incident Commander Rejection Gate: Mark route as REJECTED and record SHA-256 audit entry."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    return _execute_route_decision(route_id, "rejected", req, resolved_db)


@app.post("/api/reject", include_in_schema=False)
async def legacy_reject_plan(
    payload: dict[str, Any] = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """Legacy rejection endpoint compatibility."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    plan_id, req = _prepare_legacy_route_decision(payload, resolved_db)
    return _execute_route_decision(plan_id, "rejected", req, resolved_db).model_dump()


# ---------------------------------------------------------------------
# [Endpoint 18] Responder 1-Tap Feedback Log (Portion 4 Abdulrahman)
# ---------------------------------------------------------------------
@app.post("/api/feedback", response_model=FeedbackSubmitResponse, tags=["Portion 4: Active Learning"])
async def submit_responder_feedback(
    payload: FeedbackItem = Body(...),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> FeedbackSubmitResponse:
    """[Endpoint 18] Responder 1-tap feedback (Confirm / Correct / Skip) queued in SQLite WAL."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    item_dict = payload.model_dump(mode="json")
    res = _persist_feedback_and_replay(item_dict, resolved_db=resolved_db, allow_identical_correction=False)
    stats = get_feedback_stats(db_path=resolved_db)
    drift_distance = _compute_drift_distance(stats, db_path=resolved_db)

    return FeedbackSubmitResponse(
        feedback_id=res["feedback_id"],
        target_id=res["target_id"],
        responder_action=res["responder_action"],
        timestamp=res["timestamp"],
        status="queued_offline_wal",
        drift_distance=drift_distance,
        drift_detected=drift_distance >= 0.15,
        feedback_count=stats.get("total_count", 1),
        severity_jump=int(res.get("severity_jump", 0)),
        requires_second_opinion=bool(res.get("requires_second_opinion", False)),
        reliability_weight=float(res.get("reliability_weight", 1.0)),
        rate_limited_warning=bool(res.get("rate_limited_warning", False)),
    )


# =====================================================================
# Supporting Diagnostic & Operational Endpoints (Excluded from 18-URL Contract Schema)
# =====================================================================

@app.get("/api/feedback", include_in_schema=False, tags=["Portion 4: Active Learning"])
async def list_responder_feedback(
    limit: int = Query(100, ge=1, le=1000),
    target_id: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """List responder feedback queue entries and summary stats from SQLite WAL."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    queue_fn = _load_plugin_callable("feedback.feedback", "get_feedback_queue")
    if queue_fn is not None:
        return _invoke_plugin(queue_fn, "Portion 4 feedback", limit=limit, target_id=target_id, action=action, db_path=resolved_db)
    return {
        "items": get_feedback_logs(limit=limit, target_id=target_id, action=action, db_path=resolved_db),
        "stats": get_feedback_stats(db_path=resolved_db),
    }


@app.get("/api/adaptation/drift", response_model=DriftReportResponse, include_in_schema=False, tags=["Portion 4: Active Learning"])
async def get_drift_status(db_path: str = Query(DEFAULT_DB_PATH)) -> DriftReportResponse:
    """Retrieve statistical distribution drift distance and feedback queue telemetry."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    stats = get_feedback_stats(db_path=resolved_db)
    drift_distance = _compute_drift_distance(stats, db_path=resolved_db)
    drift_plugin_active = _load_plugin_callable("feedback.drift", "compute_drift_distance") is not None
    active = get_active_model_version(db_path=resolved_db)

    report_fn = _load_plugin_callable("feedback.drift", "compute_drift_report")
    report = (
        _invoke_plugin(report_fn, "Portion 4 drift report", feedback_stats=stats, db_path=resolved_db)
        if report_fn is not None
        else {}
    )
    soft_thresh = float(report.get("soft_threshold", 0.08))
    hard_thresh = float(report.get("threshold", 0.15))

    return DriftReportResponse(
        drift_distance=drift_distance,
        soft_threshold=soft_thresh,
        threshold=hard_thresh,
        soft_warning=drift_distance >= soft_thresh,
        drift_detected=drift_distance >= hard_thresh,
        total_feedback_samples=stats.get("total_count", 0),
        pending_retrain_trigger=drift_distance >= hard_thresh or stats.get("total_count", 0) >= 50,
        last_adaptation_tier=(
            "Tier A (Frozen-Backbone Residual Adapter, <30m target)"
            if drift_plugin_active
            else "NONE (Awaiting Portion 4 adaptation)"
        ),
        embedding_signal_available=bool(report.get("embedding_signal_available", False)),
        embedding_reason=str(report.get("embedding_reason", "")),
        operational_metrics=dict(report.get("operational_metrics", {})),
        active_adapter_version=str(active["adapter_version"]) if active else "none",
    )


@app.get("/api/adaptation/status", include_in_schema=False, tags=["Portion 4: Active Learning"])
async def get_adaptation_status(db_path: str = Query(DEFAULT_DB_PATH)) -> dict[str, Any]:
    """Retrieve active model/adapter version, replay buffer stats, drift report, and run history."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    overview_fn = _load_plugin_callable("feedback.status", "get_adaptation_overview")
    if overview_fn is not None:
        return _invoke_plugin(overview_fn, "Portion 4 adaptation status", db_path=resolved_db)
    raise HTTPException(status_code=501, detail="Portion 4 adaptation status module not integrated.")


@app.get("/api/adaptation/queue", include_in_schema=False, tags=["Portion 4: Active Learning"])
async def get_active_learning_queue(
    budget: int = Query(12, ge=1, le=75),
    responder_id: Optional[str] = Query(default=None),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """Select an uncertainty + diversity active learning batch for non-blocking responder review."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    select_fn = _load_plugin_callable("feedback.selection", "select_labeling_candidates")
    if select_fn is not None:
        return _invoke_plugin(
            select_fn,
            "Portion 4 active learning selection",
            budget=budget,
            responder_id=responder_id,
            db_path=resolved_db,
        )
    raise HTTPException(status_code=501, detail="Portion 4 active learning selector not integrated.")


@app.post("/api/adaptation/run", include_in_schema=False, tags=["Portion 4: Active Learning"])
async def run_adaptation_cycle(
    payload: AdaptationRunRequest = Body(default_factory=AdaptationRunRequest),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """Execute fail-closed safety-gated segmentation adaptation and record shadow/promotion status."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    disaster_clean = payload.disaster_id.strip()
    if not disaster_clean:
        raise HTTPException(status_code=422, detail="disaster_id must be non-empty")

    safe_token = _load_plugin_callable("feedback.adapt", "_safe_token")
    safe_disaster = safe_token(disaster_clean) if safe_token else re.sub(r"[^A-Za-z0-9_.-]+", "_", disaster_clean)
    raw_adapter = payload.adapter_version or f"{safe_disaster.lower()}-{payload.tier}-{int(datetime.now(timezone.utc).timestamp())}"
    adapter_ver = safe_token(raw_adapter) if safe_token else raw_adapter

    if payload.train_from_feedback:
        train_fn = _load_plugin_callable("feedback.adapt", "run_feedback_driven_tier_a")
        if train_fn is None:
            raise HTTPException(status_code=501, detail="Portion 4 Tier A training module not integrated.")
        try:
            return await asyncio.to_thread(
                train_fn,
                disaster_id=disaster_clean,
                adapter_version=adapter_ver,
                epochs=payload.epochs,
                learning_rate=payload.learning_rate,
                promote=payload.promote,
                tier=payload.tier,
                trigger_reason="manual_ui_tier_a",
                db_path=resolved_db,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    gate_fn = _load_plugin_callable("feedback.gate", "evaluate_safety_gate")
    promote_fn = _load_plugin_callable("feedback.status", "promote_or_stage_adapter")
    if gate_fn is None or promote_fn is None:
        raise HTTPException(status_code=501, detail="Portion 4 adaptation gate module not integrated.")

    stats = get_feedback_stats(db_path=resolved_db)
    drift_dist = _compute_drift_distance(stats, db_path=resolved_db)

    gate_report = _invoke_plugin(
        gate_fn,
        "Portion 4 safety gate",
        baseline_metrics=payload.baseline_metrics or {},
        candidate_metrics=payload.candidate_metrics or {},
    )

    resolved_weights_path = ""
    try:
        fb_cfg = importlib.import_module("feedback.config")
        adapter_root = Path(getattr(fb_cfg, "ADAPTER_DIR", ROOT_DIR / "data" / "models" / "adapters")).resolve()
        candidate_ckpt = (adapter_root / safe_disaster / adapter_ver / "adapter.pt").resolve()
        if candidate_ckpt.is_relative_to(adapter_root) and candidate_ckpt.is_file():
            resolved_weights_path = str(candidate_ckpt)
    except Exception:
        resolved_weights_path = ""

    return _invoke_plugin(
        promote_fn,
        "Portion 4 promotion",
        model_version="ResNet18-UNet-xBD-v1.0",
        adapter_version=adapter_ver,
        disaster_id=disaster_clean,
        weights_path=resolved_weights_path,
        gate_report=gate_report,
        promote=bool(payload.promote and resolved_weights_path),
        tier=payload.tier,
        trigger_reason="manual_api",
        drift_distance=drift_dist,
        db_path=resolved_db,
    )


@app.post("/api/adaptation/rollback", include_in_schema=False, tags=["Portion 4: Active Learning"])
async def rollback_adaptation_model(
    payload: AdaptationRollbackRequest = Body(default_factory=AdaptationRollbackRequest),
    db_path: str = Query(DEFAULT_DB_PATH),
) -> dict[str, Any]:
    """Perform one-click rollback of active segmentation adapter with SHA-256 audit entry."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    rb_fn = _load_plugin_callable("feedback.status", "rollback_adapter")
    if rb_fn is None:
        raise HTTPException(status_code=501, detail="Portion 4 rollback module not integrated.")
    return _invoke_plugin(
        rb_fn,
        "Portion 4 rollback",
        reason=payload.reason,
        actor_role=f"COMMANDER:{payload.commander_id}",
        db_path=resolved_db,
    )


@app.get("/api/audit-log", response_model=AuditLogReportResponse, include_in_schema=False, tags=["Portion 5: Audit & Security"])
async def api_get_audit_log(db_path: str = Query(DEFAULT_DB_PATH)) -> AuditLogReportResponse:
    """Retrieve full tamper-evident SHA-256 hash-chained audit ledger and verify continuity."""
    resolved_db = _ensure_db_ready(_resolve_db_path(db_path))
    entries = get_audit_log(limit=100, db_path=resolved_db)
    valid, count, msg = verify_audit_chain(db_path=resolved_db)
    return AuditLogReportResponse(
        valid=valid,
        count=count,
        message=msg,
        entries=[AuditLogEntry(**e) for e in entries],
    )


@app.post("/api/reset-sector", include_in_schema=False, tags=["Demo Management"])
async def api_reset_sector(db_path: str = Query(DEFAULT_DB_PATH)) -> dict[str, Any]:
    """Reset sector database to clean initial baseline state."""
    resolved_db = _resolve_db_path(db_path)
    summary = seed_sector(db_path=resolved_db)
    if Path(resolved_db).resolve() == Path(_resolve_db_path(DEFAULT_DB_PATH)).resolve():
        for fname in ("latest_pre.png", "latest_post.png", "latest_meta.json"):
            candidate = ACTIVE_UPLOADS_DIR / fname
            if candidate.is_file():
                try:
                    candidate.unlink()
                except OSError:
                    pass
    state = await api_get_state(db_path=resolved_db)
    return {"status": "reset_success", "summary": summary, "state": state.model_dump()}


# =====================================================================
# Programmatic Helpers & Server Entrypoint
# =====================================================================

def approve_and_dispatch(
    plan_id: str,
    commander_id: str = "CMD_LOCAL_01",
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Synchronous commander approval helper used by legacy callers and unit tests."""
    resolved_db = _resolve_db_path(db_path)
    outcome, audit_entry = decide_route_with_audit(
        route_id=plan_id,
        new_status="approved",
        actor_role=f"COMMANDER:{commander_id}",
        db_path=resolved_db,
    )
    if outcome != "OK" or audit_entry is None:
        raise RuntimeError(f"Unable to approve route {plan_id}: {outcome}")
    return {
        "plan_id": plan_id,
        "route_id": plan_id,
        "approval_status": "APPROVED",
        "action": "APPROVED",
        "actor": audit_entry["actor_role"],
        "commander_id": commander_id,
        "audit_hash": audit_entry["curr_hash"],
        "prev_hash": audit_entry["prev_hash"],
        "timestamp": audit_entry["utc_timestamp"],
        "audit_entry": audit_entry,
    }


def _bind_available_socket(
    host: str = "127.0.0.1",
    preferred_port: int = 8000,
    max_attempts: int = 20,
) -> tuple[socket.socket, int]:
    """Bind preferred_port or fall back to the next free port to avoid WinError 10048."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    last_err: OSError | None = None
    for candidate_port in range(preferred_port, preferred_port + max_attempts):
        sock = socket.socket(family, socket.SOCK_STREAM)
        # On Windows, enforce exclusive bind so occupied/TIME_WAIT ports fail cleanly instead of colliding.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        elif os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate_port))
            return sock, int(sock.getsockname()[1])
        except OSError as exc:
            last_err = exc
            sock.close()
    raise RuntimeError(
        f"Unable to bind {host} on ports {preferred_port}-{preferred_port + max_attempts - 1}: {last_err}"
    )


def run_web_app(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Ensure baseline DB is seeded and launch the FastAPI Tactical Web Console."""
    import uvicorn

    resolved_db = _resolve_db_path()
    if not Path(resolved_db).exists():
        seed_sector(db_path=resolved_db)

    env_port = os.getenv("AEGIS_PORT")
    requested_port = int(env_port) if (env_port and port == 8000) else port
    bound_sock, actual_port = _bind_available_socket(host=host, preferred_port=requested_port)
    url = f"http://{host}:{actual_port}"
    print("=" * 72)
    print(" AEGIS DISASTER RESPONSE COMMAND WEB APPLICATION")
    if actual_port != requested_port:
        print(f" [PORT] Port {requested_port} is already in use; using {url} instead.")
    print(f" URL: {url}")
    print(f" Swagger Documentation: {url}/docs")
    print(" Press Ctrl+C to stop the web server.")
    print("=" * 72)

    if open_browser and os.getenv("AEGIS_NO_BROWSER") != "1":
        timer = threading.Timer(0.8, webbrowser.open, args=[url])
        timer.daemon = True
        timer.start()

    config = uvicorn.Config(app, host=host, port=actual_port, reload=False, log_level="info")
    server = uvicorn.Server(config=config)
    try:
        server.run(sockets=[bound_sock])
    finally:
        if bound_sock.fileno() != -1:
            bound_sock.close()


def main() -> None:
    run_web_app()


if __name__ == "__main__":
    main()

