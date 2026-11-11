"""Portion 5 (Ahmed) — System Integration Backbone: Pydantic V2 Contract Models.

Defines the wire schemas and interface contracts per Unified System Document Section 3:
1. The 4 Shared Objects (Incident, Building, Route, Explanation).
2. Feedback Log (Section 4.4) & Audit Log structures.
3. RoadEdgeState supporting the Section 4.2 four-to-three road state mapping.
4. Request and Response shapes for all 18 system HTTP contract endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# =====================================================================
# Canonical Enums per Unified System Document
# =====================================================================

class DamageClassEnum(str, Enum):
    NO_DAMAGE = "no_damage"
    MINOR = "minor"
    MAJOR = "major"
    DESTROYED = "destroyed"


class SeverityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CapabilityEnum(str, Enum):
    MEDICAL = "medical"
    TRANSPORT = "transport"
    RESCUE = "rescue"


class IncidentStatusEnum(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    RESOLVED = "resolved"


class RouteStatusEnum(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    COMPLETED = "completed"


class RoadStateEnum(str, Enum):
    OPEN = "open"
    HAZARDOUS = "hazardous"
    BLOCKED = "blocked"


class AttributionMethodEnum(str, Enum):
    GRADCAM = "gradcam"
    SHAP = "shap"


class ResponderActionEnum(str, Enum):
    CONFIRM = "confirm"
    CORRECT = "correct"
    SKIP = "skip"


# =====================================================================
# Shared Primitive & Sub-Models
# =====================================================================

class LatLon(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="WGS84 Latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="WGS84 Longitude")


class GradcamAttribution(BaseModel):
    region: dict[str, Any] = Field(..., description="GeoJSON polygon of high activation region")
    weight: float = Field(..., description="Normalized gradient attribution intensity")


class ShapAttribution(BaseModel):
    feature: str = Field(..., description="Routing or vision feature name (e.g. travel_time, hazard_distance)")
    value: float = Field(..., description="SHAP attribution credit score")
    impact: Optional[float] = Field(default=None, description="Alias for value")
    attribution: Optional[float] = Field(default=None, description="Alias for value")


class RoadEdgeState(BaseModel):
    """Display road state plus optional Section 4.2 underlying attributes from Portion 2."""

    edge_id: str = Field(..., description="Identifier of the road segment")
    state: RoadStateEnum = Field(..., description="Display state: open, hazardous, blocked")
    closure_state: bool = Field(default=False, description="Hard closure flag from Portion 2")
    hazard_level: float = Field(default=0.0, ge=0.0, description="Aggregated hazard intensity/depth")
    capacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Normalized throughput capacity (0 = blocked)")
    vehicle_restriction: str = Field(default="none", description="Vehicle restriction class if applicable")

    def mapped_display_state(self, hazard_threshold: float = 0.15) -> RoadStateEnum:
        """Compute canonical Section 4.2 3-state display value from the 4 underlying attributes."""
        restriction_norm = str(self.vehicle_restriction or "none").strip().lower()
        if self.closure_state or self.capacity <= 0.0 or restriction_norm == "closed":
            return RoadStateEnum.BLOCKED
        if self.hazard_level >= hazard_threshold or restriction_norm not in ("", "none"):
            return RoadStateEnum.HAZARDOUS
        return RoadStateEnum.OPEN


class CounterfactualAlternative(BaseModel):
    alternative: str = Field(..., description="Counterfactual what-if description")
    delta_eta: float = Field(..., description="Change in ETA seconds compared to baseline")


# =====================================================================
# 1. The Four Shared Objects (Section 3 of Unified System Document)
# =====================================================================

# 3.1 Incident (Owner: Portion 2 Negm; Consumers: Portions 2, 3, 5)
class Incident(BaseModel):
    incident_id: str = Field(..., description="Unique incident identifier, e.g. INC-01")
    location: LatLon = Field(..., description="Geographic coordinate of incident")
    severity: SeverityEnum = Field(..., description="Incident severity level")
    required_capability: CapabilityEnum = Field(..., description="Required response capability")
    time_reported: str = Field(..., description="ISO 8601 timestamp of incident creation")
    status: IncidentStatusEnum = Field(default=IncidentStatusEnum.OPEN, description="Incident life-cycle status")


_DAMAGE_INT_MAP = {
    0: DamageClassEnum.NO_DAMAGE,
    1: DamageClassEnum.MINOR,
    2: DamageClassEnum.MAJOR,
    3: DamageClassEnum.DESTROYED,
}

_DAMAGE_STR_MAP = {
    "no_damage": DamageClassEnum.NO_DAMAGE,
    "no-damage": DamageClassEnum.NO_DAMAGE,
    "minor": DamageClassEnum.MINOR,
    "minor-damage": DamageClassEnum.MINOR,
    "major": DamageClassEnum.MAJOR,
    "major-damage": DamageClassEnum.MAJOR,
    "destroyed": DamageClassEnum.DESTROYED,
}


# 3.2 Building (Owner: Portion 1 Ammar; Consumers: Portions 3, 4, 5)
class Building(BaseModel):
    building_id: str = Field(..., description="Unique building identifier, e.g. BLD-0042")
    geometry: dict[str, Any] = Field(..., description="GeoJSON Polygon footprint")
    damage_class: DamageClassEnum = Field(..., description="4-tier EMS-98/xBD damage classification (no_damage, minor, major, destroyed)")
    class_label: str = Field(default="no_damage", description="Canonical damage label alias matching damage_class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Bayesian calibrated uncertainty score")
    max_prob: float = Field(default=0.9, ge=0.0, le=1.0, description="Calibrated maximum softmax probability")
    status_flag: str = Field(default="CALIBRATED", description="CALIBRATED or LOW_CONFIDENCE")
    gradcam_attribution: List[GradcamAttribution] = Field(default_factory=list, description="Grad-CAM spatial localization")
    shap_attribution: List[ShapAttribution] = Field(default_factory=list, description="Legacy feature attributions alias")
    shap_top3: List[ShapAttribution] = Field(default_factory=list, description="Top-3 feature attributions")
    embedding_ref: str = Field(default="", description="Pointer to frozen DINOv2 feature vector for drift monitoring")
    last_updated: str = Field(..., description="ISO 8601 timestamp")
    centroid_lat: float = Field(default=0.0, description="Building centroid latitude")
    centroid_lon: float = Field(default=0.0, description="Building centroid longitude")
    data_source: str = Field(default="xBD_houston_east", description="Imagery / assessment data source")

    @field_validator("damage_class", mode="before")
    @classmethod
    def _coerce_damage_class(cls, value: Any) -> DamageClassEnum:
        if isinstance(value, DamageClassEnum):
            return value
        if isinstance(value, bool):
            raise ValueError("damage_class cannot be boolean")
        if isinstance(value, int) and value in _DAMAGE_INT_MAP:
            return _DAMAGE_INT_MAP[value]
        text = str(value).strip().lower()
        if text.isdigit() and int(text) in _DAMAGE_INT_MAP:
            return _DAMAGE_INT_MAP[int(text)]
        if text in _DAMAGE_STR_MAP:
            return _DAMAGE_STR_MAP[text]
        return DamageClassEnum(text)

    @model_validator(mode="after")
    def _sync_class_label(self) -> "Building":
        self.class_label = self.damage_class.value
        return self


# 3.3 Route (Owner: Portion 2 Negm / Portion 3 Yousef / Portion 5 Ahmed; Consumers: Portions 3, 4, 5)
class Route(BaseModel):
    route_id: str = Field(..., description="Unique route identifier, e.g. PLAN-01 or RT-01")
    plan_id: Optional[str] = Field(default=None, description="Plan identifier alias")
    unit_id: str = Field(..., description="Assigned responder vehicle ID (MED-01, TRN-01, RSC-01)")
    incident_id: str = Field(..., description="Target incident ID")
    ppo_proposed_path: List[LatLon] = Field(default_factory=list, description="Raw PPO policy proposal before safety mask")
    masked_path: List[LatLon] = Field(default_factory=list, description="Safety-filtered path avoiding blocked hazards")
    mask_altered: bool = Field(default=False, description="True if safety filter altered the original PPO path")
    mask_reason: str = Field(default="", description="Reason why safety filter altered path")
    road_states: List[RoadEdgeState] = Field(default_factory=list, description="States of road edges along route")
    eta_seconds: float = Field(..., ge=0.0, description="Estimated time of arrival in seconds")
    total_eta_sec: float = Field(default=0.0, ge=0.0, description="Total fleet ETA in seconds")
    compute_ms: float = Field(default=0.0, ge=0.0, description="Routing computation latency in ms")
    status: RouteStatusEnum = Field(default=RouteStatusEnum.PENDING_APPROVAL, description="Commander approval status")
    approval_status: str = Field(default="PENDING_APPROVAL", description="Uppercase approval status")
    explanation: str = Field(default="", description="GTF-constrained plain-language tactical briefing")
    narrative_text: str = Field(default="", description="Operational briefing text")
    narrative_mode: str = Field(default="JINJA2_TEMPLATE", description="JINJA2_TEMPLATE or LLM_ADVISORY")
    uncertainty: float = Field(default=0.0, description="System uncertainty score")
    shap_values: List[ShapAttribution] = Field(default_factory=list, description="SHAP feature attributions on routing features")
    counterfactuals: List[CounterfactualAlternative] = Field(default_factory=list, description="Spatial what-if alternatives")
    routes: List[dict[str, Any]] = Field(default_factory=list, description="Multi-unit sub-route trajectories")
    audit_hash: str = Field(default="", description="Cryptographic SHA-256 audit digest")
    created_utc: Optional[str] = Field(default=None, description="ISO 8601 creation timestamp")


# 3.4 Explanation (Owner: Portion 3 Yousef; Consumers: Portions 3, 5)
class Explanation(BaseModel):
    explanation_id: str = Field(..., description="Unique explanation identifier, e.g. EXP-01")
    target_type: Literal["building", "route"] = Field(..., description="Target object: 'building' or 'route'")
    target_id: str = Field(..., description="Referenced building_id or route_id")
    natural_language: str = Field(..., description="GTF-constrained explanation text")
    attribution_method: AttributionMethodEnum = Field(..., description="Method used: gradcam or shap")
    system_uncertainty: float = Field(..., description="Bayesian uncertainty score")
    llm_used: bool = Field(default=False, description="Whether LLM generated the text")
    validation_passed: bool = Field(default=True, description="Whether text passed Portion 5 Pydantic and ETA gate")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


# =====================================================================
# Feedback & Audit Log Schemas
# =====================================================================

class FeedbackLog(BaseModel):
    target_id: str = Field(..., description="Building ID or tile ID being verified")
    disaster_id: str = Field(default="HOUSTON_HARVEY_2017", description="Disaster operation identifier")
    model_version: str = Field(default="ConvNeXt-Tiny-v1.0", description="Segmentation model version")
    adapter_version: str = Field(default="none", description="Active adaptation adapter ID")
    prediction: str = Field(..., description="Model predicted damage class")
    uncertainty: float = Field(..., ge=0.0, le=1.0, description="Model calibrated uncertainty")
    responder_action: ResponderActionEnum = Field(..., description="1-tap action: confirm, correct, skip")
    corrected_label: str = Field(default="", description="Corrected damage class if action is 'correct'")
    responder_id: str = Field(default="RESPONDER_FIELD_01", description="Field responder ID")
    time_taken_sec: float = Field(default=1.0, ge=0.0, description="Seconds taken to review")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp")


FeedbackItem = FeedbackLog


class AuditLogEntry(BaseModel):
    seq_id: int = Field(..., description="Monotonically increasing sequence number")
    utc_timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    actor_role: str = Field(..., description="Actor identity and role (e.g. COMMANDER:CMD_01)")
    event_type: str = Field(..., description="Action name (APPROVED, REJECTED, MODEL_PROMOTED, etc.)")
    route_id: str = Field(default="", description="Associated route plan ID")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload dictionary")
    prev_hash: str = Field(..., description="SHA-256 hash of previous audit log record")
    curr_hash: str = Field(..., description="Computed SHA-256 hash of current entry")


# =====================================================================
# Request / Response Models for the 18 HTTP Endpoints
# =====================================================================

# 1. Health Probe
class HealthResponse(BaseModel):
    status: str = "healthy"
    application: str = "aegis-disaster-response-backbone"
    version: str = "2.2.0"
    offline_ready: bool = True
    active_endpoints: int = 18


# 2. State Response
class TacticalStateResponse(BaseModel):
    sector: str
    data_mode: str
    buildings_geojson: dict[str, Any]
    roads_geojson: dict[str, Any]
    road_counts: dict[str, int]
    active_plan: Optional[Route] = None
    active_routes: List[Route]
    incidents: List[Incident]
    drift_distance: float
    feedback_count: int
    audit_status: dict[str, Any]


# 3 & 4. Assessment Upload & Task Status (Portion 1 Ammar)
class UploadAssessmentRequest(BaseModel):
    pre_disaster_image: Optional[str] = Field(default=None, description="Base64 data URI or data/ URI of pre-disaster satellite tile")
    post_disaster_image: Optional[str] = Field(default=None, description="Base64 data URI or data/ URI of post-disaster satellite tile")
    pre_tile_ref: Optional[str] = Field(default=None, description="Legacy/UI alias for pre-disaster satellite tile reference")
    post_tile_ref: Optional[str] = Field(default=None, description="Legacy/UI alias for post-disaster satellite tile reference")
    sector_id: str = Field(default="houston-east", description="Target geographical sector")
    bounds: Optional[List[float]] = Field(default=None, min_length=4, max_length=4, description="Optional [min_lon, min_lat, max_lon, max_lat] WGS84 bbox")
    footprints: Optional[List[dict[str, Any]]] = Field(default=None, description="Optional pre-supplied building footprint polygons")


class UploadAssessmentResponse(BaseModel):
    task_id: str = Field(..., description="Unique background assessment task ID")
    status: str = Field(default="pending", description="pending | in_progress | completed")
    message: str = Field(..., description="Status summary message")
    estimated_duration_sec: float = Field(default=12.5, description="Expected processing time")


class AssessmentStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_pct: float
    buildings_detected: int
    low_confidence_count: int
    timestamp: str


# 5 & 6. Buildings Query & Detail (Portion 1 Ammar)
class BuildingListResponse(BaseModel):
    total_count: int
    buildings: List[Building]


class BuildingDetailResponse(BaseModel):
    building: Building
    gradcam_overlay_url: Optional[str] = None
    historical_damage: Optional[dict[str, Any]] = None


# 7. Building Correction (Portion 3/4)
class BuildingCorrectionRequest(BaseModel):
    corrected_damage_class: DamageClassEnum
    responder_id: str = "RESPONDER_FIELD_01"
    note: str = ""


class BuildingCorrectionResponse(BaseModel):
    building_id: str
    previous_class: str
    updated_class: str
    feedback_logged: bool
    audit_logged: bool
    drift_distance: float = 0.0
    feedback_count: int = 1
    severity_jump: int = 0
    requires_second_opinion: bool = False
    reliability_weight: float = 1.0
    rate_limited_warning: bool = False


# 8 & 9. Incidents (Portion 2 Negm)
class IncidentCreateRequest(BaseModel):
    incident_id: Optional[str] = None
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    severity: SeverityEnum = SeverityEnum.MEDIUM
    required_capability: CapabilityEnum = CapabilityEnum.RESCUE
    description: Optional[str] = ""


class IncidentListResponse(BaseModel):
    total_count: int
    incidents: List[Incident]


# 10 & 11 & 12. Routing & Dispatch (Portion 2 Negm)
class DispatchComputeRequest(BaseModel):
    water_level_delta_m: float = Field(default=0.0, ge=-2.0, le=5.0)
    blocked_edges: List[str] = Field(default_factory=list)
    opened_edges: List[str] = Field(default_factory=list, description="List of road edge IDs explicitly reopened/cleared")
    arbitrary_blockage: bool = Field(default=False, description="Simulate an arbitrary live satellite-detected corridor blockage")
    arbitrary_seed: Optional[int] = Field(default=None, description="Optional deterministic seed for arbitrary blockage selection")


class DispatchComputeResponse(Route):
    """Response shape for MARL dispatch endpoint: returns a full Route object with multi-unit sub-routes."""

    blocked_edges: List[str] = Field(default_factory=list)


class RouteListResponse(BaseModel):
    total_routes: int
    routes: List[Route]


# 13. What-If Counterfactual (Portion 2 Negm / Portion 5 Ahmed)
class WhatIfRequest(BaseModel):
    blocked_edges: List[str] = Field(default_factory=list, description="List of road edge IDs manually marked as blocked")
    opened_edges: List[str] = Field(default_factory=list, description="List of road edge IDs explicitly reopened/cleared")
    flood_delta: float = Field(default=0.0, ge=-2.0, le=5.0, description="Water level surge delta in meters")


class WhatIfResponse(BaseModel):
    plan_id: str
    compute_ms: float
    flood_delta: float
    newly_blocked_edges: List[str]
    dynamic_hazards: dict[str, dict[str, str]] = Field(default_factory=dict)
    total_eta_sec: float
    whatif_eta_sec: float
    delta_sec: float
    routes: List[dict[str, Any]]
    narrative_text: str
    narrative_mode: str


# 14 & 15. Explanations (Portion 3 Yousef) — Inherit from Explanation contract object
class RouteExplanationResponse(Explanation):
    route_id: str
    shap_values: List[ShapAttribution]
    mask_altered: bool
    mask_reason: str


class BuildingExplanationResponse(Explanation):
    building_id: str
    damage_class: str
    confidence: float
    gradcam_regions: List[GradcamAttribution]
    explanation_text: str
    gradcam_image_url: Optional[str] = None


# 16 & 17. Route Commander Approval & Rejection Gate (Portion 5 Ahmed)
class RouteDecisionRequest(BaseModel):
    commander_id: str = Field(default="CMD_LOCAL_01", min_length=2, max_length=64, description="Incident Commander identifier")
    notes: Optional[str] = Field(default="", description="Operational justification or commander directive notes")


class RouteDecisionResponse(BaseModel):
    route_id: str
    action: str  # APPROVED or REJECTED
    actor: str   # Exact actor string hashed into SHA-256 chain
    commander_id: str
    audit_hash: str
    prev_hash: str
    timestamp: str
    message: str


# 18. Feedback Submission (Portion 4 Abdulrahman)
class FeedbackSubmitResponse(BaseModel):
    feedback_id: int
    target_id: str
    responder_action: str
    timestamp: str
    status: str = "queued_offline_wal"
    drift_distance: float = 0.0
    drift_detected: bool = False
    feedback_count: int = 1
    severity_jump: int = 0
    requires_second_opinion: bool = False
    reliability_weight: float = 1.0
    rate_limited_warning: bool = False


# Extra Diagnostic & Inspection Models
class DriftReportResponse(BaseModel):
    drift_distance: float
    soft_threshold: float = 0.08
    threshold: float = 0.15
    soft_warning: bool = False
    drift_detected: bool
    total_feedback_samples: int
    pending_retrain_trigger: bool
    last_adaptation_tier: str
    embedding_signal_available: bool = False
    embedding_reason: str = ""
    operational_metrics: dict[str, Any] = Field(default_factory=dict)
    active_adapter_version: str = "none"


class AdaptationRunRequest(BaseModel):
    disaster_id: str = Field(default="HOUSTON_HARVEY_2017", min_length=1)
    tier: Literal["tier_a", "tier_b", "tier_c"] = "tier_a"
    adapter_version: Optional[str] = None
    promote: bool = True
    epochs: int = Field(default=8, ge=1, le=20)
    learning_rate: float = Field(default=0.02, gt=0.0, le=0.5)
    train_from_feedback: bool = False
    baseline_metrics: Optional[dict[str, Any]] = None
    candidate_metrics: Optional[dict[str, Any]] = None


class AdaptationRollbackRequest(BaseModel):
    commander_id: str = Field(default="CMD_LOCAL_01", min_length=2, max_length=64)
    reason: str = Field(default="Operator requested model adapter rollback")


class AuditLogReportResponse(BaseModel):
    valid: bool
    count: int
    message: str
    entries: List[AuditLogEntry]

