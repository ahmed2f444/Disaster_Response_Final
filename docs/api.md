# AEGIS Disaster Response Platform — Portion 5 API & Integration Contract

**Owner:** Ahmed (Portion 5 — Integration Backbone, WebGIS & Governance)  
**Source of Truth:** `Unified System Document.pdf` (Sections 2–12), `ahmed.pdf`, and `models.py` / `db.py` / `main.py`.

---

## 1. Ownership & Plug-In Architecture

| Module / File | Portion & Lead | Status in Portion 5 Deliverable |
| :--- | :--- | :--- |
| `main.py`, `models.py`, `db.py`, `seed.py`, `static/`, `docs/api.md`, `tests/` | **Portion 5: Ahmed** | **Implemented** (FastAPI spine, Validation Gate, SHA-256 chain, SQLite WAL, 6-in-1 Web UI) |
| `damage/` (`assess.py`, `dataset.py`, `exports.py`, `model.py`, `train.py`) | **Portion 1: Ammar** | **Integrated** (Pretrained ResNet-18 U-Net inference, Grad-CAM & embedding export) |
| `routing/` (`env.py`, `features.py`, `generate.py`, `graph.py`, `mask.py`, `train.py`) | **Portion 2: Negm** | **Integrated** — Houston-East 13-node graph, PPO MARL policy + safety mask (`compute_multi_unit_routes`, `run_counterfactual`) |
| `explain/` (`generate.py`, `gradcam.py`, `llm.py`, `shap_explainer.py`, `validate.py`) | **Portion 3: Yousef** | **Integrated** — Grad-CAM saliency & PNG overlays, 7-feature KernelSHAP (`ppo_proposed_path`), GTF validator & local Mistral-7B translator |
| `feedback/` (`config.py`, `feedback.py`, `drift.py`, `calibrate.py`, `selection.py`, `replay.py`, `adapt.py`, `gate.py`, `status.py`) | **Portion 4: Abdulrahman** | **Integrated** — Multi-signal drift monitor, uncertainty+diversity active learning, non-FIFO dual-buffer replay, frozen-encoder Tier A adapter & 5-check safety gate |

> **Zero-SQL Rule:** No teammate module writes raw SQL (`SELECT`, `INSERT INTO`, `UPDATE`, `DELETE FROM`, `CREATE TABLE`). All persistence uses helper functions exported by `db.py`.

---

## 2. Shared Data Objects (`models.py` & `db.py`)

1. **`Incident`** (`incidents` table): `incident_id`, `location` (`lat`, `lon`), `severity` (`low|medium|high|critical`), `required_capability` (`medical|transport|rescue`), `time_reported`, `status` (`open|assigned|resolved`).
2. **`Building`** (`buildings` table): `building_id`, `geometry` (GeoJSON Polygon), `damage_class` (`no_damage|minor|major|destroyed`), `class_label` (canonical string alias), `confidence` (Bayesian calibrated), `max_prob`, `status_flag` (`CALIBRATED|LOW_CONFIDENCE`), `gradcam_attribution`, `shap_attribution`, `shap_top3`, `embedding_ref` (frozen DINOv2 vector pointer), `last_updated`, `centroid_lat`, `centroid_lon`, `data_source`.
3. **`Route`** (`routes` table): `route_id`, `plan_id`, `unit_id`, `incident_id`, `ppo_proposed_path`, `masked_path`, `mask_altered`, `mask_reason`, `road_states`, `eta_seconds`, `total_eta_sec`, `compute_ms`, `status` (`pending_approval|approved|rejected|active|completed`), `approval_status`, `explanation`, `narrative_text`, `narrative_mode`, `uncertainty`, `shap_values`, `counterfactuals`, `routes`, `audit_hash`, `created_utc`.
4. **`Explanation`** (`explanations` table): `explanation_id`, `target_type` (`building|route`), `target_id`, `natural_language`, `attribution_method` (`gradcam|shap`), `system_uncertainty`, `llm_used`, `validation_passed`, `timestamp`.
5. **`RoadEdgeState`** (`road_edges` table): Maps Portion 2's 4 underlying attributes (`closure_state`, `hazard_level`, `capacity`, `vehicle_restriction`) into 3 display states (`open`, `hazardous`, `blocked`) per Section 4.2.
6. **`FeedbackLog`** (`feedback_log` table, Section 4.4): `target_id`, `disaster_id`, `model_version`, `adapter_version`, `prediction`, `uncertainty`, `responder_action` (`confirm|correct|skip`), `corrected_label`, `responder_id`, `time_taken_sec`, `timestamp`.
7. **`AuditLogEntry`** (`audit_log` table): `seq_id`, `utc_timestamp`, `actor_role`, `event_type`, `route_id`, `payload`, `prev_hash`, `curr_hash`.

---

## 3. The 18 HTTP Contract Endpoints (`main.py`)

All operational endpoints accept an optional `?db_path=...` query parameter strictly scoped inside `data/` (defaulting to `data/poc_mission.db`; uninitialized custom paths return `HTTP 404`).

| # | Method & Path | Owner | Request Model | Response Model | Unimplemented / Runtime Behavior |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | `GET /healthz` | P5 (Ahmed) | — | `HealthResponse` | Returns `200` liveness, `version="2.2.0"`, & `active_endpoints: 18`. |
| **2** | `GET /api/state` | P5 (Ahmed) | `db_path` (query) | `TacticalStateResponse` | Returns sector GeoJSON (`< 100 KB`, `< 10,000` features) with canonical `damage_class`, routes, incidents, live Portion 4 `drift_distance`, audit status. |
| **3** | `POST /api/assessment/upload` | P1 (Ammar) | `UploadAssessmentRequest` | `UploadAssessmentResponse` | Delegates to `damage.assess.run_vision_assessment` (`unet_resnet18_best.pth`, loading promoted Portion 4 adapter if active), preserves responder label corrections, updates corridor damage/incidents, and triggers persistent MARL recalculation when hazards change (`status="pending"` on metadata-only requests). |
| **4** | `GET /api/assessment/status/{task_id}` | P1 (Ammar) | `task_id` (path) | `AssessmentStatusResponse` | Delegates to `damage.assess.get_assessment_status` and reports live progress, detected buildings & low-confidence counts. |
| **5** | `GET /api/buildings` | P1 (Ammar) | `damage_class` (`no_damage\|minor\|major\|destroyed` or `0..3`), `min_confidence` | `BuildingListResponse` | Lists assessed `Building` objects from `db.py` (`422` on invalid filter). |
| **6** | `GET /api/buildings/{building_id}` | P1 (Ammar) | `building_id` (path) | `BuildingDetailResponse` | Returns full `Building` record including Grad-CAM regions & `embedding_ref`. |
| **7** | `POST /api/buildings/{building_id}/correct` | P3 / P4 | `BuildingCorrectionRequest` | `BuildingCorrectionResponse` | Updates label via `db.update_building_label`, validates & queues feedback with noisy-label `severity_jump`/`requires_second_opinion`, ingests into dual-buffer replay, logs audit, and computes live multi-signal drift. |
| **8** | `POST /api/incidents` | P2 (Negm) | `IncidentCreateRequest` | `Incident` | Inserts new `Incident` via `db.insert_incident` and logs audit event. |
| **9** | `GET /api/incidents` | P2 (Negm) | `status` (query) | `IncidentListResponse` | Queries `Incident` objects via `db.get_incidents`. |
| **10** | `POST /api/routes/dispatch` | P2 (Negm) | `DispatchComputeRequest` | `DispatchComputeResponse` | **Active (Portion 2):** Updates DB hazards (`blocked_edges`, `opened_edges`, optional `arbitrary_blockage`), computes PPO + safety-mask multi-unit routes (`< 500ms`), assigns open incidents, persists pending `Route`, and logs `ROUTE_DISPATCHED` (`HTTP 501` if plugin missing). |
| **11** | `GET /api/routes` | P2 (Negm) | `include_counterfactuals` | `RouteListResponse` | Lists seeded/computed `Route` plans with `ppo_proposed_path` and `masked_path`. |
| **12** | `GET /api/routes/{route_id}` | P2 (Negm) | `route_id` (path) | `Route` | Returns detailed `Route` record via `db.get_route`. |
| **13** | `POST /api/routes/what-if` | P2 / P3 / P5 | `WhatIfRequest` | `WhatIfResponse` | **Active (Portion 2):** Runs dry-run counterfactual flood/open/blockage reroute (`blocked_edges`, `opened_edges`, `flood_delta`) via `routing.generate.run_counterfactual` (`dynamic_hazards`, `newly_blocked_edges`, GTF narrative) without mutating baseline DB (`HTTP 501` if plugin missing). |
| **14** | `GET /api/explanations/route/{route_id}` | P3 / P5 | `route_id` (path) | `RouteExplanationResponse` | Queries stored/P3 `explain.generate.explain_route` narrative, SHAP attributions, separate `system_uncertainty`, and enforces Portion 5 Validation & ETA Gate (persisting `validation_passed=false` on failure). |
| **15** | `GET /api/explanations/building/{building_id}` | P3 / P5 | `building_id` (path) | `BuildingExplanationResponse` | Queries stored/P3 `explain.generate.explain_building` narrative, Grad-CAM regions (`[]` if unpopulated), separate `system_uncertainty`, and enforces Portion 5 Validation Gate. |
| **16** | `POST /api/routes/{route_id}/approve` | P5 (Ahmed) | `RouteDecisionRequest` (required body) | `RouteDecisionResponse` | Atomically transitions `pending_approval -> approved` & appends SHA-256 Merkle chain entry; returns **`HTTP 409 Conflict`** if already decided. |
| **17** | `POST /api/routes/{route_id}/reject` | P5 (Ahmed) | `RouteDecisionRequest` (required body) | `RouteDecisionResponse` | Atomically transitions `pending_approval -> rejected` & appends SHA-256 Merkle chain entry; returns **`HTTP 409 Conflict`** if already decided. |
| **18** | `POST /api/feedback` | P4 (Abdulrahman) | `FeedbackItem` | `FeedbackSubmitResponse` | Validates damage classes/bounds, enforces noisy-label second-opinion flags & responder rate warnings, ingests `confirm`/`correct` into dual-buffer replay, queues in SQLite WAL, and returns live multi-signal drift. |

**Auxiliary, Compatibility & Static Routes (`include_in_schema=False`):**
- `GET /` — Serves `static/index.html` (6-in-1 Tactical Web Command Console backed by Leaflet v1.9.4 SVG/Canvas markers & `< 50 ms` client render telemetry)
- `GET /static/*` — Serves offline frontend assets (`style.css`, `app.js`, Leaflet v1.9.4 `leaflet.js`, `leaflet.css`, and `images/`)
- `POST /api/what-if` — Legacy compatibility alias for `POST /api/routes/what-if`
- `POST /api/approve` — Legacy compatibility alias for `POST /api/routes/{route_id}/approve` (explicit JSON body required, enforces `409 Conflict` gate lock)
- `POST /api/reject` — Legacy compatibility alias for `POST /api/routes/{route_id}/reject` (explicit JSON body required, enforces `409 Conflict` gate lock)
- `GET /api/adaptation/drift` — Returns `DriftReportResponse` with live Portion 4 multi-signal drift distance, soft warning (`>= 0.08`), hard trigger (`>= 0.15`), embedding availability, and active adapter version
- `GET /api/feedback` — Lists WAL-queued responder feedback records with aggregate stats
- `GET /api/adaptation/status` — Returns active segmentation model/adapter version, replay buffer counts, and safety-gate run history
- `GET /api/adaptation/queue` — Returns uncertainty + diversity ranked building candidates for non-blocking responder labeling
- `POST /api/adaptation/run` — Executes fail-closed 5-check safety-gated segmentation adaptation (`tier_a|tier_b|tier_c`); when `train_from_feedback=true`, trains a frozen-backbone Tier-A ResNet-18 adapter on active imagery + responder feedback before gate evaluation & promotion
- `POST /api/adaptation/rollback` — Performs one-click adapter rollback and appends `MODEL_ROLLBACK` to the SHA-256 audit ledger
- `GET /api/explanations/building/{building_id}/overlay.png` — Serves rendered Portion 3 Grad-CAM PNG heatmap overlay when `.npy` exports exist (`404` otherwise)
- `GET /api/audit-log` — Returns `AuditLogReportResponse` with `verify_audit_chain()` status
- `POST /api/reset-sector` — Re-seeds `data/poc_mission.db` cleanly

---

## 4. Portion 5 Governance & Validation Contracts

### 4.1 SHA-256 Merkle Audit Chain & Atomic Gate Lock
Every commander approval/rejection is committed atomically via `db.decide_route_with_audit()` inside a `BEGIN IMMEDIATE` transaction, enforcing single-decision immutability (`pending_approval` only; subsequent attempts return `HTTP 409 Conflict`) and chaining deterministically from `GENESIS_PREV_HASH = "0" * 64`:

```text
curr_hash = SHA-256(prev_hash + route_id + action + actor + timestamp)
```

`db.verify_audit_chain()` recomputes every link from `seq_id = 1..N`.

### 4.2 Explanation Validation & ETA Gate (`Unified System Document.pdf` Sec. 7 & Stage 7)
Portion 5 validates explanation text before display in Endpoints 13, 14 & 15:
- Route explanations: checks numeric ETAs (`ETA 42s`, `ETA of 120 seconds`, `ETA is 45s`) against computed `Route.eta_seconds` (`<= 0.5s` tolerance), cited road edge IDs against known route/sector edges, `mask_altered` consistency, and blocks LLM hedging, PPO override phrasing, or self-reported confidence claims.
- Building explanations: checks damage class label (including `"no damage"` / `"no_damage"`) and confidence percentage against `Building.class_label` / `Building.confidence` (`<= 2.0%` tolerance) and blocks LLM hedging/self-reported confidence.
- On failure: substitutes deterministic `[TEMPLATE FALLBACK]` text and persists/returns `validation_passed: false`.

### 4.3 Section 4.2 Road Mapping & Stage 10 Model Transition Helpers (`db.py`)
- `db.derive_road_display_state(closure_state, hazard_level, capacity, vehicle_restriction)` maps Portion 2's 4 underlying road attributes into the 3 display states (`open`, `hazardous`, `blocked`).
- `db.record_model_transition(...)` appends Stage 10 safety-gate promotions and rollbacks to the SHA-256 `audit_log` chain.

---

## 5. Teammate Plug-In Hooks (Auto-Detected by `main.py`)

When teammates populate their own directories, `main.py` detects these entry points lazily with signature-filtered invocation and `HTTP 502` error isolation:

- **Portion 1 (`damage/assess.py`):** `run_vision_assessment(...)` and `get_assessment_status(task_id=..., db_path=...)`
- **Portion 2 (`routing/graph.py` & `routing/generate.py`):**
  - `update_road_hazards(db_path=..., water_level_delta_m=...)`
  - `compute_multi_unit_routes(db_path=...) -> dict` (matching `DispatchComputeResponse`)
  - `run_counterfactual(blocked_edges=..., flood_delta=..., db_path=...) -> dict`
- **Portion 3 (`explain/generate.py`):**
  - `generate_narrative(plan_or_diff, use_llm_stretch=False) -> tuple[str, str]`
  - `explain_route(route=..., route_id=..., db_path=...)`
  - `explain_building(building=..., building_id=..., db_path=...)`
  - `schedule_async_route_explanation(route, db_path=...)` (Stage 6 background explanation outside `<500 ms` reroute window)
- **Portion 4 (`feedback/drift.py`):** `compute_drift_distance(feedback_stats: dict) -> float`


