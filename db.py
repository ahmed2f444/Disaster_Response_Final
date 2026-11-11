"""Portion 5 (Ahmed) — System Integration & Edge Persistence Layer.

Shared SQLite WAL Database Engine & Tamper-Evident SHA-256 Audit Chain.
Implements the 4 core shared objects (Building, Route, Incident, Explanation),
Road Edges (with Section 4.2 underlying attributes), Feedback Log (Section 4.4),
and SHA-256 Merkle Audit Log (including Stage 10 model version transition logging).
Provides comprehensive zero-SQL database access helpers for all portions.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = "data/poc_mission.db"
GENESIS_PREV_HASH = "0" * 64

DAMAGE_INT_TO_LABEL = {0: "no_damage", 1: "minor", 2: "major", 3: "destroyed"}
DAMAGE_LABEL_TO_INT = {v: k for k, v in DAMAGE_INT_TO_LABEL.items()}
_DAMAGE_STR_ALIASES = {
    "no_damage": "no_damage",
    "no-damage": "no_damage",
    "minor": "minor",
    "minor-damage": "minor",
    "major": "major",
    "major-damage": "major",
    "destroyed": "destroyed",
}


def normalize_damage_class(value: Any) -> tuple[int, str]:
    """Coerce int (0..3), Enum, or canonical/hyphenated string into (tier_int, canonical_label)."""
    raw = getattr(value, "value", value)
    if isinstance(raw, bool):
        raise ValueError(f"Invalid boolean damage_class: {value!r}")
    if isinstance(raw, int):
        if raw in DAMAGE_INT_TO_LABEL:
            return raw, DAMAGE_INT_TO_LABEL[raw]
        raise ValueError(f"Invalid integer damage_class: {raw}")
    text = str(raw).strip().lower()
    if text.isdigit() and int(text) in DAMAGE_INT_TO_LABEL:
        tier = int(text)
        return tier, DAMAGE_INT_TO_LABEL[tier]
    canonical = _DAMAGE_STR_ALIASES.get(text)
    if canonical is not None:
        return DAMAGE_LABEL_TO_INT[canonical], canonical
    raise ValueError(f"Invalid damage_class: {value!r}")


def derive_road_display_state(
    closure_state: bool,
    hazard_level: float,
    capacity: float,
    vehicle_restriction: str = "none",
    hazard_threshold: float = 0.15,
) -> str:
    """Map Section 4.2 four underlying attributes into display state (open|hazardous|blocked)."""
    restriction_norm = str(vehicle_restriction or "none").strip().lower()
    if closure_state or float(capacity) <= 0.0 or restriction_norm == "closed":
        return "blocked"
    if float(hazard_level) >= hazard_threshold or restriction_norm not in ("", "none"):
        return "hazardous"
    return "open"


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 1. Shared Building Object (Owner: Portion 1 Ammar; Consumers: Portions 3, 4, 5)
CREATE TABLE IF NOT EXISTS buildings (
    building_id              TEXT PRIMARY KEY,
    geom_geojson             TEXT NOT NULL,
    centroid_lat             REAL NOT NULL,
    centroid_lon             REAL NOT NULL,
    damage_class             TEXT NOT NULL CHECK (damage_class IN ('no_damage', 'minor', 'major', 'destroyed')),
    damage_tier              INTEGER NOT NULL DEFAULT 0 CHECK (damage_tier IN (0, 1, 2, 3)),
    class_label              TEXT NOT NULL CHECK (class_label IN ('no_damage', 'minor', 'major', 'destroyed', 'NO_DAMAGE', 'MINOR', 'MAJOR', 'DESTROYED')),
    confidence               REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    max_prob                 REAL NOT NULL DEFAULT 0.90,
    status_flag              TEXT NOT NULL DEFAULT 'CALIBRATED' CHECK (status_flag IN ('CALIBRATED', 'LOW_CONFIDENCE')),
    gradcam_attribution_json TEXT NOT NULL DEFAULT '[]',
    shap_top3_json           TEXT NOT NULL DEFAULT '[]',
    embedding_ref            TEXT NOT NULL DEFAULT '',
    last_updated             TEXT NOT NULL,
    data_source              TEXT NOT NULL DEFAULT 'xBD_houston_east'
);

-- 2. Road Edges (Owner: Portion 2 Negm; Consumers: Portions 2, 5)
-- Stores both PoC flood/debris telemetry and Section 4.2 four-attribute mapping inputs.
CREATE TABLE IF NOT EXISTS road_edges (
    edge_id             TEXT PRIMARY KEY,
    u_node              INTEGER NOT NULL,
    v_node              INTEGER NOT NULL,
    geom_geojson        TEXT NOT NULL,
    length_m            REAL NOT NULL,
    base_travel_sec     REAL NOT NULL,
    flood_depth_m       REAL NOT NULL DEFAULT 0.0,
    debris_score        REAL NOT NULL DEFAULT 0.0,
    closure_state       INTEGER NOT NULL DEFAULT 0 CHECK (closure_state IN (0, 1)),
    hazard_level        REAL NOT NULL DEFAULT 0.0,
    capacity            REAL NOT NULL DEFAULT 1.0,
    vehicle_restriction TEXT NOT NULL DEFAULT 'none',
    edge_status         TEXT NOT NULL CHECK (edge_status IN ('open', 'hazardous', 'blocked', 'OPEN', 'HAZARDOUS', 'BLOCKED'))
);

-- 3. Shared Incident Object (Owner: Portion 2 Negm; Consumers: Portions 2, 3, 5)
CREATE TABLE IF NOT EXISTS incidents (
    incident_id         TEXT PRIMARY KEY,
    location_lat        REAL NOT NULL,
    location_lon        REAL NOT NULL,
    severity            TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    required_capability TEXT NOT NULL CHECK (required_capability IN ('medical', 'transport', 'rescue')),
    time_reported       TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('open', 'assigned', 'resolved'))
);

-- 4. Shared Route Object (Owner: Portion 2 Negm / Portion 3 Yousef / Portion 5 Ahmed)
CREATE TABLE IF NOT EXISTS routes (
    plan_id                 TEXT PRIMARY KEY,
    route_id                TEXT,
    approval_status         TEXT NOT NULL DEFAULT 'PENDING_APPROVAL',
    unit_id                 TEXT NOT NULL DEFAULT 'MED-01',
    incident_id             TEXT NOT NULL DEFAULT 'INC-01',
    ppo_proposed_path_json  TEXT NOT NULL DEFAULT '[]',
    masked_path_json        TEXT NOT NULL DEFAULT '[]',
    mask_altered            INTEGER NOT NULL DEFAULT 0 CHECK (mask_altered IN (0, 1)),
    mask_reason             TEXT NOT NULL DEFAULT '',
    road_states_json        TEXT NOT NULL DEFAULT '[]',
    eta_seconds             REAL NOT NULL DEFAULT 0.0,
    status                  TEXT NOT NULL DEFAULT 'pending_approval' CHECK (status IN ('pending_approval', 'approved', 'rejected', 'active', 'completed', 'PENDING_APPROVAL', 'APPROVED', 'REJECTED')),
    explanation             TEXT NOT NULL DEFAULT '',
    uncertainty             REAL NOT NULL DEFAULT 0.0,
    shap_values_json        TEXT NOT NULL DEFAULT '[]',
    counterfactuals_json    TEXT NOT NULL DEFAULT '[]',
    audit_hash              TEXT NOT NULL DEFAULT '',
    created_utc             TEXT NOT NULL,
    is_counterfactual       INTEGER NOT NULL DEFAULT 0 CHECK (is_counterfactual IN (0, 1)),
    total_eta_sec           REAL NOT NULL DEFAULT 0.0,
    compute_ms              REAL NOT NULL DEFAULT 0.0,
    routes_json             TEXT NOT NULL DEFAULT '[]',
    narrative_text          TEXT NOT NULL DEFAULT '',
    narrative_mode          TEXT NOT NULL DEFAULT 'JINJA2_TEMPLATE'
);

-- 5. Shared Explanation Object (Owner: Portion 3 Yousef; Consumers: Portions 3, 5)
CREATE TABLE IF NOT EXISTS explanations (
    explanation_id      TEXT PRIMARY KEY,
    target_type         TEXT NOT NULL CHECK (target_type IN ('building', 'route')),
    target_id           TEXT NOT NULL,
    natural_language    TEXT NOT NULL,
    attribution_method  TEXT NOT NULL CHECK (attribution_method IN ('gradcam', 'shap')),
    system_uncertainty  REAL NOT NULL,
    llm_used            INTEGER NOT NULL DEFAULT 0 CHECK (llm_used IN (0, 1)),
    validation_passed   INTEGER NOT NULL DEFAULT 1 CHECK (validation_passed IN (0, 1)),
    timestamp           TEXT NOT NULL
);

-- 6. Feedback Log (Owner: Portion 4 Abdulrahman; Consumers: Portions 4, 5)
CREATE TABLE IF NOT EXISTS feedback_log (
    feedback_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id               TEXT NOT NULL,
    disaster_id             TEXT NOT NULL DEFAULT 'HOUSTON_HARVEY_2017',
    model_version           TEXT NOT NULL DEFAULT 'ConvNeXt-Tiny-v1.0',
    adapter_version         TEXT NOT NULL DEFAULT 'none',
    prediction              TEXT NOT NULL,
    uncertainty             REAL NOT NULL,
    responder_action        TEXT NOT NULL CHECK (responder_action IN ('confirm', 'correct', 'skip', 'CONFIRM', 'CORRECT', 'SKIP')),
    corrected_label         TEXT NOT NULL DEFAULT '',
    responder_id            TEXT NOT NULL,
    time_taken_sec          REAL NOT NULL DEFAULT 0.0,
    timestamp               TEXT NOT NULL,
    severity_jump           INTEGER NOT NULL DEFAULT 0,
    requires_second_opinion INTEGER NOT NULL DEFAULT 0 CHECK (requires_second_opinion IN (0, 1)),
    reliability_weight      REAL NOT NULL DEFAULT 1.0
);

-- 7. Audit Log (Owner: Portion 5 Ahmed - Commander Approval Gate & Merkle Chain)
CREATE TABLE IF NOT EXISTS audit_log (
    seq_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    utc_timestamp   TEXT NOT NULL,
    actor_role      TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    route_id        TEXT NOT NULL DEFAULT '',
    payload_json    TEXT NOT NULL,
    payload_hash    TEXT NOT NULL DEFAULT '',
    prev_hash       TEXT NOT NULL,
    curr_hash       TEXT NOT NULL
);

-- 8. Dual-Buffer Replay Memory (Owner: Portion 4 Abdulrahman)
CREATE TABLE IF NOT EXISTS replay_buffer (
    sample_id       TEXT PRIMARY KEY,
    target_id       TEXT NOT NULL,
    disaster_id     TEXT NOT NULL,
    damage_class    TEXT NOT NULL CHECK (damage_class IN ('no_damage', 'minor', 'major', 'destroyed')),
    buffer_tier     TEXT NOT NULL CHECK (buffer_tier IN ('long_term', 'recent')),
    priority        REAL NOT NULL DEFAULT 1.0,
    uncertainty     REAL NOT NULL DEFAULT 0.0,
    embedding_ref   TEXT NOT NULL DEFAULT '',
    sample_json     TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

-- 9. Model & Adapter Registry (Owner: Portion 4 Abdulrahman / Portion 5 Audit)
CREATE TABLE IF NOT EXISTS model_registry (
    registry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version   TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    disaster_id     TEXT NOT NULL DEFAULT 'HOUSTON_HARVEY_2017',
    weights_path    TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'shadow' CHECK (status IN ('shadow', 'active', 'archived', 'rolled_back')),
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (model_version, adapter_version)
);

-- 10. Adaptation & Safety Gate Run History (Owner: Portion 4 Abdulrahman)
CREATE TABLE IF NOT EXISTS adaptation_runs (
    run_id          TEXT PRIMARY KEY,
    tier            TEXT NOT NULL,
    disaster_id     TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    trigger_reason  TEXT NOT NULL,
    drift_distance  REAL NOT NULL DEFAULT 0.0,
    gate_passed     INTEGER NOT NULL DEFAULT 0 CHECK (gate_passed IN (0, 1)),
    promoted        INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
    gate_report_json TEXT NOT NULL DEFAULT '{}',
    started_at      TEXT NOT NULL,
    completed_at    TEXT NOT NULL
);
"""

_ROAD_EDGE_MIGRATIONS = (
    ("closure_state", "INTEGER NOT NULL DEFAULT 0"),
    ("hazard_level", "REAL NOT NULL DEFAULT 0.0"),
    ("capacity", "REAL NOT NULL DEFAULT 1.0"),
    ("vehicle_restriction", "TEXT NOT NULL DEFAULT 'none'"),
)


@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with Row factory and per-connection FK pragma enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_road_edge_columns(conn: sqlite3.Connection) -> None:
    """Safely add Section 4.2 road attributes when opening pre-existing SQLite databases."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(road_edges)").fetchall()}
    for col_name, col_def in _ROAD_EDGE_MIGRATIONS:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE road_edges ADD COLUMN {col_name} {col_def}")
    # Deterministically backfill legacy BLOCKED/HAZARDOUS rows that still carry uninitialized static defaults
    conn.execute(
        """
        UPDATE road_edges
        SET closure_state = 1,
            hazard_level = MAX(hazard_level, flood_depth_m, debris_score),
            capacity = 0.0,
            vehicle_restriction = 'closed'
        WHERE UPPER(edge_status) = 'BLOCKED'
          AND closure_state = 0 AND hazard_level = 0.0 AND capacity = 1.0 AND vehicle_restriction = 'none'
        """
    )
    conn.execute(
        """
        UPDATE road_edges
        SET hazard_level = MAX(hazard_level, flood_depth_m, debris_score),
            capacity = 0.5,
            vehicle_restriction = 'high_clearance_only'
        WHERE UPPER(edge_status) = 'HAZARDOUS'
          AND closure_state = 0 AND hazard_level = 0.0 AND capacity = 1.0 AND vehicle_restriction = 'none'
        """
    )


def _ensure_building_damage_schema(conn: sqlite3.Connection) -> None:
    """Safely upgrade legacy integer buildings.damage_class to Section 3.2 enum strings."""
    info = {row["name"]: row["type"].upper() for row in conn.execute("PRAGMA table_info(buildings)").fetchall()}
    if not info:
        return
    if info.get("damage_class") == "TEXT" and "damage_tier" in info:
        return
    conn.executescript(
        """
        CREATE TABLE buildings_new (
            building_id              TEXT PRIMARY KEY,
            geom_geojson             TEXT NOT NULL,
            centroid_lat             REAL NOT NULL,
            centroid_lon             REAL NOT NULL,
            damage_class             TEXT NOT NULL CHECK (damage_class IN ('no_damage', 'minor', 'major', 'destroyed')),
            damage_tier              INTEGER NOT NULL DEFAULT 0 CHECK (damage_tier IN (0, 1, 2, 3)),
            class_label              TEXT NOT NULL CHECK (class_label IN ('no_damage', 'minor', 'major', 'destroyed', 'NO_DAMAGE', 'MINOR', 'MAJOR', 'DESTROYED')),
            confidence               REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            max_prob                 REAL NOT NULL DEFAULT 0.90,
            status_flag              TEXT NOT NULL DEFAULT 'CALIBRATED' CHECK (status_flag IN ('CALIBRATED', 'LOW_CONFIDENCE')),
            gradcam_attribution_json TEXT NOT NULL DEFAULT '[]',
            shap_top3_json           TEXT NOT NULL DEFAULT '[]',
            embedding_ref            TEXT NOT NULL DEFAULT '',
            last_updated             TEXT NOT NULL,
            data_source              TEXT NOT NULL DEFAULT 'xBD_houston_east'
        );
        INSERT INTO buildings_new (
            building_id, geom_geojson, centroid_lat, centroid_lon,
            damage_class, damage_tier, class_label, confidence, max_prob, status_flag,
            gradcam_attribution_json, shap_top3_json, embedding_ref, last_updated, data_source
        )
        SELECT
            building_id, geom_geojson, centroid_lat, centroid_lon,
            CASE CAST(damage_class AS TEXT)
                WHEN '0' THEN 'no_damage'
                WHEN '1' THEN 'minor'
                WHEN '2' THEN 'major'
                WHEN '3' THEN 'destroyed'
                ELSE LOWER(CAST(damage_class AS TEXT))
            END,
            CASE LOWER(CAST(damage_class AS TEXT))
                WHEN 'no_damage' THEN 0 WHEN '0' THEN 0
                WHEN 'minor' THEN 1 WHEN '1' THEN 1
                WHEN 'major' THEN 2 WHEN '2' THEN 2
                WHEN 'destroyed' THEN 3 WHEN '3' THEN 3
                ELSE 0
            END,
            LOWER(class_label), confidence, max_prob, status_flag,
            gradcam_attribution_json, shap_top3_json, embedding_ref, last_updated, data_source
        FROM buildings;
        DROP TABLE buildings;
        ALTER TABLE buildings_new RENAME TO buildings;
        """
    )


_FEEDBACK_COLUMN_MIGRATIONS = (
    ("severity_jump", "INTEGER NOT NULL DEFAULT 0"),
    ("requires_second_opinion", "INTEGER NOT NULL DEFAULT 0"),
    ("reliability_weight", "REAL NOT NULL DEFAULT 1.0"),
)


def _ensure_adaptation_schema(conn: sqlite3.Connection) -> None:
    """Safely upgrade pre-existing SQLite DBs with Portion 4 feedback/replay/registry tables."""
    fb_cols = {row["name"] for row in conn.execute("PRAGMA table_info(feedback_log)").fetchall()}
    if fb_cols:
        for col_name, col_def in _FEEDBACK_COLUMN_MIGRATIONS:
            if col_name not in fb_cols:
                conn.execute(f"ALTER TABLE feedback_log ADD COLUMN {col_name} {col_def}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS replay_buffer (
            sample_id       TEXT PRIMARY KEY,
            target_id       TEXT NOT NULL,
            disaster_id     TEXT NOT NULL,
            damage_class    TEXT NOT NULL CHECK (damage_class IN ('no_damage', 'minor', 'major', 'destroyed')),
            buffer_tier     TEXT NOT NULL CHECK (buffer_tier IN ('long_term', 'recent')),
            priority        REAL NOT NULL DEFAULT 1.0,
            uncertainty     REAL NOT NULL DEFAULT 0.0,
            embedding_ref   TEXT NOT NULL DEFAULT '',
            sample_json     TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            registry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version   TEXT NOT NULL,
            adapter_version TEXT NOT NULL,
            disaster_id     TEXT NOT NULL DEFAULT 'HOUSTON_HARVEY_2017',
            weights_path    TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'shadow' CHECK (status IN ('shadow', 'active', 'archived', 'rolled_back')),
            metrics_json    TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            UNIQUE (model_version, adapter_version)
        );
        CREATE TABLE IF NOT EXISTS adaptation_runs (
            run_id           TEXT PRIMARY KEY,
            tier             TEXT NOT NULL,
            disaster_id      TEXT NOT NULL,
            model_version    TEXT NOT NULL,
            adapter_version  TEXT NOT NULL,
            trigger_reason   TEXT NOT NULL,
            drift_distance   REAL NOT NULL DEFAULT 0.0,
            gate_passed      INTEGER NOT NULL DEFAULT 0 CHECK (gate_passed IN (0, 1)),
            promoted         INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
            gate_report_json TEXT NOT NULL DEFAULT '{}',
            started_at       TEXT NOT NULL,
            completed_at     TEXT NOT NULL
        );
        """
    )


def init_db(db_path: str = DEFAULT_DB_PATH, reset: bool = False) -> None:
    """Initialize all SQLite tables in WAL mode."""
    path = Path(db_path)
    if reset:
        for sidecar in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except PermissionError:
                    pass
    with get_connection(db_path) as conn:
        if reset and path.exists():
            # Fallback if an external handle prevented unlinking the file on Windows
            for tbl in (
                "buildings",
                "road_edges",
                "incidents",
                "routes",
                "explanations",
                "feedback_log",
                "audit_log",
                "replay_buffer",
                "model_registry",
                "adaptation_runs",
            ):
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        conn.executescript(SCHEMA_SQL)
        _ensure_building_damage_schema(conn)
        _ensure_road_edge_columns(conn)
        _ensure_adaptation_schema(conn)
        _ensure_audit_log_schema(conn)
        conn.commit()


# =====================================================================
# Cryptographic Audit Log Engine (Real SHA-256 Chaining)
# Formula from workflow/ahmed.pdf:
# "current hash equals SHA-256 of the previous hash concatenated with
# the route ID, the action, the actor, and the timestamp."
# =====================================================================

def compute_audit_hash(
    prev_hash: str,
    route_id: str,
    action: str,
    actor: str,
    timestamp: str,
) -> str:
    """Compute SHA-256(prev_hash + route_id + action + actor + timestamp)."""
    raw_str = f"{prev_hash}{route_id}{action}{actor}{timestamp}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def _compute_payload_hash(curr_hash: str, payload_json: str) -> str:
    """Bind canonical payload_json to curr_hash with SHA-256."""
    return hashlib.sha256(f"{curr_hash}:{payload_json}".encode("utf-8")).hexdigest()


def _ensure_audit_log_schema(conn: sqlite3.Connection) -> None:
    """Ensure audit_log has payload_hash column and backfill existing canonical rows."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
    if "payload_hash" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN payload_hash TEXT NOT NULL DEFAULT ''")
    rows = conn.execute(
        "SELECT seq_id, curr_hash, payload_json, payload_hash FROM audit_log WHERE payload_hash = '' OR payload_hash IS NULL"
    ).fetchall()
    for r in rows:
        p_hash = _compute_payload_hash(str(r["curr_hash"]), str(r["payload_json"]))
        conn.execute(
            "UPDATE audit_log SET payload_hash = ? WHERE seq_id = ?",
            (p_hash, int(r["seq_id"])),
        )


def append_audit_event(
    actor_role: str,
    event_type: str,
    payload: dict[str, Any],
    route_id: str = "",
    timestamp: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Append a tamper-evident SHA-256 hash-chained entry to audit_log."""
    utc_now = timestamp or datetime.now(timezone.utc).isoformat()
    if not route_id and "route_id" in payload:
        route_id = str(payload["route_id"])
    elif not route_id and "plan_id" in payload:
        route_id = str(payload["plan_id"])

    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    with get_connection(db_path) as conn:
        _ensure_audit_log_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.cursor()
        cur.execute("SELECT seq_id, curr_hash FROM audit_log ORDER BY seq_id DESC LIMIT 1")
        last_row = cur.fetchone()
        next_seq_id = 1 if last_row is None else int(last_row["seq_id"]) + 1
        prev_hash = GENESIS_PREV_HASH if last_row is None else str(last_row["curr_hash"])

        curr_hash = compute_audit_hash(
            prev_hash=prev_hash,
            route_id=route_id,
            action=event_type,
            actor=actor_role,
            timestamp=utc_now,
        )
        payload_hash = _compute_payload_hash(curr_hash, payload_str)

        cur.execute(
            """
            INSERT INTO audit_log (seq_id, utc_timestamp, actor_role, event_type, route_id, payload_json, payload_hash, prev_hash, curr_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (next_seq_id, utc_now, actor_role, event_type, route_id, payload_str, payload_hash, prev_hash, curr_hash),
        )
        conn.commit()

    return {
        "seq_id": next_seq_id,
        "utc_timestamp": utc_now,
        "actor_role": actor_role,
        "event_type": event_type,
        "route_id": route_id,
        "payload": payload,
        "payload_hash": payload_hash,
        "prev_hash": prev_hash,
        "curr_hash": curr_hash,
    }


def verify_audit_chain(db_path: str = DEFAULT_DB_PATH) -> tuple[bool, int, str]:
    """Verify integrity of the entire SHA-256 hash chain and payload_json bindings in audit_log."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY seq_id ASC").fetchall()

    if not rows:
        return True, 0, "Audit log is empty (Genesis)"

    expected_prev = GENESIS_PREV_HASH
    for idx, row in enumerate(rows, start=1):
        seq_id = int(row["seq_id"])
        if seq_id != idx:
            return False, idx, f"Sequence gap: expected {idx}, found {seq_id}"
        if row["prev_hash"] != expected_prev:
            return False, seq_id, f"Broken chain link at seq_id={seq_id}: prev_hash mismatch"

        recomputed = compute_audit_hash(
            prev_hash=row["prev_hash"],
            route_id=row["route_id"] if "route_id" in row.keys() else "",
            action=row["event_type"],
            actor=row["actor_role"],
            timestamp=row["utc_timestamp"],
        )
        if recomputed != row["curr_hash"]:
            return False, seq_id, f"Hash verification mismatch at seq_id={seq_id}"

        raw_payload = str(row["payload_json"] if "payload_json" in row.keys() else "")
        try:
            parsed_payload = json.loads(raw_payload)
        except Exception:
            return False, seq_id, f"Malformed payload_json at seq_id={seq_id}"
        if not isinstance(parsed_payload, dict):
            return False, seq_id, f"Non-object payload_json at seq_id={seq_id}"
        canonical_payload = json.dumps(parsed_payload, sort_keys=True, separators=(",", ":"))
        if raw_payload != canonical_payload:
            return False, seq_id, f"Payload canonicalization mismatch at seq_id={seq_id}"

        stored_payload_hash = str(row["payload_hash"] if "payload_hash" in row.keys() else "")
        if stored_payload_hash:
            expected_payload_hash = _compute_payload_hash(recomputed, raw_payload)
            if stored_payload_hash != expected_payload_hash:
                return False, seq_id, f"Payload hash verification mismatch at seq_id={seq_id}"

        expected_prev = recomputed

    return True, len(rows), "Audit chain verified intact"


def get_audit_log(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve audit log entries in reverse chronological order."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY seq_id DESC LIMIT ?", (limit,)
        ).fetchall()
        entries = []
        for r in rows:
            entry = dict(r)
            try:
                entry["payload"] = json.loads(entry["payload_json"])
            except Exception:
                entry["payload"] = {}
            entries.append(entry)
        return entries


def record_model_transition(
    model_version: str,
    adapter_version: str = "none",
    action: str = "MODEL_PROMOTED",
    gate_checks: dict[str, Any] | None = None,
    actor_role: str = "PORTION4:ADAPTATION",
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Record Stage 10 safety-gate model promotion or rollback in the SHA-256 audit chain."""
    return append_audit_event(
        actor_role=actor_role,
        event_type=action,
        payload={
            "model_version": model_version,
            "adapter_version": adapter_version,
            "gate_checks": gate_checks or {},
        },
        db_path=db_path,
    )


# =====================================================================
# Database Access Helper Functions for Team Members (No Raw SQL)
# =====================================================================

# --- 1. Building Helpers (Portion 1 Ammar & Consumers) ---

def insert_building(building: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace a Building object in the database."""
    now_utc = datetime.now(timezone.utc).isoformat()
    raw_dmg = building.get("damage_class", building.get("class_label", "no_damage"))
    dmg_tier, dmg_label = normalize_damage_class(raw_dmg)
    with get_connection(db_path) as conn:
        _ensure_building_damage_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO buildings (
                building_id, geom_geojson, centroid_lat, centroid_lon,
                damage_class, damage_tier, class_label, confidence, max_prob, status_flag,
                gradcam_attribution_json, shap_top3_json, embedding_ref, last_updated, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                building["building_id"],
                json.dumps(building.get("geometry", {})) if isinstance(building.get("geometry"), dict) else str(building.get("geom_geojson", "{}")),
                float(building.get("centroid_lat", 0.0)),
                float(building.get("centroid_lon", 0.0)),
                dmg_label,
                dmg_tier,
                dmg_label,
                float(building.get("confidence", 0.9)),
                float(building.get("max_prob", building.get("confidence", 0.9))),
                str(building.get("status_flag", "CALIBRATED")),
                json.dumps(building.get("gradcam_attribution", [])),
                json.dumps(building.get("shap_top3") if building.get("shap_top3") is not None else building.get("shap_attribution", [])),
                str(building.get("embedding_ref", "")),
                str(building.get("last_updated", now_utc)),
                str(building.get("data_source", "xBD_houston_east")),
            ),
        )
        conn.commit()


def _normalize_building_row(r: sqlite3.Row) -> dict[str, Any]:
    """Normalize a raw SQLite building row into a dictionary matching Building contract."""
    b = dict(r)
    tier, label = normalize_damage_class(b.get("damage_class", b.get("class_label", "no_damage")))
    b["damage_class"] = label
    b["class_label"] = label
    b["damage_tier"] = tier
    b["geometry"] = json.loads(b["geom_geojson"])
    b["gradcam_attribution"] = json.loads(b["gradcam_attribution_json"])
    raw_shap = json.loads(b["shap_top3_json"])
    norm_shap = []
    for s in raw_shap:
        val = float(s.get("value", s.get("impact", s.get("attribution", 0.0))))
        norm_shap.append({
            "feature": str(s.get("feature", "unknown")),
            "value": val,
            "impact": val,
            "attribution": val,
        })
    b["shap_top3"] = norm_shap
    b["shap_attribution"] = norm_shap
    conf_val = b.get("confidence")
    if conf_val is None:
        conf_val = b.get("max_prob", 0.9)
    b["confidence"] = float(conf_val)
    return b


def get_buildings(db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all Building records parsed into clean dictionaries."""
    with get_connection(db_path) as conn:
        _ensure_building_damage_schema(conn)
        rows = conn.execute("SELECT * FROM buildings ORDER BY building_id ASC").fetchall()
        return [_normalize_building_row(r) for r in rows]


def get_building(building_id: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieve a single Building record by building_id."""
    with get_connection(db_path) as conn:
        _ensure_building_damage_schema(conn)
        r = conn.execute("SELECT * FROM buildings WHERE building_id = ?", (building_id,)).fetchone()
        if r is None:
            return None
        return _normalize_building_row(r)


def get_damaged_buildings(min_damage_class: int | str = "major", db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve buildings with severity >= min_damage_class for debris halo modeling."""
    min_tier, _ = normalize_damage_class(min_damage_class)
    with get_connection(db_path) as conn:
        _ensure_building_damage_schema(conn)
        rows = conn.execute(
            "SELECT * FROM buildings WHERE damage_tier >= ? ORDER BY building_id ASC",
            (min_tier,),
        ).fetchall()
        return [_normalize_building_row(r) for r in rows]


def update_building_assessment(
    building_id: str,
    damage_class: int | str,
    class_label: str,
    max_prob: float,
    status_flag: str,
    data_source: str,
    shap_top3: list[dict[str, Any]],
    gradcam_attribution: list[dict[str, Any]] | None = None,
    embedding_ref: str | None = None,
    confidence: float | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Update building vision assessment fields (used by Portion 1 without raw SQL)."""
    now_utc = datetime.now(timezone.utc).isoformat()
    dmg_tier, dmg_label = normalize_damage_class(damage_class if damage_class not in (None, "") else class_label)
    conf_val = float(max_prob if confidence is None else confidence)
    with get_connection(db_path) as conn:
        _ensure_building_damage_schema(conn)
        cur = conn.cursor()
        if gradcam_attribution is not None and embedding_ref is not None:
            cur.execute(
                """
                UPDATE buildings
                SET damage_class = ?, damage_tier = ?, class_label = ?, confidence = ?, max_prob = ?,
                    status_flag = ?, data_source = ?, shap_top3_json = ?,
                    gradcam_attribution_json = ?, embedding_ref = ?, last_updated = ?
                WHERE building_id = ?
                """,
                (
                    dmg_label, dmg_tier, dmg_label, conf_val, max_prob, status_flag,
                    data_source, json.dumps(shap_top3),
                    json.dumps(gradcam_attribution), embedding_ref, now_utc, building_id,
                ),
            )
        elif gradcam_attribution is not None:
            cur.execute(
                """
                UPDATE buildings
                SET damage_class = ?, damage_tier = ?, class_label = ?, confidence = ?, max_prob = ?,
                    status_flag = ?, data_source = ?, shap_top3_json = ?,
                    gradcam_attribution_json = ?, last_updated = ?
                WHERE building_id = ?
                """,
                (
                    dmg_label, dmg_tier, dmg_label, conf_val, max_prob, status_flag,
                    data_source, json.dumps(shap_top3),
                    json.dumps(gradcam_attribution), now_utc, building_id,
                ),
            )
        elif embedding_ref is not None:
            cur.execute(
                """
                UPDATE buildings
                SET damage_class = ?, damage_tier = ?, class_label = ?, confidence = ?, max_prob = ?,
                    status_flag = ?, data_source = ?, shap_top3_json = ?,
                    embedding_ref = ?, last_updated = ?
                WHERE building_id = ?
                """,
                (
                    dmg_label, dmg_tier, dmg_label, conf_val, max_prob, status_flag,
                    data_source, json.dumps(shap_top3),
                    embedding_ref, now_utc, building_id,
                ),
            )
        else:
            cur.execute(
                """
                UPDATE buildings
                SET damage_class = ?, damage_tier = ?, class_label = ?, confidence = ?, max_prob = ?,
                    status_flag = ?, data_source = ?, shap_top3_json = ?, last_updated = ?
                WHERE building_id = ?
                """,
                (
                    dmg_label, dmg_tier, dmg_label, conf_val, max_prob, status_flag,
                    data_source, json.dumps(shap_top3), now_utc, building_id,
                ),
            )
        conn.commit()
        return cur.rowcount > 0


def update_building_label(
    building_id: str,
    damage_class: int | str,
    class_label: str = "",
    confidence: float = 1.0,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Update building damage classification after responder verification/correction."""
    now_utc = datetime.now(timezone.utc).isoformat()
    dmg_tier, dmg_label = normalize_damage_class(damage_class if damage_class not in (None, "") else class_label)
    with get_connection(db_path) as conn:
        _ensure_building_damage_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE buildings
            SET damage_class = ?, damage_tier = ?, class_label = ?, confidence = ?, max_prob = ?, status_flag = 'CALIBRATED', last_updated = ?
            WHERE building_id = ?
            """,
            (dmg_label, dmg_tier, dmg_label, confidence, confidence, now_utc, building_id),
        )
        conn.commit()
        return cur.rowcount > 0


# --- 1B. Road Edge Helpers (Portion 2 Negm & Consumers) ---

def insert_road_edge(edge: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace a road segment in road_edges without raw SQL."""
    geom_str = (
        json.dumps(edge["geometry"])
        if isinstance(edge.get("geometry"), dict)
        else str(edge.get("geom_geojson", "{}"))
    )
    flood_depth = float(edge.get("flood_depth_m", 0.0))
    debris_score = float(edge.get("debris_score", 0.0))
    raw_status = edge.get("edge_status")
    closure_default = bool(edge.get("closure_state", str(raw_status or "").upper() == "BLOCKED"))
    hazard_val = float(edge.get("hazard_level", max(flood_depth, debris_score)))
    if "capacity" in edge:
        capacity_val = float(edge["capacity"])
    elif str(raw_status or "").upper() == "BLOCKED":
        capacity_val = 0.0
    elif str(raw_status or "").upper() == "HAZARDOUS":
        capacity_val = 0.5
    else:
        capacity_val = 1.0
    restriction_val = str(edge.get("vehicle_restriction", "none"))
    status_str = (
        str(raw_status)
        if raw_status
        else derive_road_display_state(closure_default, hazard_val, capacity_val, restriction_val).upper()
    )
    with get_connection(db_path) as conn:
        _ensure_road_edge_columns(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO road_edges (
                edge_id, u_node, v_node, geom_geojson,
                length_m, base_travel_sec, flood_depth_m, debris_score,
                closure_state, hazard_level, capacity, vehicle_restriction, edge_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(edge["edge_id"]),
                int(edge["u_node"]),
                int(edge["v_node"]),
                geom_str,
                float(edge.get("length_m", 1200.0)),
                float(edge.get("base_travel_sec", 90.0)),
                flood_depth,
                debris_score,
                1 if closure_default else 0,
                hazard_val,
                capacity_val,
                restriction_val,
                status_str,
            ),
        )
        conn.commit()


def get_road_edges(status: str | None = None, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all road_edges records, optionally filtered by edge_status."""
    with get_connection(db_path) as conn:
        _ensure_road_edge_columns(conn)
        if status:
            rows = conn.execute(
                "SELECT * FROM road_edges WHERE UPPER(edge_status) = UPPER(?) ORDER BY edge_id ASC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM road_edges ORDER BY edge_id ASC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["geometry"] = json.loads(d["geom_geojson"])
            d["closure_state"] = bool(d.get("closure_state", 0))
            d["hazard_level"] = float(d.get("hazard_level", max(d.get("flood_depth_m", 0.0), d.get("debris_score", 0.0))))
            d["capacity"] = float(d.get("capacity", 1.0))
            d["vehicle_restriction"] = str(d.get("vehicle_restriction", "none"))
            result.append(d)
        return result


def update_road_edge(
    edge_id: str,
    flood_depth_m: float,
    debris_score: float,
    edge_status: str,
    closure_state: bool | None = None,
    hazard_level: float | None = None,
    capacity: float | None = None,
    vehicle_restriction: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Update hazard telemetry, Section 4.2 attributes, and status of a road edge."""
    with get_connection(db_path) as conn:
        _ensure_road_edge_columns(conn)
        cur = conn.cursor()
        set_clauses = ["flood_depth_m = ?", "debris_score = ?", "edge_status = ?"]
        params: list[Any] = [round(flood_depth_m, 3), round(debris_score, 3), edge_status]
        if closure_state is not None:
            set_clauses.append("closure_state = ?")
            params.append(1 if closure_state else 0)
        if hazard_level is not None:
            set_clauses.append("hazard_level = ?")
            params.append(round(hazard_level, 3))
        if capacity is not None:
            set_clauses.append("capacity = ?")
            params.append(round(capacity, 3))
        if vehicle_restriction is not None:
            set_clauses.append("vehicle_restriction = ?")
            params.append(vehicle_restriction)
        params.append(edge_id)
        cur.execute(
            f"UPDATE road_edges SET {', '.join(set_clauses)} WHERE edge_id = ?",
            tuple(params),
        )
        conn.commit()
        return cur.rowcount > 0


# --- 2. Incident Helpers (Portion 2 Negm & Consumers) ---

def insert_incident(incident: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace an Incident object."""
    loc = incident.get("location", {})
    lat = loc.get("lat", incident.get("location_lat", 0.0))
    lon = loc.get("lon", incident.get("location_lon", 0.0))
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO incidents (
                incident_id, location_lat, location_lon, severity, required_capability, time_reported, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident["incident_id"],
                float(lat),
                float(lon),
                str(incident.get("severity", "medium")),
                str(incident.get("required_capability", "rescue")),
                str(incident.get("time_reported", datetime.now(timezone.utc).isoformat())),
                str(incident.get("status", "open")),
            ),
        )
        conn.commit()


def get_incidents(status: str | None = None, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all Incidents, optionally filtered by status."""
    with get_connection(db_path) as conn:
        if status:
            rows = conn.execute("SELECT * FROM incidents WHERE status = ? ORDER BY time_reported DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM incidents ORDER BY time_reported DESC").fetchall()
        result = []
        for r in rows:
            inc = dict(r)
            inc["location"] = {"lat": inc["location_lat"], "lon": inc["location_lon"]}
            result.append(inc)
        return result


def get_incident(incident_id: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieve a single Incident by incident_id."""
    with get_connection(db_path) as conn:
        r = conn.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
        if not r:
            return None
        inc = dict(r)
        inc["location"] = {"lat": inc["location_lat"], "lon": inc["location_lon"]}
        return inc


def update_incident_status(incident_id: str, status: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Update status of an Incident (open, assigned, resolved)."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE incidents SET status = ? WHERE incident_id = ?", (status, incident_id))
        conn.commit()
        return cur.rowcount > 0


# --- 3. Route Helpers (Portion 2 Negm / Portion 3 Yousef / Portion 5 Ahmed) ---

def insert_route(route: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace a Route object in the database."""
    now_utc = datetime.now(timezone.utc).isoformat()
    route_id = str(route.get("route_id") or route.get("plan_id") or f"RT-{int(datetime.now().timestamp())}")
    plan_id = str(route.get("plan_id") or route_id)

    raw_status = route.get("status", route.get("approval_status", "pending_approval"))
    status_val = str(getattr(raw_status, "value", raw_status))
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO routes (
                plan_id, route_id, approval_status, unit_id, incident_id,
                ppo_proposed_path_json, masked_path_json, mask_altered, mask_reason,
                road_states_json, eta_seconds, status, explanation, uncertainty,
                shap_values_json, counterfactuals_json, audit_hash, created_utc,
                is_counterfactual, total_eta_sec, compute_ms, routes_json, narrative_text, narrative_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                route_id,
                status_val.upper(),
                str(route.get("unit_id", "MED-01")),
                str(route.get("incident_id", "INC-01")),
                json.dumps(route.get("ppo_proposed_path", [])),
                json.dumps(route.get("masked_path", [])),
                1 if route.get("mask_altered") else 0,
                str(route.get("mask_reason", "")),
                json.dumps(route.get("road_states", [])),
                float(route.get("eta_seconds", route.get("total_eta_sec", 0.0))),
                status_val.lower(),
                str(route.get("explanation", route.get("narrative_text", ""))),
                float(route.get("uncertainty", 0.0)),
                json.dumps(route.get("shap_values", [])),
                json.dumps(route.get("counterfactuals", [])),
                str(route.get("audit_hash", "")),
                str(route.get("created_utc", now_utc)),
                1 if route.get("is_counterfactual") else 0,
                float(route.get("total_eta_sec", route.get("eta_seconds", 0.0))),
                float(route.get("compute_ms", 0.0)),
                json.dumps(route.get("routes", [])),
                str(route.get("narrative_text", route.get("explanation", ""))),
                str(route.get("narrative_mode", "JINJA2_TEMPLATE")),
            ),
        )
        conn.commit()


def _normalize_route_row(r: sqlite3.Row) -> dict[str, Any]:
    """Normalize a raw SQLite route row into a clean dictionary matching Route contract."""
    rt = dict(r)
    rt["route_id"] = str(rt.get("route_id") or rt.get("plan_id") or "RT-01")
    rt["plan_id"] = str(rt.get("plan_id") or rt["route_id"])
    rt["ppo_proposed_path"] = json.loads(rt["ppo_proposed_path_json"])
    rt["masked_path"] = json.loads(rt["masked_path_json"])
    rt["road_states"] = json.loads(rt["road_states_json"])
    rt["shap_values"] = json.loads(rt["shap_values_json"])
    rt["counterfactuals"] = json.loads(rt["counterfactuals_json"])
    rt["routes"] = json.loads(rt["routes_json"])
    rt["mask_altered"] = bool(rt["mask_altered"])
    rt["eta_seconds"] = float(rt.get("eta_seconds") or rt.get("total_eta_sec") or 0.0)
    rt["total_eta_sec"] = float(rt.get("total_eta_sec") or rt["eta_seconds"] or 0.0)
    rt["explanation"] = str(rt.get("explanation") or rt.get("narrative_text") or "")
    rt["narrative_text"] = str(rt.get("narrative_text") or rt["explanation"] or "")
    rt["status"] = str(rt.get("status") or rt.get("approval_status") or "pending_approval").lower()
    return rt


def get_routes(include_counterfactuals: bool = False, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all Route objects parsed into dicts."""
    with get_connection(db_path) as conn:
        if include_counterfactuals:
            rows = conn.execute("SELECT * FROM routes ORDER BY created_utc DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM routes WHERE is_counterfactual = 0 ORDER BY created_utc DESC").fetchall()
        return [_normalize_route_row(r) for r in rows]


def get_route(route_id: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieve a single Route by route_id or plan_id."""
    with get_connection(db_path) as conn:
        r = conn.execute("SELECT * FROM routes WHERE route_id = ? OR plan_id = ?", (route_id, route_id)).fetchone()
        if not r:
            return None
        return _normalize_route_row(r)


def update_route_status(route_id: str, new_status: str, audit_hash: str = "", db_path: str = DEFAULT_DB_PATH) -> bool:
    """Update status of a route (approved, rejected, active, completed) and record audit hash."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE routes SET status = ?, approval_status = ?, audit_hash = ? WHERE route_id = ? OR plan_id = ?",
            (new_status.lower(), new_status.upper(), audit_hash, route_id, route_id),
        )
        conn.commit()
        return cur.rowcount > 0


def decide_route_with_audit(
    route_id: str,
    new_status: str,
    actor_role: str,
    notes: str = "",
    timestamp: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[str, dict[str, Any] | None]:
    """Atomically enforce pending_approval -> approved/rejected, append SHA-256 audit entry, and update route."""
    utc_now = timestamp or datetime.now(timezone.utc).isoformat()
    norm_status = new_status.lower()
    action = norm_status.upper()
    payload = {"route_id": route_id, "notes": notes, "status": norm_status}
    payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    with get_connection(db_path) as conn:
        _ensure_audit_log_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.cursor()
        route_row = cur.execute(
            "SELECT status, approval_status FROM routes WHERE route_id = ? OR plan_id = ?",
            (route_id, route_id),
        ).fetchone()
        if route_row is None:
            conn.rollback()
            return "NOT_FOUND", None

        current_status = str(route_row["status"] or route_row["approval_status"] or "pending_approval").lower()
        if current_status != "pending_approval":
            conn.rollback()
            return f"CONFLICT:{current_status}", None

        last_row = cur.execute("SELECT seq_id, curr_hash FROM audit_log ORDER BY seq_id DESC LIMIT 1").fetchone()
        next_seq_id = 1 if last_row is None else int(last_row["seq_id"]) + 1
        prev_hash = GENESIS_PREV_HASH if last_row is None else str(last_row["curr_hash"])
        curr_hash = compute_audit_hash(
            prev_hash=prev_hash,
            route_id=route_id,
            action=action,
            actor=actor_role,
            timestamp=utc_now,
        )
        payload_hash = _compute_payload_hash(curr_hash, payload_str)

        cur.execute(
            """
            INSERT INTO audit_log (seq_id, utc_timestamp, actor_role, event_type, route_id, payload_json, payload_hash, prev_hash, curr_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (next_seq_id, utc_now, actor_role, action, route_id, payload_str, payload_hash, prev_hash, curr_hash),
        )
        cur.execute(
            "UPDATE routes SET status = ?, approval_status = ?, audit_hash = ? WHERE route_id = ? OR plan_id = ?",
            (norm_status, action, curr_hash, route_id, route_id),
        )
        conn.commit()

    return "OK", {
        "seq_id": next_seq_id,
        "utc_timestamp": utc_now,
        "actor_role": actor_role,
        "event_type": action,
        "route_id": route_id,
        "payload": payload,
        "prev_hash": prev_hash,
        "curr_hash": curr_hash,
    }


# --- 4. Explanation Helpers (Portion 3 Yousef & Consumers) ---

def insert_explanation(explanation: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace an Explanation object."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO explanations (
                explanation_id, target_type, target_id, natural_language,
                attribution_method, system_uncertainty, llm_used, validation_passed, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                explanation["explanation_id"],
                str(explanation["target_type"]),
                str(explanation["target_id"]),
                str(explanation["natural_language"]),
                str(explanation["attribution_method"]),
                float(explanation.get("system_uncertainty", 0.0)),
                1 if explanation.get("llm_used") else 0,
                1 if explanation.get("validation_passed", True) else 0,
                str(explanation.get("timestamp", datetime.now(timezone.utc).isoformat())),
            ),
        )
        conn.commit()


def _normalize_explanation_row(r: sqlite3.Row) -> dict[str, Any]:
    """Normalize SQLite integer flags to booleans on Explanation records."""
    d = dict(r)
    d["llm_used"] = bool(d.get("llm_used", 0))
    d["validation_passed"] = bool(d.get("validation_passed", 1))
    return d


def get_explanations(target_id: str | None = None, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve Explanation objects, optionally filtered by target_id."""
    with get_connection(db_path) as conn:
        if target_id:
            rows = conn.execute("SELECT * FROM explanations WHERE target_id = ? ORDER BY timestamp DESC", (target_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM explanations ORDER BY timestamp DESC").fetchall()
        return [_normalize_explanation_row(r) for r in rows]


def get_explanation_by_target(target_type: str, target_id: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieve the latest explanation for a specific building or route."""
    with get_connection(db_path) as conn:
        r = conn.execute(
            "SELECT * FROM explanations WHERE target_type = ? AND target_id = ? ORDER BY timestamp DESC LIMIT 1",
            (target_type, target_id),
        ).fetchone()
        return _normalize_explanation_row(r) if r else None


# --- 5. Feedback Log Helpers (Portion 4 Abdulrahman & Consumers) ---

def _normalize_utc_iso(raw_ts: Any, fallback_utc: str | None = None) -> str:
    """Normalize ISO-8601 timestamps (Z or offset-aware) into canonical UTC ISO-8601 strings ending in Z."""
    if not raw_ts:
        if fallback_utc:
            return fallback_utc
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(raw_ts).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return text


def log_feedback(feedback: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Record responder 1-tap feedback (Confirm / Correct / Skip) in SQLite WAL queue."""
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    recorded_ts = _normalize_utc_iso(feedback.get("timestamp"), fallback_utc=now_utc)
    raw_act = feedback.get("responder_action", "confirm")
    action_str = getattr(raw_act, "value", str(raw_act)).lower().split(".")[-1]
    severity_jump = int(feedback.get("severity_jump", 0))
    requires_second_opinion = 1 if bool(feedback.get("requires_second_opinion", False)) else 0
    reliability_weight = float(feedback.get("reliability_weight", 1.0))

    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback_log (
                target_id, disaster_id, model_version, adapter_version,
                prediction, uncertainty, responder_action, corrected_label,
                responder_id, time_taken_sec, timestamp,
                severity_jump, requires_second_opinion, reliability_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(feedback["target_id"]),
                str(feedback.get("disaster_id", "HOUSTON_HARVEY_2017")),
                str(feedback.get("model_version", "ConvNeXt-Tiny-v1.0")),
                str(feedback.get("adapter_version", "none")),
                str(feedback.get("prediction", "")),
                float(feedback.get("uncertainty", 0.0)),
                action_str,
                str(feedback.get("corrected_label", "")),
                str(feedback.get("responder_id", "FIELD_RESPONDER_01")),
                float(feedback.get("time_taken_sec", 1.5)),
                recorded_ts,
                severity_jump,
                requires_second_opinion,
                reliability_weight,
            ),
        )
        conn.commit()
        feedback_id = cur.lastrowid

    return {
        "feedback_id": feedback_id,
        "target_id": feedback["target_id"],
        "responder_action": action_str,
        "timestamp": recorded_ts,
        "status": "queued_offline_wal",
        "severity_jump": severity_jump,
        "requires_second_opinion": bool(requires_second_opinion),
        "reliability_weight": reliability_weight,
    }


def get_feedback_logs(
    limit: int = 100,
    target_id: str | None = None,
    action: str | None = None,
    since_utc: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve recent responder feedback records with optional filtering."""
    clauses: list[str] = []
    args: list[Any] = []
    if target_id:
        clauses.append("target_id = ?")
        args.append(str(target_id))
    if action:
        clauses.append("LOWER(responder_action) = ?")
        args.append(str(action).strip().lower())
    if since_utc:
        clauses.append("julianday(timestamp) >= julianday(?)")
        args.append(_normalize_utc_iso(since_utc))
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    safe_limit = max(1, min(int(limit), 5000))
    args.append(safe_limit)
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        rows = conn.execute(
            f"SELECT * FROM feedback_log{where_sql} ORDER BY feedback_id DESC LIMIT ?",
            args,
        ).fetchall()
        return [dict(r) for r in rows]


def count_responder_recent_actions(
    responder_id: str,
    since_utc: str,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Count feedback actions recorded by a responder since a UTC ISO-8601 timestamp."""
    since_norm = _normalize_utc_iso(since_utc)
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        row = conn.execute(
            "SELECT COUNT(*) FROM feedback_log WHERE responder_id = ? AND julianday(timestamp) >= julianday(?)",
            (str(responder_id), since_norm),
        ).fetchone()
        return int(row[0]) if row else 0


def get_feedback_stats(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Retrieve feedback summary statistics (count, actions breakdown, drift indicators)."""
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        total = conn.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        confirmed = conn.execute("SELECT COUNT(*) FROM feedback_log WHERE responder_action = 'confirm'").fetchone()[0]
        corrected = conn.execute("SELECT COUNT(*) FROM feedback_log WHERE responder_action = 'correct'").fetchone()[0]
        skipped = conn.execute("SELECT COUNT(*) FROM feedback_log WHERE responder_action = 'skip'").fetchone()[0]
        return {
            "total_count": total,
            "confirmed": confirmed,
            "corrected": corrected,
            "skipped": skipped,
            "correction_rate": round(corrected / total, 3) if total > 0 else 0.0,
        }


def get_second_opinion_count(db_path: str = DEFAULT_DB_PATH) -> int:
    """Count logged feedback corrections flagged for a second opinion (>=2 damage-tier jump)."""
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        row = conn.execute("SELECT COUNT(*) FROM feedback_log WHERE requires_second_opinion = 1").fetchone()
        return int(row[0]) if row else 0


# --- 6. Dual-Buffer Replay Memory & Model Registry Helpers (Portion 4) ---

def upsert_replay_sample(sample: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> None:
    """Insert or replace a replay buffer sample without raw SQL in Portion 4."""
    _, canonical_cls = normalize_damage_class(sample["damage_class"])
    now_utc = datetime.now(timezone.utc).isoformat()
    sample_json = (
        json.dumps(sample["sample_json"])
        if isinstance(sample.get("sample_json"), dict)
        else str(sample.get("sample_json", "{}"))
    )
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO replay_buffer (
                sample_id, target_id, disaster_id, damage_class, buffer_tier,
                priority, uncertainty, embedding_ref, sample_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(sample["sample_id"]),
                str(sample["target_id"]),
                str(sample.get("disaster_id", "HOUSTON_HARVEY_2017")),
                canonical_cls,
                str(sample.get("buffer_tier", "recent")),
                float(sample.get("priority", 1.0)),
                float(sample.get("uncertainty", 0.0)),
                str(sample.get("embedding_ref", "")),
                sample_json,
                str(sample.get("created_at") or now_utc),
            ),
        )
        conn.commit()


def delete_replay_samples(sample_ids: list[str], db_path: str = DEFAULT_DB_PATH) -> int:
    """Delete specific replay sample IDs (used for stratified reservoir or recent cap maintenance)."""
    if not sample_ids:
        return 0
    placeholders = ",".join("?" for _ in sample_ids)
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        cur = conn.execute(
            f"DELETE FROM replay_buffer WHERE sample_id IN ({placeholders})",
            [str(sid) for sid in sample_ids],
        )
        conn.commit()
        return int(cur.rowcount)


def get_replay_samples(
    buffer_tier: str | None = None,
    disaster_id: str | None = None,
    damage_class: str | None = None,
    limit: int = 1000,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve replay buffer samples with optional stratification filters."""
    clauses: list[str] = []
    args: list[Any] = []
    if buffer_tier:
        clauses.append("buffer_tier = ?")
        args.append(str(buffer_tier))
    if disaster_id:
        clauses.append("disaster_id = ?")
        args.append(str(disaster_id))
    if damage_class:
        _, canonical_cls = normalize_damage_class(damage_class)
        clauses.append("damage_class = ?")
        args.append(canonical_cls)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    safe_limit = max(1, min(int(limit), 5000))
    args.append(safe_limit)
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        rows = conn.execute(
            f"SELECT * FROM replay_buffer{where_sql} ORDER BY created_at DESC, rowid DESC LIMIT ?",
            args,
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            try:
                item["metadata"] = json.loads(item["sample_json"])
            except Exception:
                item["metadata"] = {}
            out.append(item)
        return out


def register_model_version(entry: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Register or update a model/adapter version entry in model_registry."""
    now_utc = datetime.now(timezone.utc).isoformat()
    metrics_str = (
        json.dumps(entry["metrics"])
        if isinstance(entry.get("metrics"), dict)
        else str(entry.get("metrics_json", "{}"))
    )
    model_version = str(entry["model_version"])
    adapter_version = str(entry.get("adapter_version", "none"))
    disaster_id = str(entry.get("disaster_id", "HOUSTON_HARVEY_2017"))
    weights_path = str(entry.get("weights_path", ""))
    status_val = str(entry.get("status", "shadow"))

    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        conn.execute(
            """
            INSERT INTO model_registry (
                model_version, adapter_version, disaster_id, weights_path,
                status, metrics_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_version, adapter_version) DO UPDATE SET
                disaster_id = excluded.disaster_id,
                weights_path = excluded.weights_path,
                status = excluded.status,
                metrics_json = excluded.metrics_json,
                updated_at = excluded.updated_at
            """,
            (
                model_version,
                adapter_version,
                disaster_id,
                weights_path,
                status_val,
                metrics_str,
                str(entry.get("created_at") or now_utc),
                now_utc,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM model_registry WHERE model_version = ? AND adapter_version = ?",
            (model_version, adapter_version),
        ).fetchone()
        res = dict(row) if row else {}
        try:
            res["metrics"] = json.loads(res.get("metrics_json", "{}"))
        except Exception:
            res["metrics"] = {}
        return res


def set_active_model_version(
    model_version: str,
    adapter_version: str,
    db_path: str = DEFAULT_DB_PATH,
    archive_status: str = "archived",
) -> dict[str, Any] | None:
    """Atomically demote any previous active adapter and activate the specified version."""
    now_utc = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            target = conn.execute(
                "SELECT * FROM model_registry WHERE model_version = ? AND adapter_version = ?",
                (str(model_version), str(adapter_version)),
            ).fetchone()
            if not target:
                conn.rollback()
                return None
            conn.execute(
                "UPDATE model_registry SET status = ?, updated_at = ? WHERE status = 'active' AND NOT (model_version = ? AND adapter_version = ?)",
                (str(archive_status), now_utc, str(model_version), str(adapter_version)),
            )
            conn.execute(
                "UPDATE model_registry SET status = 'active', updated_at = ? WHERE model_version = ? AND adapter_version = ?",
                (now_utc, str(model_version), str(adapter_version)),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM model_registry WHERE model_version = ? AND adapter_version = ?",
                (str(model_version), str(adapter_version)),
            ).fetchone()
            res = dict(updated) if updated else {}
            try:
                res["metrics"] = json.loads(res.get("metrics_json", "{}"))
            except Exception:
                res["metrics"] = {}
            return res
        except Exception:
            conn.rollback()
            raise


def get_active_model_version(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return the currently active model/adapter version row, or None if base weights only."""
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        row = conn.execute(
            "SELECT * FROM model_registry WHERE status = 'active' ORDER BY updated_at DESC, registry_id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["metrics"] = json.loads(res.get("metrics_json", "{}"))
        except Exception:
            res["metrics"] = {}
        return res


def list_model_versions(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """List registered model/adapter versions ordered most-recently-updated first."""
    safe_limit = max(1, min(int(limit), 500))
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        rows = conn.execute(
            "SELECT * FROM model_registry ORDER BY updated_at DESC, registry_id DESC LIMIT ?",
            (safe_limit,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            try:
                item["metrics"] = json.loads(item.get("metrics_json", "{}"))
            except Exception:
                item["metrics"] = {}
            out.append(item)
        return out


def record_adaptation_run(run: dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Persist a Stage 9/10 adaptation and safety gate run record."""
    now_utc = datetime.now(timezone.utc).isoformat()
    report_str = (
        json.dumps(run["gate_report"])
        if isinstance(run.get("gate_report"), dict)
        else str(run.get("gate_report_json", "{}"))
    )
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO adaptation_runs (
                run_id, tier, disaster_id, model_version, adapter_version,
                trigger_reason, drift_distance, gate_passed, promoted,
                gate_report_json, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run["run_id"]),
                str(run.get("tier", "tier_a")),
                str(run.get("disaster_id", "HOUSTON_HARVEY_2017")),
                str(run.get("model_version", "ResNet18-UNet-xBD-v1.0")),
                str(run.get("adapter_version", "none")),
                str(run.get("trigger_reason", "manual")),
                float(run.get("drift_distance", 0.0)),
                1 if bool(run.get("gate_passed", False)) else 0,
                1 if bool(run.get("promoted", False)) else 0,
                report_str,
                str(run.get("started_at") or now_utc),
                str(run.get("completed_at") or now_utc),
            ),
        )
        conn.commit()
    return dict(run)


def get_adaptation_runs(limit: int = 25, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve recent adaptation/safety-gate run records."""
    with get_connection(db_path) as conn:
        _ensure_adaptation_schema(conn)
        rows = conn.execute(
            "SELECT * FROM adaptation_runs ORDER BY completed_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            try:
                item["gate_report"] = json.loads(item.get("gate_report_json", "{}"))
            except Exception:
                item["gate_report"] = {}
            out.append(item)
        return out

