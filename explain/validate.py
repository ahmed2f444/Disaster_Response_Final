"""Portion 3 (Yousef) — Ground-Truth-First (GTF) Prohibition Validator & Template Fallback.

Implements Portion 3's half of the Unified System Document Section 7 shared LLM constraint
(`Mismatch 3`), complementing Portion 5's Pydantic + ETA Validation Gate in `main.py`:
- Enforces the GTF prohibition list (no certainty hype, hedging, PPO override, `%`, or unverified numbers).
- Verifies route ETA, known edge IDs, `mask_altered` consistency, and building damage labels.
- Provides deterministic `template_fallback(...)` replacements when validation fails or LLM is offline.
"""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_GTF_PHRASES: tuple[str, ...] = (
    "definitely",
    "highly confident",
    "highly",
    "guaranteed",
    "absolutely",
    "certainly",
    "certain",
    "undoubtedly",
    "very ",
    "confidence",
    "certainty",
    "confident",
)

_HEDGE_PATTERN = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|might\s+be|could\s+be|likely|presumably|i\s+think|i\s+believe)\b",
    re.IGNORECASE,
)
_OVERRIDE_PATTERN = re.compile(
    r"\b(?:override|disregard|ignore\s+(?:the\s+)?ppo|bypass\s+(?:the\s+)?(?:safety\s+)?mask)\b",
    re.IGNORECASE,
)
_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_ETA_PATTERN = re.compile(
    r"(?:eta\s*(?:is|of|[:=])?\s*)(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
    re.IGNORECASE,
)
_EDGE_PATTERN = re.compile(r"\bE-\d+\b")
_ENTITY_ID_PATTERN = re.compile(
    r"\b(?:E|BLD|INC|PLAN|RT|WHATIF|EXP|MED|TRN|RSC|TASK|RESPONDER|CMD|SUP|SYN)-[A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")
_MASK_NO_CHANGE_PATTERN = re.compile(
    r"\b(?:mask\s+did\s+not\s+alter|not\s+altered\s+by\s+(?:the\s+)?mask|unchanged\s+by\s+(?:the\s+)?mask)\b",
    re.IGNORECASE,
)
_MASK_CHANGED_PATTERN = re.compile(
    r"\b(?:(?:safety\s+)?mask\s+altered\s+the|diverted\s+by\s+the\s+safety\s+mask)\b",
    re.IGNORECASE,
)
_DAMAGE_LABEL_PATTERNS: dict[str, str] = {
    "no_damage": r"\bno(?:[\s_]+damage)\b",
    "minor": r"\bminor\b",
    "major": r"\bmajor\b",
    "destroyed": r"\bdestroyed\b",
}


def coerce_shap_dict(shap_values: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, float]:
    """Normalize dict or `List[ShapAttribution]` into `{feature_name: float_score}`."""
    if not shap_values:
        return {}
    if isinstance(shap_values, dict):
        return {str(k): float(v) for k, v in shap_values.items()}
    out: dict[str, float] = {}
    for item in shap_values:
        if isinstance(item, dict) and "feature" in item:
            val = item.get("value", item.get("impact", item.get("attribution", 0.0)))
            out[str(item["feature"])] = float(val)
    return out


def _collect_allowed_numbers(
    shap_dict: dict[str, float],
    uncertainty: float,
    route: dict[str, Any] | None,
    building: dict[str, Any] | None,
    gradcam_info: dict[str, Any] | None,
    extra_allowed: list[float] | tuple[float, ...] | set[float] | None,
) -> list[tuple[float, float]]:
    """Build `(value, tolerance)` pairs of grounded numbers permitted in LLM output."""
    allowed: list[tuple[float, float]] = [
        (float(uncertainty), 0.02),
        (abs(float(uncertainty)), 0.02),
    ]
    for v in shap_dict.values():
        fv = float(v)
        allowed.append((fv, 0.02))
        allowed.append((abs(fv), 0.02))

    if isinstance(route, dict):
        for key in ("eta_seconds", "total_eta_sec", "whatif_eta_sec", "baseline_eta_sec"):
            if route.get(key) is not None:
                allowed.append((float(route[key]), 0.5))
        for key in ("delta_sec", "eta_delta_sec", "flood_delta", "flood_delta_m", "uncertainty"):
            if route.get(key) is not None:
                val = float(route[key])
                allowed.append((val, 0.15))
                allowed.append((abs(val), 0.15))
        for item in route.get("shap_values", []) or []:
            if isinstance(item, dict):
                for sk in ("value", "impact", "attribution", "raw_shap"):
                    if item.get(sk) is not None:
                        allowed.append((float(item[sk]), 0.02))
                        allowed.append((abs(float(item[sk])), 0.02))
        if isinstance(route.get("routes"), list) and route["routes"]:
            allowed.append((float(len(route["routes"])), 0.01))
        for cf in route.get("counterfactuals", []) or []:
            if isinstance(cf, dict) and cf.get("delta_eta") is not None:
                allowed.append((float(cf["delta_eta"]), 0.5))
                allowed.append((abs(float(cf["delta_eta"])), 0.5))

    if isinstance(building, dict):
        if building.get("confidence") is not None:
            conf = float(building["confidence"])
            allowed.append((conf, 0.02))
            allowed.append((max(0.0, 1.0 - conf), 0.02))
        for reg in building.get("gradcam_attribution", []) or []:
            if isinstance(reg, dict) and reg.get("weight") is not None:
                allowed.append((float(reg["weight"]), 0.02))

    if isinstance(gradcam_info, dict):
        for key in ("peak_saliency", "mean_saliency", "active_fraction"):
            if gradcam_info.get(key) is not None:
                allowed.append((float(gradcam_info[key]), 0.02))
        for reg in gradcam_info.get("regions", []) or []:
            if isinstance(reg, dict) and reg.get("weight") is not None:
                allowed.append((float(reg["weight"]), 0.02))

    if extra_allowed:
        for num in extra_allowed:
            allowed.append((float(num), 0.05))

    return allowed


def _is_number_grounded(val: float, allowed: list[tuple[float, float]]) -> bool:
    return any(abs(val - ref) <= tol for ref, tol in allowed)


def _check_gtf_prohibitions(text: str, llm_used: bool) -> tuple[bool, str]:
    lowered = text.lower()
    if llm_used:
        for phrase in FORBIDDEN_GTF_PHRASES:
            if re.search(rf"\b{re.escape(phrase.strip())}", lowered):
                return False, f"Forbidden GTF phrase detected: {phrase.strip()}"
        if _PCT_PATTERN.search(text):
            return False, "LLM output must not emit percentage confidence claims."
        if _HEDGE_PATTERN.search(text):
            return False, "Forbidden hedging language detected."
    if _OVERRIDE_PATTERN.search(text):
        return False, "Forbidden PPO/mask override directive detected."
    return True, "OK"


def _check_route_facts(text: str, route: dict[str, Any], llm_used: bool) -> tuple[bool, str]:
    expected_eta = float(route.get("eta_seconds") or route.get("whatif_eta_sec") or route.get("total_eta_sec") or 0.0)
    if expected_eta > 0.0:
        stated_etas = [float(m) for m in _ETA_PATTERN.findall(text)]
        if stated_etas and not all(abs(val - expected_eta) <= 0.5 for val in stated_etas):
            return False, f"ETA mismatch against computed route ({expected_eta:.1f}s)."

    known_edges: set[str] = {str(rs.get("edge_id")) for rs in route.get("road_states", []) if rs.get("edge_id")}
    for sub in route.get("routes", []) or []:
        if isinstance(sub, dict):
            known_edges.update(str(e) for e in sub.get("edge_ids", []))
            known_edges.update(str(e) for e in sub.get("ppo_edge_ids", []))
    known_edges.update(str(e) for e in route.get("newly_blocked_edges", []) or [])
    known_edges.update(str(e) for e in route.get("blocked_edges", []) or [])

    if known_edges:
        cited_edges = set(_EDGE_PATTERN.findall(text))
        if cited_edges and not cited_edges.issubset(known_edges):
            unknown = sorted(cited_edges - known_edges)
            return False, f"Unverified road edge IDs cited: {unknown}"

    if llm_used and "mask_altered" in route:
        if route["mask_altered"] and _MASK_NO_CHANGE_PATTERN.search(text):
            return False, "Narrative denies safety mask alteration when mask_altered is True."
        if (not route["mask_altered"]) and _MASK_CHANGED_PATTERN.search(text):
            return False, "Narrative claims safety mask alteration when mask_altered is False."

    return True, "OK"


def _check_building_facts(text: str, building: dict[str, Any], llm_used: bool) -> tuple[bool, str]:
    dmg_lbl = str(building.get("damage_class", building.get("class_label", "minor"))).lower()
    lowered = text.lower()
    for other, pat in _DAMAGE_LABEL_PATTERNS.items():
        if other == dmg_lbl:
            continue
        for match in re.finditer(pat, lowered):
            prefix = lowered[max(0, match.start() - 64) : match.start()]
            if not re.search(r"\b(?:not|no|neither|nor|unlike|without|non)\b[^.!?;]*$", prefix):
                return False, f"Contradictory damage class '{other}' asserted for '{dmg_lbl}' building."

    if not llm_used:
        conf = float(building.get("confidence", 0.85))
        stated_pcts = [float(m) for m in _PCT_PATTERN.findall(text)]
        if stated_pcts and not all(abs(p - conf * 100.0) <= 2.0 for p in stated_pcts):
            return False, "Building percentage mismatch against calibrated confidence."

    return True, "OK"


def validate_llm_output(
    llm_text: str,
    shap_values: dict[str, Any] | list[dict[str, Any]] | None = None,
    mask_reason: str = "",
    uncertainty: float = 0.0,
    *,
    route: dict[str, Any] | None = None,
    building: dict[str, Any] | None = None,
    gradcam_info: dict[str, Any] | None = None,
    llm_used: bool = True,
    allowed_numbers: list[float] | tuple[float, ...] | set[float] | None = None,
) -> tuple[bool, str]:
    """Validate explanation text against Unified System Document Section 7 GTF constraints."""
    text = (llm_text or "").strip()
    if len(text) < 10:
        return False, "Explanation text is empty or too short."

    gtf_ok, gtf_msg = _check_gtf_prohibitions(text, llm_used=llm_used)
    if not gtf_ok:
        return False, gtf_msg

    if route is not None:
        rt_ok, rt_msg = _check_route_facts(text, route, llm_used=llm_used)
        if not rt_ok:
            return False, rt_msg

    if building is not None:
        bld_ok, bld_msg = _check_building_facts(text, building, llm_used=llm_used)
        if not bld_ok:
            return False, bld_msg

    if llm_used:
        shap_dict = coerce_shap_dict(shap_values)
        if not shap_dict and isinstance(route, dict):
            shap_dict = coerce_shap_dict(route.get("shap_values"))
        allowed = _collect_allowed_numbers(
            shap_dict=shap_dict,
            uncertainty=uncertainty,
            route=route,
            building=building,
            gradcam_info=gradcam_info,
            extra_allowed=allowed_numbers,
        )
        scrubbed = _ENTITY_ID_PATTERN.sub(" ", text)
        for num_str in _NUMBER_PATTERN.findall(scrubbed):
            try:
                num_val = float(num_str)
            except ValueError:
                continue
            if not _is_number_grounded(num_val, allowed):
                return False, f"Unverified number: {num_str}"

    return True, "Valid"


def template_fallback(
    shap_values: dict[str, Any] | list[dict[str, Any]] | None = None,
    mask_altered: bool = False,
    mask_reason: str = "",
    uncertainty: float = 0.0,
    *,
    route: dict[str, Any] | None = None,
    building: dict[str, Any] | None = None,
    gradcam_summary: str | None = None,
) -> str:
    """Deterministic GTF-compliant explanation replacement when LLM fails or is disabled."""
    if building is not None:
        bld_id = str(building.get("building_id") or "BLD")
        dmg_lbl = str(building.get("damage_class", building.get("class_label", "minor"))).upper()
        conf = float(building.get("confidence", max(0.0, 1.0 - float(uncertainty))))
        focus = (gradcam_summary or "").strip()
        focus_clause = f" {focus}" if focus else ""
        return (
            f"[TEMPLATE FALLBACK] Structural assessment for {bld_id} ({dmg_lbl}). "
            f"Calibrated confidence: {conf * 100:.1f}%.{focus_clause}"
        )

    shap_dict = coerce_shap_dict(shap_values)
    if not shap_dict and isinstance(route, dict):
        shap_dict = coerce_shap_dict(route.get("shap_values"))
    top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    features_text = ", ".join(f"{name} ({val:+.2f})" for name, val in top_features) or "routing_factors"

    if route is not None:
        route_id = str(route.get("route_id") or route.get("plan_id") or "PLAN")
        eta = float(route.get("eta_seconds") or route.get("whatif_eta_sec") or route.get("total_eta_sec") or 0.0)
        altered = bool(route.get("mask_altered", mask_altered))
        reason = str(route.get("mask_reason") or mask_reason or "Safety mask active")
        if altered:
            mask_clause = f"Safety mask altered the PPO proposal ({reason})."
        else:
            mask_clause = "Safety mask did not alter the PPO proposal."
        return (
            f"[TEMPLATE FALLBACK] Route {route_id} follows safety-filtered corridor "
            f"(ETA {eta:.1f}s). Top SHAP factors: {features_text}. {mask_clause}"
        )

    mask_clause = (
        f"Safety filter altered route: True ({mask_reason or 'constraint active'})."
        if mask_altered
        else "Safety filter altered route: False."
    )
    return f"[TEMPLATE FALLBACK] Top contributing factors: {features_text}. {mask_clause} System uncertainty: {uncertainty:.2f}."
