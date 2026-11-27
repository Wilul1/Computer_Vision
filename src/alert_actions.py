"""Patient monitoring status from CV signals (Phase 1A).

Status is computed from *current* signals only — never from event ACK/RESOLVE.
Lying alone is not a fall; fall_alert uses temporal drop heuristics → "possible fall".

Priority (highest first):
  1. camera_unavailable → ATTENTION
  2. possible_fall (fall_alert) → ALERT
  3. inactive → ATTENTION
  4. framing: out_of_view / partial_body → ATTENTION
  5. otherwise → SAFE
"""
from __future__ import annotations

from . import config

ALERT_DIR = config.OUTPUTS_DIR / "alerts"
STATUS_COLORS = {
    "SAFE": "#26734A",
    "ATTENTION": "#8A4A00",
    "ALERT": "#B42318",
}


def decide(r: dict | None = None, *, camera_unavailable: bool = False) -> dict:
    """Map pipeline signals → monitoring status + caregiver action.

    Returns keys: status, trigger, event_type, action, action_code, reason
    """
    if camera_unavailable or r is None:
        return {
            "status": "ATTENTION",
            "trigger": "camera_unavailable",
            "event_type": "camera_unavailable",
            "action": "Check camera — feed unavailable",
            "action_code": "FIX_CAMERA",
            "reason": "Camera unavailable or no frame received",
        }

    # 1) Possible fall (temporal heuristic — not "lying" alone)
    if r.get("fall_alert"):
        detail = r.get("fall_reason") or "sudden drop / tilt into lying"
        if detail == "alert hold":
            detail = "possible fall hold"
        return {
            "status": "ALERT",
            "trigger": "possible_fall",
            "event_type": "possible_fall",
            "action": "Notify caregiver — check on the person now",
            "action_code": "NOTIFY_CAREGIVER",
            "reason": f"Possible fall detected ({detail})",
        }

    # 2) Prolonged inactivity
    if r.get("inactive_alert"):
        detail = r.get("inactive_reason") or "little motion while in view"
        if detail == "alert hold":
            detail = "inactivity hold"
        return {
            "status": "ATTENTION",
            "trigger": "inactive",
            "event_type": "inactive",
            "action": "Check in — person still for too long",
            "action_code": "CHECK_IN",
            "reason": f"Prolonged inactivity ({detail})",
        }

    # 3) Framing / zone warnings (not restricted-room entry)
    if r.get("zone_alert"):
        kind = r.get("zone_kind") or "partial_body"
        if kind == "out_of_view":
            return {
                "status": "ATTENTION",
                "trigger": "out_of_view",
                "event_type": "out_of_view",
                "action": "Locate person — they left the camera view",
                "action_code": "FIX_ZONE",
                "reason": "Person out of camera view",
            }
        return {
            "status": "ATTENTION",
            "trigger": "partial_body",
            "event_type": "partial_body",
            "action": "Fix framing — move camera back for full body (~2-5 m)",
            "action_code": "FIX_ZONE",
            "reason": "Partial body / incomplete pose (framing)",
        }

    activity = r.get("activity", "unknown")
    # Sustained framing issues are covered by zone_alert hold; avoid flicker on
    # brief unknown frames without elevating status every frame.
    return {
        "status": "SAFE",
        "trigger": "ok",
        "event_type": "ok",
        "action": f"Continue monitoring — activity: {activity}",
        "action_code": "CONTINUE",
        "reason": f"No active concern (activity: {activity})",
    }


def format_decision(d: dict, *, activity: str | None = None) -> str:
    """Markdown for Gradio status panel with color cue."""
    status_key = str(d.get("status") or "SAFE").upper()
    status = status_key.replace("_", " ").title()
    color = STATUS_COLORS.get(status_key, "#64748B")
    action = d.get("action", "Continue monitoring")
    reason = d.get("reason", d.get("trigger", ""))
    act = (activity or "").replace("_", " ").title()
    act_line = f"**Activity:** {act}  \n" if act else ""
    reason = str(reason or "").strip()
    action = str(action or "").strip()
    reason = reason[:1].upper() + reason[1:] if reason else "—"
    action = action[:1].upper() + action[1:] if action else "—"
    return (
        f"### <span style='color:{color}'>Status: {status}</span>\n"
        f"{act_line}"
        f"**Reason:** {reason}  \n"
        f"**Next Action:** {action}"
    )
