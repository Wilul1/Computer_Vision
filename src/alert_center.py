"""P4 Smart Alert Center — priority/category helpers, filtering, ownership.

Priority and category are derived at read time from event_type.
Caregiver response outcomes are stored on the event when acknowledging/resolving.
"""
from __future__ import annotations

import base64
import html as html_lib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .care_demo import get_patient, patients_for
from .event_store import ACTIVE_STATUSES, EventStore

# Rank: lower sorts first within priority band.
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
STATUS_RANK = {"OPEN": 0, "ACKNOWLEDGED": 1, "RESOLVED": 2}

# Case-management outcomes: (code, title, subtitle) — Care Response cards.
HANDLE_OUTCOMES: list[tuple[str, str, str]] = [
    ("checked_ok", "Patient is safe", "No injury found"),
    ("assisted", "Patient assisted", "Returned to safe position"),
    ("possible_injury", "Patient has a possible injury", "Requires medical attention"),
    ("escalated", "Emergency assistance requested", "Call for help immediately"),
    ("not_responding", "Patient not responding / not located", "No response"),
]
HANDLE_OUTCOME_LABELS = {k: title for k, title, _sub in HANDLE_OUTCOMES}
# Legacy codes still readable in History.
HANDLE_OUTCOME_LABELS.update({
    "false_alarm": "False alarm / no incident",
    "monitoring": "Continue monitoring",
    "checked_ok_legacy": "Checked patient — OK",
})


def outcome_label(code: str | None) -> str:
    if not code:
        return "—"
    return HANDLE_OUTCOME_LABELS.get(str(code).strip(), str(code))


def outcome_radio_choices() -> list[tuple[str, str]]:
    """Gradio Radio labels: title + subtitle (newline for card layout)."""
    return [(f"{title}\n{sub}", code) for code, title, sub in HANDLE_OUTCOMES]


def build_care_response_summary_html(
    event: dict | None,
    *,
    group: dict | None = None,
) -> str:
    """Header + incident card for the Care Response panel (no icons).

    Prefer a patient ``group`` (P4.1 reviewed bundle). Falls back to a single event.
    """
    if group:
        return _care_response_group_html(group)
    if not event:
        return """
        <div class="cv-cr">
          <div class="cv-cr-head">
            <div>
              <div class="cv-cr-title">Care Response</div>
              <div class="cv-cr-sub">Respond to the latest alert and record your findings.</div>
            </div>
          </div>
          <div class="cv-cr-empty">Select an alert above to review the incident.</div>
        </div>
        """
    eid = int(event.get("id") or 0)
    et = str(event.get("event_type") or "")
    meta = enrich_event(dict(event)) if "display_title" not in event else event
    title = html_lib.escape(
        str(meta.get("display_title") or meta.get("short_title") or event_display_title(et))
    )
    short = str(meta.get("short_title") or event_display_title(et, short=True))
    when = html_lib.escape(_fmt_care_when(str(event.get("created_at") or "")))
    status = str(event.get("event_status") or "OPEN").upper()
    is_fall = et == "possible_fall"
    badge = "Fall Alert" if is_fall else "Open Alert"
    wait = (
        "Awaiting Response" if status == "OPEN"
        else ("Acknowledged" if status == "ACKNOWLEDGED" else status.title())
    )
    activity = html_lib.escape("Sudden Fall" if is_fall else short)
    if is_fall:
        blurb_raw = (
            "Our system detected a sudden fall. "
            "Please check the patient as soon as possible."
        )
    else:
        blurb_raw = (
            str(event.get("reason_text") or event.get("action_text") or "").strip()
            or "Please check the patient as soon as possible."
        )
    blurb = html_lib.escape(blurb_raw)
    return f"""
    <div class="cv-cr">
      <div class="cv-cr-head">
        <div>
          <div class="cv-cr-title">Care Response</div>
          <div class="cv-cr-sub">Respond to the latest alert and record your findings.</div>
        </div>
        <span class="cv-cr-badge">{html_lib.escape(badge)}</span>
      </div>
      <div class="cv-cr-incident">
        <div class="cv-cr-incident-main">
          <div class="cv-cr-incident-id">Incident #{eid}</div>
          <div class="cv-cr-incident-title">{title}</div>
          <div class="cv-cr-incident-blurb">{blurb}</div>
        </div>
        <div class="cv-cr-incident-side">
          <div><span>Detected Activity</span><b>{activity}</b></div>
          <div><span>Time</span><b>{when}</b></div>
          <div class="cv-cr-wait">{html_lib.escape(wait)}</div>
        </div>
      </div>
    </div>
    """


def _care_response_group_html(group: dict[str, Any]) -> str:
    primary = group.get("primary_event") or {}
    et = str(primary.get("event_type") or "")
    is_fall = et == "possible_fall"
    n = int(group.get("finding_count") or len(group.get("children") or []) or 1)
    pname = html_lib.escape(str(group.get("patient_name") or "Patient"))
    title = html_lib.escape(str(group.get("display_title") or event_display_title(et)))
    when = html_lib.escape(_fmt_care_when(str(group.get("latest_created_at") or "")))
    status = str(group.get("overall_status") or "OPEN").upper()
    badge = "Fall Alert" if is_fall else "Patient Alert"
    wait = (
        "Awaiting Response" if status == "OPEN"
        else ("Acknowledged" if status == "ACKNOWLEDGED" else status.title())
    )
    why_lines = group.get("why_lines") or []
    blurb = html_lib.escape(
        " · ".join(why_lines[:3])
        if why_lines
        else (
            "Our system detected a sudden fall. Please check the patient as soon as possible."
            if is_fall
            else "Please check the patient as soon as possible."
        )
    )
    confirm = html_lib.escape(
        f"Resolving will close {n} reviewed finding{'s' if n != 1 else ''} for {group.get('patient_name') or 'this patient'}. "
        "New findings that arrive after this review stay open."
    )
    findings = "".join(
        f"<li>{html_lib.escape(line)}</li>" for line in why_lines
    )
    return f"""
    <div class="cv-cr">
      <div class="cv-cr-head">
        <div>
          <div class="cv-cr-title">Care Response</div>
          <div class="cv-cr-sub">Respond to the reviewed findings for this patient.</div>
        </div>
        <span class="cv-cr-badge">{html_lib.escape(badge)}</span>
      </div>
      <div class="cv-cr-incident">
        <div class="cv-cr-incident-main">
          <div class="cv-cr-incident-id">{pname} · {n} finding{'s' if n != 1 else ''}</div>
          <div class="cv-cr-incident-title">{title}</div>
          <div class="cv-cr-incident-blurb">{blurb}</div>
          {"<ul class='cv-cr-findings'>" + findings + "</ul>" if findings else ""}
        </div>
        <div class="cv-cr-incident-side">
          <div><span>Priority</span><b>{html_lib.escape(str(group.get('priority') or 'Unknown'))}</b></div>
          <div><span>Latest</span><b>{when}</b></div>
          <div class="cv-cr-wait">{html_lib.escape(wait)}</div>
        </div>
      </div>
      <div class="cv-cr-info">{confirm}</div>
    </div>
    """


def _fmt_care_when(iso: str | None) -> str:
    raw = (iso or "").strip()
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        # e.g. Nov 26, 2026 • 12:07 AM
        return dt.strftime("%b %d, %Y · %I:%M %p").replace(" 0", " ")
    except ValueError:
        return raw[:16] + " UTC" if len(raw) >= 16 else raw

_EVENT_META: dict[str, dict[str, str]] = {
    "possible_fall": {
        "priority": "High",
        "category": "Safety",
        "title": "Possible fall — urgent review",
        "short": "Possible fall detected",
    },
    "inactive": {
        "priority": "Medium",
        "category": "Activity",
        "title": "Inactivity — check patient",
        "short": "Prolonged inactivity",
    },
    "camera_unavailable": {
        "priority": "Medium",
        "category": "Monitoring",
        "title": "Camera unavailable — check monitoring",
        "short": "Camera unavailable",
    },
    "out_of_view": {
        "priority": "Low",
        "category": "Camera framing",
        "title": "Patient out of view",
        "short": "Out of view",
    },
    "partial_body": {
        "priority": "Low",
        "category": "Camera framing",
        "title": "Partial body detected",
        "short": "Partial body",
    },
}


def event_meta(event_type: str | None) -> dict[str, str]:
    """Safe mapping for known and unknown event types."""
    key = str(event_type or "").strip()
    if key in _EVENT_META:
        return dict(_EVENT_META[key])
    label = key.replace("_", " ").title() if key else "Unknown event"
    return {
        "priority": "Unknown",
        "category": "Other",
        "title": label,
        "short": label,
    }


def event_priority(event_type: str | None) -> str:
    return event_meta(event_type)["priority"]


def event_category(event_type: str | None) -> str:
    return event_meta(event_type)["category"]


def event_display_title(event_type: str | None, *, short: bool = False) -> str:
    meta = event_meta(event_type)
    return meta["short"] if short else meta["title"]


def enrich_event(event: dict[str, Any]) -> dict[str, Any]:
    """Attach derived fields + roster patient/room (no filesystem paths)."""
    out = dict(event)
    meta = event_meta(out.get("event_type"))
    out["priority"] = meta["priority"]
    out["category"] = meta["category"]
    out["display_title"] = meta["title"]
    out["short_title"] = meta["short"]
    patient = get_patient(str(out.get("patient_id") or ""))
    out["patient_name"] = patient["name"] if patient else str(out.get("patient_id") or "—")
    out["room"] = patient["room"] if patient else "—"
    out["caregiver_id"] = patient.get("caregiver_id") if patient else None
    return out


def sort_alerts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """High → Medium → Low; OPEN before ACKNOWLEDGED; newest id first."""

    def key(e: dict[str, Any]):
        pri = PRIORITY_RANK.get(str(e.get("priority") or "Unknown"), 99)
        st = STATUS_RANK.get(str(e.get("event_status") or ""), 99)
        eid = int(e.get("id") or 0)
        return (pri, st, -eid)

    return sorted(events, key=key)


def encode_alert_bundle(group: dict[str, Any]) -> str:
    """Stable reviewed-bundle token: ``primaryId|id1,id2,…`` (IDs frozen at present time)."""
    ids = [int(x) for x in (group.get("child_event_ids") or []) if x is not None]
    primary = int(group.get("primary_event_id") or (ids[0] if ids else 0))
    if primary and primary not in ids:
        ids = [primary, *ids]
    return f"{primary}|{','.join(str(i) for i in ids)}"


def parse_alert_bundle(value: Any) -> tuple[int | None, list[int]]:
    """Parse dropdown/token value into (primary_event_id, reviewed_child_ids)."""
    if value is None or value == "":
        return None, []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        eid = int(value)
        return eid, [eid]
    s = str(value).strip()
    if not s:
        return None, []
    if "|" in s:
        prim_s, ids_s = s.split("|", 1)
        try:
            primary = int(float(prim_s)) if prim_s.strip() else None
        except (TypeError, ValueError):
            primary = None
        ids: list[int] = []
        seen: set[int] = set()
        for part in ids_s.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                eid = int(float(part))
            except (TypeError, ValueError):
                continue
            if eid in seen:
                continue
            seen.add(eid)
            ids.append(eid)
        if primary is None and ids:
            primary = ids[0]
        elif primary is not None and primary not in seen:
            ids = [primary, *ids]
        return primary, ids
    try:
        eid = int(float(s))
    except (TypeError, ValueError):
        return None, []
    return eid, [eid]


def group_alerts_by_patient(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse enriched active events into one caregiver-facing alert per patient."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        pid = str(e.get("patient_id") or "").strip()
        if not pid:
            continue
        buckets.setdefault(pid, []).append(e)

    groups: list[dict[str, Any]] = []
    for pid, children in buckets.items():
        kids = sort_alerts([enrich_event(c) if "priority" not in c else dict(c) for c in children])
        if not kids:
            continue
        primary = kids[0]
        has_open = any(str(c.get("event_status")) == "OPEN" for c in kids)
        overall = "OPEN" if has_open else "ACKNOWLEDGED"
        latest = max(str(c.get("created_at") or "") for c in kids)
        latest_id = max(int(c.get("id") or 0) for c in kids)
        why_lines: list[str] = []
        for c in kids:
            short = str(c.get("short_title") or event_display_title(c.get("event_type"), short=True))
            when = _fmt_ts(c.get("created_at"))
            reason = str(c.get("reason_text") or c.get("action_text") or "").strip()
            line = f"{short} at {when}"
            if reason:
                line = f"{line} — {reason}"
            why_lines.append(line)
        other = [
            str(c.get("short_title") or event_display_title(c.get("event_type"), short=True))
            for c in kids[1:]
        ]
        snap = primary.get("snapshot_path")
        if not snap:
            for c in kids:
                if c.get("snapshot_path"):
                    snap = c.get("snapshot_path")
                    break
        groups.append({
            "patient_id": pid,
            "patient_name": primary.get("patient_name") or pid,
            "room": primary.get("room") or "—",
            "caregiver_id": primary.get("caregiver_id"),
            "children": kids,
            "child_event_ids": [int(c["id"]) for c in kids if c.get("id") is not None],
            "finding_count": len(kids),
            "primary_event": primary,
            "primary_event_id": int(primary.get("id") or 0),
            "priority": str(primary.get("priority") or "Unknown"),
            "category": str(primary.get("category") or "Other"),
            "display_title": str(primary.get("display_title") or event_display_title(primary.get("event_type"))),
            "short_title": str(primary.get("short_title") or event_display_title(primary.get("event_type"), short=True)),
            "overall_status": overall,
            "event_status": overall,
            "latest_created_at": latest,
            "created_at": latest,
            "latest_event_id": latest_id,
            "snapshot_path": snap,
            "why_lines": why_lines,
            "other_findings": other,
            "action_text": primary.get("action_text") or "",
            "reason_text": primary.get("reason_text") or "",
            "bundle_token": "",  # filled below
        })
        groups[-1]["bundle_token"] = encode_alert_bundle(groups[-1])

    def gkey(g: dict[str, Any]):
        pri = PRIORITY_RANK.get(str(g.get("priority") or "Unknown"), 99)
        # Groups with any OPEN finding before fully acknowledged.
        unacked = 0 if str(g.get("overall_status")) == "OPEN" else 1
        latest = str(g.get("latest_created_at") or "")
        return (pri, unacked, -_iso_sort_key(latest), -int(g.get("latest_event_id") or 0))

    return sorted(groups, key=gkey)


def _iso_sort_key(iso: str) -> float:
    raw = (iso or "").strip()
    if not raw:
        return 0.0
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return 0.0


def filter_patient_alert_groups(
    groups: list[dict[str, Any]],
    *,
    priority: str | None = None,
    patient_id: str | None = None,
    room: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Filter patient groups. Event-type/category/date match if any child matches."""
    pri = (priority or "").strip()
    if pri.lower() in ("", "all", "any"):
        pri = ""
    pid = (patient_id or "").strip()
    if pid.lower() in ("", "all", "any"):
        pid = ""
    room_f = (room or "").strip()
    if room_f.lower() in ("", "all", "any"):
        room_f = ""
    st = (status or "").strip()
    if st.lower() in ("", "all", "any"):
        st = ""
    et = (event_type or "").strip()
    if et.lower() in ("", "all", "any"):
        et = ""
    cat = (category or "").strip()
    if cat.lower() in ("", "all", "any"):
        cat = ""
    since = _parse_day_bound(date_from, end=False)
    until = _parse_day_bound(date_to, end=True)

    out: list[dict[str, Any]] = []
    for g in groups:
        if pri and str(g.get("priority")) != pri:
            continue
        if pid and str(g.get("patient_id")) != pid:
            continue
        if room_f and str(g.get("room")) != room_f:
            continue
        if st:
            if st.lower() == "active":
                if g.get("overall_status") not in ACTIVE_STATUSES:
                    continue
            elif str(g.get("overall_status")) != st:
                continue
        kids = list(g.get("children") or [])
        if et and not any(str(c.get("event_type")) == et for c in kids):
            continue
        if cat and not any(str(c.get("category")) == cat for c in kids):
            continue
        if since or until:
            if not any(
                (not since or str(c.get("created_at") or "") >= since)
                and (not until or str(c.get("created_at") or "") <= until)
                for c in kids
            ):
                continue
        out.append(g)
    return out


def list_caregiver_patient_alerts(
    store: EventStore,
    caregiver_id: str,
    *,
    limit: int = 80,
    priority: str | None = None,
    patient_id: str | None = None,
    room: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Active patient-level alert groups for assigned patients (P4.1)."""
    # Load all active findings first so each group still contains every child.
    active = list_caregiver_active_alerts(
        store, caregiver_id, limit=max(limit * 5, 120), status="Active",
    )
    groups = group_alerts_by_patient(active)
    filtered = filter_patient_alert_groups(
        groups,
        priority=priority,
        patient_id=patient_id,
        room=room,
        status=status or "Active",
        event_type=event_type,
        category=category,
        date_from=date_from,
        date_to=date_to,
    )
    return filtered[:limit]


def patient_alert_counts(groups: list[dict[str, Any]]) -> dict[str, int]:
    """Counts over patient-level alert groups."""
    counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
        "unacknowledged": 0,
        "acknowledged_active": 0,
        "active": 0,
        "findings": 0,
    }
    for g in groups:
        counts["active"] += 1
        counts["findings"] += int(g.get("finding_count") or 0)
        pri = str(g.get("priority") or "")
        if pri == "High":
            counts["high"] += 1
        elif pri == "Medium":
            counts["medium"] += 1
        elif pri == "Low":
            counts["low"] += 1
        if g.get("overall_status") == "OPEN":
            counts["unacknowledged"] += 1
        elif g.get("overall_status") == "ACKNOWLEDGED":
            counts["acknowledged_active"] += 1
    return counts


def find_patient_alert_group(
    store: EventStore,
    caregiver_id: str,
    *,
    primary_event_id: int | None = None,
    patient_id: str | None = None,
    reviewed_ids: list[int] | None = None,
) -> dict[str, Any] | None:
    """Locate a current patient group, or synthesize one from reviewed IDs."""
    groups = list_caregiver_patient_alerts(store, caregiver_id, limit=120)
    if primary_event_id is not None:
        for g in groups:
            if int(g.get("primary_event_id") or 0) == int(primary_event_id):
                return g
            if int(primary_event_id) in [int(x) for x in (g.get("child_event_ids") or [])]:
                return g
    pid = (patient_id or "").strip()
    if pid:
        for g in groups:
            if str(g.get("patient_id")) == pid:
                return g
    # Reviewed snapshot may be stale vs live groups — rebuild from captured IDs.
    if reviewed_ids:
        kids: list[dict[str, Any]] = []
        for eid in reviewed_ids:
            row = store.get(int(eid))
            if row is None:
                continue
            if row.get("event_status") not in ACTIVE_STATUSES:
                # Still show what was reviewed even if some resolved mid-flight.
                kids.append(enrich_event(row))
                continue
            kids.append(enrich_event(row))
        if kids:
            built = group_alerts_by_patient(kids)
            return built[0] if built else None
    return None


def _parse_day_bound(value: str | None, *, end: bool) -> str | None:
    """Normalize YYYY-MM-DD (or ISO prefix) to comparable UTC ISO bound."""
    raw = (value or "").strip()
    if not raw:
        return None
    day = raw[:10]
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return None
    if end:
        return f"{day}T23:59:59"
    return f"{day}T00:00:00"


def filter_alerts(
    events: list[dict[str, Any]],
    *,
    priority: str | None = None,
    patient_id: str | None = None,
    room: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Filter enriched events. Empty / 'All' values keep the list unchanged for that axis."""
    pri = (priority or "").strip()
    if pri.lower() in ("", "all", "any"):
        pri = ""
    pid = (patient_id or "").strip()
    if pid.lower() in ("", "all", "any"):
        pid = ""
    room_f = (room or "").strip()
    if room_f.lower() in ("", "all", "any"):
        room_f = ""
    st = (status or "").strip()
    if st.lower() in ("", "all", "any"):
        st = ""
    et = (event_type or "").strip()
    if et.lower() in ("", "all", "any"):
        et = ""
    cat = (category or "").strip()
    if cat.lower() in ("", "all", "any"):
        cat = ""
    since = _parse_day_bound(date_from, end=False)
    until = _parse_day_bound(date_to, end=True)

    out = []
    for e in events:
        if pri and str(e.get("priority")) != pri:
            continue
        if pid and str(e.get("patient_id")) != pid:
            continue
        if room_f and str(e.get("room")) != room_f:
            continue
        if st:
            if st.lower() == "active":
                if e.get("event_status") not in ACTIVE_STATUSES:
                    continue
            elif str(e.get("event_status")) != st:
                continue
        if et and str(e.get("event_type")) != et:
            continue
        if cat and str(e.get("category")) != cat:
            continue
        created = str(e.get("created_at") or "")
        if since and created < since:
            continue
        if until and created > until:
            continue
        out.append(e)
    return out


def list_caregiver_active_alerts(
    store: EventStore,
    caregiver_id: str,
    *,
    limit: int = 80,
    priority: str | None = None,
    patient_id: str | None = None,
    room: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Active (OPEN+ACK) alerts for assigned patients, enriched + prioritized."""
    ids = [p["id"] for p in patients_for(caregiver_id)]
    if not ids:
        return []
    # Fetch extra then filter status — list_for_patients is newest-first across patients.
    raw = store.list_for_patients(ids, limit=max(limit * 3, 60))
    active = [enrich_event(e) for e in raw if e.get("event_status") in ACTIVE_STATUSES]
    filtered = filter_alerts(
        active,
        priority=priority,
        patient_id=patient_id,
        room=room,
        status=status or "Active",
        event_type=event_type,
        category=category,
        date_from=date_from,
        date_to=date_to,
    )
    return sort_alerts(filtered)[:limit]


def alert_center_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    """Counts over an active (OPEN+ACK) enriched list (per finding)."""
    counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
        "unacknowledged": 0,
        "acknowledged_active": 0,
        "active": 0,
    }
    for e in events:
        counts["active"] += 1
        pri = str(e.get("priority") or "")
        if pri == "High":
            counts["high"] += 1
        elif pri == "Medium":
            counts["medium"] += 1
        elif pri == "Low":
            counts["low"] += 1
        if e.get("event_status") == "OPEN":
            counts["unacknowledged"] += 1
        elif e.get("event_status") == "ACKNOWLEDGED":
            counts["acknowledged_active"] += 1
    return counts


def caregiver_owns_patient(caregiver_id: str, patient_id: str) -> bool:
    """True when roster assigns ``patient_id`` to ``caregiver_id``."""
    patient = get_patient(str(patient_id or ""))
    return bool(patient and patient.get("caregiver_id") == caregiver_id)


def caregiver_owns_event(
    store: EventStore,
    event_id: int,
    caregiver_id: str,
) -> dict[str, Any] | None:
    """Return event row if it belongs to a patient assigned to caregiver; else None."""
    try:
        eid = int(event_id)
    except (TypeError, ValueError):
        return None
    row = store.get(eid)
    if row is None:
        return None
    patient = get_patient(str(row.get("patient_id") or ""))
    if not patient or patient.get("caregiver_id") != caregiver_id:
        return None
    return row


def snapshot_data_uri(rel_or_abs: str | None, *, max_bytes: int = 400_000) -> str | None:
    """Inline thumbnail for UI; returns None if missing/invalid. Never exposes path."""
    if not rel_or_abs:
        return None
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = config.ROOT / path
    try:
        if not path.is_file():
            return None
        raw = path.read_bytes()
        if not raw or len(raw) > max_bytes:
            return None
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except OSError:
        return None


def session_ready_for_event(store: EventStore, event: dict[str, Any]) -> dict[str, Any] | None:
    """If event.session_id has a READY playable recording for that patient, return brief info."""
    sid = event.get("session_id")
    if sid in (None, ""):
        return None
    try:
        sid_i = int(sid)
    except (TypeError, ValueError):
        return None
    row = store.get_session(sid_i)
    if not row:
        return None
    if str(row.get("patient_id") or "") != str(event.get("patient_id") or ""):
        return None
    from .patient_details import recording_availability

    avail = recording_availability(store, row)
    if not avail.get("playable"):
        return None
    return {
        "session_id": sid_i,
        "label": avail.get("label") or "READY",
    }


def _fmt_ts(iso: str | None) -> str:
    if not iso:
        return "—"
    return str(iso).replace("T", " ")[:19] + " UTC"


def _rel_age(iso: str | None) -> str:
    """Short age for collapsed cards / urgent banner."""
    if not iso:
        return ""
    try:
        raw = str(iso).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    except ValueError:
        return ""
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def build_urgent_watch_html(
    store: EventStore,
    caregiver_id: str,
    last_seen_id: int = 0,
) -> tuple[str, int]:
    """Fall-alert popup for OPEN possible_fall findings.

    Returns (html, newest_open_id). Client JS loops beep/vibrate until the
    caregiver presses **Silence alarm** (stored in the browser for that event id).

    `last_seen_id` is ignored for HTML content — the caregiver poller decides when
    to show vs seed quietly. Always builds a sticky popup when an OPEN possible_fall
    exists so Silence remains available.
    """
    try:
        prev = int(last_seen_id or 0)
    except (TypeError, ValueError):
        prev = 0
    # Interrupt only for possible-fall — inactivity / framing stay in Alert Center.
    rows = list_caregiver_active_alerts(
        store, caregiver_id, limit=20, status="OPEN", event_type="possible_fall",
    )
    if not rows:
        return "", abs(prev)

    focus = rows[0]
    eid = int(focus.get("id") or 0)
    newest = max(int(r.get("id") or 0) for r in rows)
    pid = html_lib.escape(str(focus.get("patient_id") or ""))
    pname = html_lib.escape(str(focus.get("patient_name") or "Patient"))
    room = html_lib.escape(str(focus.get("room") or "—"))
    title = html_lib.escape(
        str(focus.get("display_title") or focus.get("short_title") or "Possible fall")
    )
    reason = html_lib.escape(str(focus.get("reason_text") or focus.get("action_text") or ""))
    pri = html_lib.escape(str(focus.get("priority") or "High"))
    age = html_lib.escape(_rel_age(focus.get("created_at")))
    n_fall = len(rows)
    # data-alert-loop=1 → client repeats alarm until Silence (or event leaves OPEN).
    html = f"""
    <div class="cv-urgent-modal" data-event-id="{eid}" data-patient-id="{pid}"
         data-alert-loop="1" role="alertdialog" aria-modal="true"
         aria-label="Possible fall alert">
      <div class="cv-urgent-modal-backdrop"></div>
      <div class="cv-urgent-modal-card">
        <div class="cv-urgent-kicker">
          URGENT · possible fall
          {" · " + str(n_fall) + " open" if n_fall > 1 else ""}
        </div>
        <div class="cv-urgent-title">{title}</div>
        <div class="cv-urgent-meta"><b>{pname}</b> · {room} · {pri} · {age or "just now"}</div>
        <div class="cv-urgent-reason">{reason or "Check this patient now."}</div>
        <div class="cv-urgent-actions">
          <button type="button" class="cv-urgent-silence" data-event-id="{eid}">
            Silence alarm
          </button>
          <a href="#" role="button" class="cv-ac-open-patient cv-urgent-open"
             data-patient-id="{pid}" data-event-id="{eid}">Open patient</a>
        </div>
        <div class="cv-urgent-hint">
          Alarm keeps sounding until you press <b>Silence alarm</b> or <b>Open patient</b>.
        </div>
      </div>
    </div>
    """
    return html, newest


def poll_urgent_watch(
    store: EventStore,
    caregiver_id: str,
    last_seen_id: int | str = 0,
    *,
    interrupt_event_id: int = 0,
) -> tuple[str, str]:
    """Decide popup vs quiet seed for the caregiver alert timer.

    last_seen encoding:
      0   → cold page open
      +N  → quiet baseline (leftover OPEN falls up to N; no popup)
      -N  → sticky interrupt armed for event N
    """
    try:
        raw = int(float(last_seen_id or 0))
    except (TypeError, ValueError):
        raw = 0
    sticky = raw < 0
    prev = abs(raw)

    rows = list_caregiver_active_alerts(
        store, caregiver_id, limit=20, status="OPEN", event_type="possible_fall",
    )
    newest = max((int(r.get("id") or 0) for r in rows), default=0)

    def _arm(html: str, eid: int) -> tuple[str, str]:
        return html, str(-abs(int(eid)))

    interrupt_id = int(interrupt_event_id or 0)
    if interrupt_id > 0 and interrupt_id > prev:
        html, focus_newest = build_urgent_watch_html(
            store, caregiver_id, max(0, interrupt_id - 1),
        )
        if html:
            return _arm(html, max(focus_newest, interrupt_id))

    if newest <= 0:
        return "", "0"

    # Cold start: adopt leftover OPEN falls quietly — no beep on reopen.
    if prev == 0:
        return "", str(newest)

    # Sticky: keep showing while that fall (or a newer one) is still OPEN.
    if sticky and newest >= prev:
        html, _ = build_urgent_watch_html(store, caregiver_id, prev)
        if html:
            return _arm(html, newest)
        return "", str(prev)

    # Brand-new fall since quiet baseline.
    if newest > prev:
        html, _ = build_urgent_watch_html(store, caregiver_id, prev)
        if html:
            return _arm(html, newest)
        return "", str(prev)

    return "", str(prev)


def build_alert_center_html(
    store: EventStore,
    caregiver_id: str,
    *,
    priority: str | None = None,
    patient_id: str | None = None,
    room: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 40,
) -> str:
    """HTML body for the Alerts panel (counts + one card per patient)."""
    all_groups = list_caregiver_patient_alerts(store, caregiver_id, limit=120, status="Active")
    counts = patient_alert_counts(all_groups)
    rows = list_caregiver_patient_alerts(
        store,
        caregiver_id,
        limit=limit,
        priority=priority,
        patient_id=patient_id,
        room=room,
        status=status,
        event_type=event_type,
        category=category,
        date_from=date_from,
        date_to=date_to,
    )

    # Compact filter chips (not a 5-card empty grid) — list comes first for caregivers.
    chips = f"""
    <div class="cv-ac-chips" aria-label="Alert summary">
      <span class="cv-ac-chip cv-ac-chip-high"><b>{counts['high']}</b> High</span>
      <span class="cv-ac-chip cv-ac-chip-med"><b>{counts['medium']}</b> Med</span>
      <span class="cv-ac-chip"><b>{counts['low']}</b> Low</span>
      <span class="cv-ac-chip"><b>{counts['unacknowledged']}</b> Unacked</span>
      <span class="cv-ac-chip"><b>{counts['acknowledged_active']}</b> Acked</span>
    </div>
    """

    if not rows:
        list_html = '<div class="cv-alert-empty">No open alerts right now.</div>'
    else:
        bits = [_patient_alert_card_html(store, g) for g in rows]
        list_html = '<div class="cv-ac-list">' + "".join(bits) + "</div>"

    return f"""
    <div class="cv-section-head">
      <h2>Open cases</h2>
      <span class="cv-link">{counts['active']} patient{'s' if counts['active'] != 1 else ''} · {counts['findings']} finding{'s' if counts['findings'] != 1 else ''}</span>
    </div>
    {chips}
    {list_html}
    """


def _patient_alert_card_html(store: EventStore, g: dict[str, Any]) -> str:
    """One expandable card for all active findings on a patient."""
    eid = int(g.get("primary_event_id") or 0)
    pid = html_lib.escape(str(g.get("patient_id") or ""))
    pname = html_lib.escape(str(g.get("patient_name") or "—"))
    room = html_lib.escape(str(g.get("room") or "—"))
    title = html_lib.escape(str(g.get("display_title") or g.get("short_title") or "Alert"))
    cat = html_lib.escape(str(g.get("category") or "Other"))
    pri = html_lib.escape(str(g.get("priority") or "Unknown"))
    status = html_lib.escape(str(g.get("overall_status") or ""))
    when = html_lib.escape(_fmt_ts(g.get("latest_created_at") or g.get("created_at")))
    n = int(g.get("finding_count") or 1)
    action = html_lib.escape(str(g.get("action_text") or ""))
    pri_class = {
        "High": "cv-pri-high",
        "Medium": "cv-pri-med",
        "Low": "cv-pri-low",
    }.get(str(g.get("priority")), "cv-pri-unk")
    st_class = "cv-st-open" if g.get("overall_status") == "OPEN" else "cv-st-ack"

    thumb = snapshot_data_uri(g.get("snapshot_path"))
    if thumb:
        thumb_html = f'<img class="cv-ac-thumb" src="{thumb}" alt="Alert snapshot">'
    else:
        thumb_html = f'<div class="cv-ac-thumb cv-ac-thumb-empty">#{eid}</div>'

    why_lines = g.get("why_lines") or []
    brief = html_lib.escape(why_lines[0] if why_lines else (g.get("reason_text") or action or "Open for details"))
    other = g.get("other_findings") or []
    other_html = ""
    if other:
        other_html = (
            f'<div class="cv-ac-brief"><b>Also:</b> '
            + html_lib.escape(", ".join(other[:4]))
            + ("…" if len(other) > 4 else "")
            + "</div>"
        )

    child_bits: list[str] = []
    for c in g.get("children") or []:
        cid = int(c.get("id") or 0)
        ctitle = html_lib.escape(
            str(c.get("short_title") or event_display_title(c.get("event_type"), short=True))
        )
        cpri = html_lib.escape(str(c.get("priority") or "Unknown"))
        cst = html_lib.escape(str(c.get("event_status") or ""))
        cwhen = html_lib.escape(_fmt_ts(c.get("created_at")))
        creason = html_lib.escape(str(c.get("reason_text") or c.get("action_text") or ""))
        cthumb = snapshot_data_uri(c.get("snapshot_path"))
        cthumb_html = (
            f'<img class="cv-ac-thumb cv-ac-thumb-sm" src="{cthumb}" alt="Finding snapshot">'
            if cthumb
            else f'<div class="cv-ac-thumb cv-ac-thumb-empty cv-ac-thumb-sm">#{cid}</div>'
        )
        ready = session_ready_for_event(store, c)
        rec = ""
        if ready:
            rec = (
                f"<div><b>Recording:</b> Session #{ready['session_id']} ready — "
                "open patient → Watch recording.</div>"
            )
        child_bits.append(
            f"""
            <div class="cv-ac-finding" data-event-id="{cid}">
              {cthumb_html}
              <div>
                <div><b>{ctitle}</b> · {cpri} · {cst}</div>
                <div class="cv-muted">{cwhen}</div>
                <div>{creason}</div>
                {rec}
              </div>
            </div>
            """
        )

    age = html_lib.escape(_rel_age(g.get("latest_created_at") or g.get("created_at")))
    needs = ""
    if g.get("overall_status") == "OPEN" and g.get("priority") == "High":
        needs = '<span class="cv-needs-check">Needs check</span>'
    find_pill = f'<span class="cv-ac-count">{n} finding{"s" if n != 1 else ""}</span>'
    return f"""
    <details class="cv-ac-card" data-event-id="{eid}" data-patient-id="{pid}"
             data-bundle="{html_lib.escape(str(g.get('bundle_token') or ''))}">
      <summary class="cv-ac-summary">
        {thumb_html}
        <div class="cv-ac-main">
          <div class="cv-ac-top">
            <span class="cv-ac-name">{pname}</span>
            <span class="cv-ac-room">{room}</span>
            <span class="cv-pri {pri_class}">{pri}</span>
            <span class="cv-st {st_class}">{status}</span>
            {find_pill}
            {needs}
          </div>
          <div class="cv-ac-title">{title}</div>
          <div class="cv-ac-sub">{cat} · {when}{" · " + age if age else ""}</div>
          <div class="cv-ac-brief"><b>Why:</b> {brief}</div>
          {other_html}
          <div class="cv-ac-next"><b>Next:</b> {action or "Check the patient, then Acknowledge / Resolve reviewed findings"}</div>
        </div>
      </summary>
      <div class="cv-ac-detail">
        <div class="cv-ac-findings-head"><b>All findings for this patient</b></div>
        {''.join(child_bits)}
        <div class="cv-ac-actions">
          <a href="#" role="button" class="cv-ac-open-patient"
             data-patient-id="{pid}" data-event-id="{eid}">
            Open patient
          </a>
          <span class="cv-muted">
            Select this patient alert in Care Response. Resolve closes only these {n} reviewed finding{'s' if n != 1 else ''} —
            new alerts that arrive later stay open.
          </span>
        </div>
      </div>
    </details>
    """
