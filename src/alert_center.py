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


def build_care_response_summary_html(event: dict | None) -> str:
    """Header + incident card for the Care Response panel (no icons)."""
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
    """Counts over an active (OPEN+ACK) enriched list."""
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
    """Sticky interrupt banner for OPEN alerts (especially High).

    Returns (html, newest_open_id). Beeps only when a newer OPEN alert appears
    than last_seen_id — so a distracted caregiver still gets attention while
    watching live video.
    """
    try:
        prev = int(last_seen_id or 0)
    except (TypeError, ValueError):
        prev = 0
    # Interrupt only for possible-fall — inactivity / framing alerts stay in Alert Center.
    rows = list_caregiver_active_alerts(
        store, caregiver_id, limit=20, status="OPEN", event_type="possible_fall",
    )
    if not rows:
        return "", prev

    focus = rows[0]
    eid = int(focus.get("id") or 0)
    newest = max(int(r.get("id") or 0) for r in rows)
    is_new = newest > prev
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
    pulse = " cv-urgent-new" if is_new else ""
    # data-alert-new triggers the page-level beep/vibrate observer (Messenger strips <script>).
    html = f"""
    <div class="cv-urgent-banner{pulse}" data-event-id="{eid}" data-patient-id="{pid}"
         data-alert-new="{'1' if is_new else '0'}" role="alert">
      <div class="cv-urgent-kicker">
        URGENT · possible fall
        {" · " + str(n_fall) + " open" if n_fall > 1 else ""} · live
      </div>
      <div class="cv-urgent-title">{title}</div>
      <div class="cv-urgent-meta"><b>{pname}</b> · {room} · {pri} · {age or "just now"}</div>
      <div class="cv-urgent-reason">{reason or "Check this patient now."}</div>
      <div class="cv-urgent-actions">
        <a href="#" role="button" class="cv-ac-open-patient"
           data-patient-id="{pid}" data-event-id="{eid}">Open patient now</a>
        <span class="cv-muted">Fall alerts only — Acknowledge / Resolve after you check.</span>
      </div>
    </div>
    """
    return html, newest


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
    """HTML body for the Alerts panel (counts + prioritized expandable list)."""
    # Unfiltered active set for overview counts (preserve OPEN+ACK semantics).
    all_active = list_caregiver_active_alerts(store, caregiver_id, limit=120, status="Active")
    counts = alert_center_counts(all_active)
    rows = list_caregiver_active_alerts(
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

    stats = f"""
    <div class="cv-ac-stats">
      <div class="cv-ac-stat cv-ac-high"><b>{counts['high']}</b><span>High</span></div>
      <div class="cv-ac-stat cv-ac-med"><b>{counts['medium']}</b><span>Medium</span></div>
      <div class="cv-ac-stat cv-ac-low"><b>{counts['low']}</b><span>Low</span></div>
      <div class="cv-ac-stat"><b>{counts['unacknowledged']}</b><span>Unacked</span></div>
      <div class="cv-ac-stat"><b>{counts['acknowledged_active']}</b><span>Acked (active)</span></div>
    </div>
    <p class="cv-muted cv-ac-note">
      Active = OPEN + ACKNOWLEDGED. Acknowledge means seen — it does not force SAFE or close the alert.
      Possible fall means a heuristic signal, not a confirmed diagnosis.
    </p>
    """

    if not rows:
        list_html = '<div class="cv-alert-empty">No active alerts match these filters.</div>'
    else:
        bits = []
        for e in rows:
            bits.append(_alert_card_html(store, e))
        list_html = '<div class="cv-ac-list">' + "".join(bits) + "</div>"

    return f"""
    <div class="cv-section-head">
      <h2>Alert Center</h2>
      <span class="cv-link">{counts['active']} active</span>
    </div>
    {stats}
    <div class="cv-section-head" style="margin-top:14px">
      <h2>Prioritized alerts</h2>
      <span class="cv-link">{len(rows)} shown</span>
    </div>
    {list_html}
    """


def _alert_card_html(store: EventStore, e: dict[str, Any]) -> str:
    eid = int(e.get("id") or 0)
    pid = html_lib.escape(str(e.get("patient_id") or ""))
    pname = html_lib.escape(str(e.get("patient_name") or "—"))
    room = html_lib.escape(str(e.get("room") or "—"))
    title = html_lib.escape(str(e.get("display_title") or e.get("short_title") or "Alert"))
    cat = html_lib.escape(str(e.get("category") or "Other"))
    pri = html_lib.escape(str(e.get("priority") or "Unknown"))
    status = html_lib.escape(str(e.get("event_status") or ""))
    when = html_lib.escape(_fmt_ts(e.get("created_at")))
    action = html_lib.escape(str(e.get("action_text") or ""))
    reason = html_lib.escape(str(e.get("reason_text") or ""))
    pri_class = {
        "High": "cv-pri-high",
        "Medium": "cv-pri-med",
        "Low": "cv-pri-low",
    }.get(str(e.get("priority")), "cv-pri-unk")
    st_class = "cv-st-open" if e.get("event_status") == "OPEN" else "cv-st-ack"

    thumb = snapshot_data_uri(e.get("snapshot_path"))
    if thumb:
        thumb_html = f'<img class="cv-ac-thumb" src="{thumb}" alt="Alert snapshot">'
    else:
        thumb_html = f'<div class="cv-ac-thumb cv-ac-thumb-empty">#{eid}</div>'

    ctx_bits = []
    if action:
        ctx_bits.append(f"<div><b>Action:</b> {action}</div>")
    if reason:
        ctx_bits.append(f"<div><b>Reason:</b> {reason}</div>")
    ctx_bits.append(f"<div><b>Created:</b> {when}</div>")
    ctx_bits.append(
        f"<div><b>Acknowledged:</b> {html_lib.escape(_fmt_ts(e.get('acknowledged_at')))}</div>"
    )
    ctx_bits.append(
        f"<div><b>Resolved:</b> {html_lib.escape(_fmt_ts(e.get('resolved_at')))}</div>"
    )
    if e.get("handled_by"):
        ctx_bits.append(
            f"<div><b>Handled by:</b> {html_lib.escape(str(e.get('handled_by')))}</div>"
        )
    if e.get("handle_outcome"):
        ctx_bits.append(
            f"<div><b>Outcome:</b> {html_lib.escape(outcome_label(e.get('handle_outcome')))}</div>"
        )
    if e.get("handle_note"):
        ctx_bits.append(
            f"<div><b>Note:</b> {html_lib.escape(str(e.get('handle_note')))}</div>"
        )
    sid = e.get("session_id")
    if sid not in (None, ""):
        ctx_bits.append(f"<div><b>Session:</b> #{html_lib.escape(str(sid))}</div>")
    ready = session_ready_for_event(store, e)
    if ready:
        ctx_bits.append(
            f"<div><b>Recording:</b> Session #{ready['session_id']} has a ready recording — "
            "open the patient page and choose it under Session recording.</div>"
        )
    elif sid not in (None, ""):
        ctx_bits.append(
            "<div><b>Recording:</b> No playable READY recording for this session yet.</div>"
        )

    age = html_lib.escape(_rel_age(e.get("created_at")))
    needs = ""
    if e.get("event_status") == "OPEN" and e.get("priority") == "High":
        needs = '<span class="cv-needs-check">Needs check</span>'
    detail_hint = reason or action or "Open for full context + Open patient"
    return f"""
    <details class="cv-ac-card" data-event-id="{eid}" data-patient-id="{pid}">
      <summary class="cv-ac-summary">
        {thumb_html}
        <div class="cv-ac-main">
          <div class="cv-ac-top">
            <span class="cv-ac-name">{pname}</span>
            <span class="cv-ac-room">{room}</span>
            <span class="cv-pri {pri_class}">{pri}</span>
            <span class="cv-st {st_class}">{status}</span>
            {needs}
          </div>
          <div class="cv-ac-title">{title}</div>
          <div class="cv-ac-sub">{cat} · {when}{" · " + age if age else ""}</div>
          <div class="cv-ac-brief"><b>Why:</b> {detail_hint}</div>
          <div class="cv-ac-next"><b>Next:</b> {action or "Check the patient, then Acknowledge / Resolve"}</div>
        </div>
      </summary>
      <div class="cv-ac-detail">
        {''.join(ctx_bits)}
        <div class="cv-ac-actions">
          <a href="#" role="button" class="cv-ac-open-patient"
             data-patient-id="{pid}" data-event-id="{eid}">
            Open patient
          </a>
          <span class="cv-muted">Event #{eid} · use Care response below after selecting this event</span>
        </div>
      </div>
    </details>
    """
