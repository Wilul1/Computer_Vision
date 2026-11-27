"""Patient details page for CareView (filtered to one patient)."""
from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .care_demo import get_patient, resolve_video_path
from .care_home import (
    CV,
    _event_title,
    _rel_time,
    effective_status,
    reconcile_patient_status,
)
from .event_store import EventStore
from .playback import PlaybackError, event_snapshot_url, session_meta_payload

_ACTIVITIES = ("sitting", "standing", "walking", "running", "lying", "unknown")

EMPTY_PLAYER_HTML = """
<div class="pd-player-panel" data-patient="" data-session="">
  <div class="pd-empty">No issue recording to play yet.
  Open an alert to load the session video linked to that finding.</div>
</div>
"""

EMPTY_PLAYER_INFO_HTML = """
<div class="pd-player-info pd-muted">The recording for this alert appears here when available.
Past sessions are not listed.</div>
"""


def _assert_access(user, patient_id: str):
    import gradio as gr

    if not user:
        raise gr.Error("Sign in first.")
    patient = get_patient(patient_id)
    if not patient or patient.get("caregiver_id") != user["id"]:
        raise gr.Error("Patient not found or not assigned to you.")
    return patient


def _fmt_dur(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m}m {rem}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _fmt_clock(seconds: float) -> str:
    """Compact mm:ss / h:mm:ss for incident labels."""
    s = max(0, int(round(float(seconds or 0))))
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m}:{rem:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{rem:02d}"


def _parse_iso(iso: str | None) -> datetime | None:
    raw = (iso or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def incident_offset_s(
    event: dict[str, Any] | None,
    session: dict[str, Any] | None,
) -> float | None:
    """Seconds from recording start to alert time, clamped to session duration.

    Uses ``recording_started_at`` then ``started_at``. Returns None when timing
    cannot be computed.
    """
    if not event or not session:
        return None
    created = _parse_iso(str(event.get("created_at") or ""))
    started = _parse_iso(
        str(session.get("recording_started_at") or session.get("started_at") or "")
    )
    if created is None or started is None:
        return None
    offset = (created - started).total_seconds()
    if offset < 0:
        offset = 0.0
    duration = float(
        session.get("recording_duration_s")
        or session.get("duration_s")
        or 0.0
    )
    if duration > 0:
        offset = min(offset, duration)
    return float(offset)


def incident_marker_payload(
    event: dict[str, Any] | None,
    session: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """UI payload for classification chip + yellow timeline tick."""
    if not event:
        return None
    et = str(event.get("event_type") or "")
    label = _event_title(et)
    is_fall = et == "possible_fall"
    offset = incident_offset_s(event, session)
    duration = float(
        (session or {}).get("recording_duration_s")
        or (session or {}).get("duration_s")
        or 0.0
    )
    pct = None
    jump_s = None
    if offset is not None:
        jump_s = max(0.0, float(offset) - 5.0)
        if duration > 0:
            pct = min(100.0, max(0.0, 100.0 * float(offset) / duration))
    try:
        eid = int(event.get("id") or 0) or None
    except (TypeError, ValueError):
        eid = None
    return {
        "event_id": eid,
        "label": label,
        "is_fall": is_fall,
        "offset_s": offset,
        "jump_s": jump_s,
        "pct": pct,
        "duration_s": duration if duration > 0 else None,
    }


def build_incident_timeline_html(marker: dict[str, Any] | None) -> str:
    """Custom timeline strip (YouTube-ad-style tick) under the native player."""
    if not marker:
        return ""
    label = html_lib.escape(str(marker.get("label") or "Alert"))
    is_fall = bool(marker.get("is_fall"))
    chip_cls = "cv-incident-chip cv-incident-fall" if is_fall else "cv-incident-chip"
    jump_label = "Jump to fall" if is_fall else "Jump to incident"
    offset = marker.get("offset_s")
    if offset is None:
        return f"""
<div class="cv-incident-bar cv-incident-unavailable">
  <div class="{chip_cls}">{label} · Incident time unavailable</div>
</div>
"""
    clock = html_lib.escape(_fmt_clock(float(offset)))
    jump_s = float(marker.get("jump_s") if marker.get("jump_s") is not None else offset)
    pct = marker.get("pct")
    duration = marker.get("duration_s")
    pct_attr = f'{float(pct):.2f}' if isinstance(pct, (int, float)) else ""
    dur_attr = f'{float(duration):.3f}' if isinstance(duration, (int, float)) else ""
    left_style = (
        f'style="left:{float(pct):.2f}%"'
        if isinstance(pct, (int, float))
        else ""
    )
    mark_cls = "cv-incident-mark cv-incident-mark-fall" if is_fall else "cv-incident-mark"
    return f"""
<div class="cv-incident-bar" data-offset="{float(offset):.3f}"
     data-jump="{jump_s:.3f}" data-duration="{dur_attr}" data-pct="{pct_attr}">
  <div class="cv-incident-row">
    <div class="{chip_cls}">{label} · {clock} into recording</div>
    <button type="button" class="cv-incident-jump">{jump_label}</button>
  </div>
  <div class="cv-incident-track" role="img"
       aria-label="{label} at {clock}">
    <button type="button" class="{mark_cls}" {left_style}
            aria-label="{jump_label} at {clock}"></button>
  </div>
</div>
"""


def _today_prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _activity_seconds_today(store: EventStore, patient_id: str) -> dict[str, float]:
    totals = {a: 0.0 for a in _ACTIVITIES}
    # Prefer durable activity_segments (+ legacy session JSON fallback in store).
    for k, v in store.activity_seconds_today(patient_id, _today_prefix()).items():
        key = str(k).lower()
        if key not in totals:
            totals[key] = 0.0
        totals[key] += float(v or 0)
    return totals


def _live_snap(patient_id: str) -> dict | None:
    try:
        from .live_hub import get_shared_hub

        hub = get_shared_hub()
        if hub is None or hub.patient_id != patient_id:
            return None
        return hub.snapshot()
    except Exception:
        return None


def _snapshot_abs(rel: str | None) -> Path | None:
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = config.ROOT / path
    return path if path.exists() else None


def _fmt_session_when(iso: str | None) -> str:
    raw = (iso or "").strip()
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return raw


def recording_availability(
    store: EventStore,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Label + playable flag for a session row (scoped to that row's patient_id)."""
    status_raw = session.get("recording_status")
    status = str(status_raw or "NONE").upper().strip() or "NONE"
    sid = int(session.get("id") or 0)
    pid = str(session.get("patient_id") or "")

    if status_raw in (None, "") or status == "NONE":
        return {
            "label": "No recording",
            "status": "NONE",
            "playable": False,
            "video_url": None,
        }
    if status == "FAILED":
        return {
            "label": "Recording failed",
            "status": status,
            "playable": False,
            "video_url": None,
        }
    if status == "INCOMPLETE":
        return {
            "label": "Recording incomplete",
            "status": status,
            "playable": False,
            "video_url": None,
        }
    if status in ("RECORDING", "FINALIZING"):
        return {
            "label": "Recording in progress",
            "status": status,
            "playable": False,
            "video_url": None,
        }
    if status == "READY" and sid and pid:
        try:
            meta = session_meta_payload(store, sid, patient_id=pid)
            if (
                meta.get("playback_available")
                and meta.get("video_url")
                and int(meta.get("session_id") or 0) == sid
                and str(meta.get("patient_id") or "") == pid
            ):
                return {
                    "label": "Recording available",
                    "status": status,
                    "playable": True,
                    "video_url": meta["video_url"],
                    "meta": meta,
                }
        except PlaybackError:
            pass
        return {
            "label": "Recording unavailable",
            "status": status,
            "playable": False,
            "video_url": None,
        }
    return {
        "label": "Recording unavailable",
        "status": status,
        "playable": False,
        "video_url": None,
    }


def session_playback_choices(
    store: EventStore,
    patient_id: str,
    *,
    limit: int = 12,
    playable_only: bool = True,
) -> list[tuple[str, int]]:
    """Dropdown choices for one patient.

    Default playable_only=True so only verified READY recordings can be selected.
    History rows still list unavailable sessions with explicit labels.
    """
    choices: list[tuple[str, int]] = []
    for s in store.list_sessions(patient_id, limit=limit):
        if str(s.get("patient_id") or "") != patient_id:
            continue
        avail = recording_availability(store, s)
        if playable_only and not avail["playable"]:
            continue
        sid = int(s["id"])
        when = _fmt_session_when(s.get("started_at"))
        dur = _fmt_dur(float(s.get("recording_duration_s") or s.get("duration_s") or 0))
        mark = "Playable" if avail["playable"] else avail["label"]
        label = f"#{sid} · {when} · {dur} · {mark}"
        choices.append((label, sid))
    return choices


def empty_playback_panel() -> tuple[Any, str, str, None]:
    """Cleared player (used on patient switch / back). Session picker stays hidden."""
    import gradio as gr

    return (
        gr.update(choices=[], value=None, visible=False),
        EMPTY_PLAYER_HTML,
        EMPTY_PLAYER_INFO_HTML,
        None,
    )


def playback_panel_for_patient(
    store: EventStore,
    user,
    patient_id: str,
    *,
    focus_event_id: int | None = None,
) -> tuple[Any, str, str, int | None]:
    """Watch recording for one alert only — no past-session picker.

    When ``focus_event_id`` is set, auto-loads that event's linked session video
    (if READY). Otherwise the player stays empty.
    """
    import gradio as gr

    _assert_access(user, patient_id)
    hidden_dd = gr.update(choices=[], value=None, visible=False)
    if focus_event_id in (None, ""):
        return hidden_dd, EMPTY_PLAYER_HTML, EMPTY_PLAYER_INFO_HTML, None
    try:
        eid = int(focus_event_id)
    except (TypeError, ValueError):
        return (
            hidden_dd,
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">Invalid alert — cannot load a recording.</div>',
            None,
        )

    event = store.get(eid)
    if event is None or str(event.get("patient_id") or "") != patient_id:
        return (
            hidden_dd,
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">That alert is not available for this patient.</div>',
            None,
        )

    sid_raw = event.get("session_id")
    if sid_raw in (None, ""):
        return (
            hidden_dd,
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">This alert has no linked session recording yet.</div>',
            None,
        )
    try:
        sid = int(sid_raw)
    except (TypeError, ValueError):
        return (
            hidden_dd,
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">This alert has no linked session recording yet.</div>',
            None,
        )

    player, info, loaded = select_session_playback(
        user, patient_id, sid, store, event=event,
    )
    if loaded is None:
        return hidden_dd, player, info, None
    label = f"Issue recording · session #{sid}"
    return (
        gr.update(choices=[(label, sid)], value=sid, visible=False),
        player,
        info,
        sid,
    )


def build_playback_player_html(
    *,
    patient_id: str,
    session_id: int,
    video_url: str,
    incident: dict[str, Any] | None = None,
) -> str:
    """Trusted P2 video_url only — never invent paths."""
    safe_url = html_lib.escape(video_url, quote=True)
    safe_pid = html_lib.escape(patient_id, quote=True)
    timeline = build_incident_timeline_html(incident)
    return f"""
<div class="pd-player-panel" data-patient="{safe_pid}" data-session="{int(session_id)}">
  <video class="pd-player" controls playsinline preload="metadata"
    src="{safe_url}"></video>
  {timeline}
  <div class="pd-muted" style="margin-top:8px">
    Uses the browser player (Range streaming). Older recordings may not play on every phone.
  </div>
</div>
"""


def build_playback_info_html(
    store: EventStore,
    user,
    patient_id: str,
    session_id: int,
    meta: dict[str, Any],
    *,
    incident: dict[str, Any] | None = None,
) -> str:
    patient = _assert_access(user, patient_id)
    row = store.get_session(int(session_id)) or {}
    if str(row.get("patient_id") or "") != patient_id:
        return '<div class="pd-empty">Session is not available for this patient.</div>'
    name = html_lib.escape(str(patient.get("name") or meta.get("patient_name") or ""))
    room = html_lib.escape(
        str(meta.get("room_id") or row.get("room_id") or patient.get("room") or "—")
    )
    when = html_lib.escape(_fmt_session_when(meta.get("started_at") or row.get("started_at")))
    rec_dur = _fmt_dur(float(meta.get("recording_duration_s") or row.get("recording_duration_s") or 0))
    sess_st = html_lib.escape(str(meta.get("session_status") or row.get("session_status") or "—"))
    final_act = (row.get("final_activity") or "—")
    if isinstance(final_act, str):
        final_act = final_act.replace("_", " ").title()
    final_act = html_lib.escape(str(final_act))
    rec_st = html_lib.escape(str(meta.get("recording_status") or row.get("recording_status") or "—"))
    incident_row = ""
    if incident:
        lab = html_lib.escape(str(incident.get("label") or "Alert"))
        off = incident.get("offset_s")
        when_inc = (
            html_lib.escape(_fmt_clock(float(off)))
            if off is not None
            else "unavailable"
        )
        incident_row = (
            f"<div><b>Incident</b><br>{lab} · {when_inc}</div>"
        )
    return f"""
<div class="pd-player-info">
  <div class="pd-meta" style="margin-top:0">
    <div><b>Patient</b><br>{name}</div>
    <div><b>Room</b><br>{room}</div>
    <div><b>Session time</b><br>{when}</div>
    <div><b>Recording duration</b><br>{rec_dur}</div>
    <div><b>Session status</b><br>{sess_st}</div>
    <div><b>Final activity</b><br>{final_act}</div>
    <div><b>Recording status</b><br>{rec_st}</div>
    {incident_row}
  </div>
</div>
"""


def select_session_playback(
    user,
    patient_id: str,
    session_id,
    store: EventStore,
    *,
    event: dict[str, Any] | None = None,
) -> tuple[str, str, int | None]:
    """Revalidate P2 meta and build player HTML. Rejects cross-patient / stale ids."""
    pid = (patient_id or "").strip()
    if not pid:
        return (
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">Open a patient before selecting a recording.</div>',
            None,
        )
    _assert_access(user, pid)
    if session_id in (None, ""):
        return EMPTY_PLAYER_HTML, EMPTY_PLAYER_INFO_HTML, None
    try:
        sid = int(session_id)
    except (TypeError, ValueError):
        return (
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">Invalid session selection.</div>',
            None,
        )

    row = store.get_session(sid)
    if row is None or str(row.get("patient_id") or "") != pid:
        # Wrong patient or unknown — never keep a previous player's src.
        return (
            EMPTY_PLAYER_HTML,
            '<div class="pd-empty">That session is not available for this patient.</div>',
            None,
        )

    try:
        meta = session_meta_payload(store, sid, patient_id=pid)
    except PlaybackError as exc:
        return (
            EMPTY_PLAYER_HTML,
            f'<div class="pd-empty">{html_lib.escape(exc.message)}</div>',
            None,
        )

    if (
        not meta.get("playback_available")
        or not meta.get("video_url")
        or int(meta.get("session_id") or 0) != sid
        or str(meta.get("patient_id") or "") != pid
    ):
        avail = recording_availability(store, row)
        return (
            EMPTY_PLAYER_HTML,
            f'<div class="pd-empty">{html_lib.escape(avail["label"])}. '
            "This session cannot be played.</div>",
            None,
        )

    incident = incident_marker_payload(event, row) if event else None
    player = build_playback_player_html(
        patient_id=pid,
        session_id=sid,
        video_url=str(meta["video_url"]),
        incident=incident,
    )
    info = build_playback_info_html(
        store, user, pid, sid, meta, incident=incident,
    )
    return player, info, sid


def build_patient_details_html(
    store: EventStore,
    user,
    patient_id: str,
    *,
    focus_event_id: int | None = None,
    alert_context: bool = False,
) -> str:
    """HTML for one patient's details page (server-filtered).

    ``alert_context`` (from alert banner / review) shows Current alerts + snapshots.
    Plain patient open stays lean: identity + monitoring info only (live cam is separate).
    """
    patient = _assert_access(user, patient_id)
    reconcile_patient_status(store, patient_id)
    status, st, active = effective_status(store, patient_id)
    show_alerts = bool(alert_context or focus_event_id is not None)
    activity = (st.get("activity") or "—").replace("_", " ").title()
    reason = st.get("reason") or "No recent monitoring reason recorded."
    updated = _rel_time(st.get("updated_at"))
    source_type = patient.get("source_type") or "unknown"
    live = _live_snap(patient_id) if source_type == "live" else None
    active_sess = store.get_active_session(patient_id) if source_type == "live" else None
    if source_type == "live":
        source_label = "LIVE CAMERA"
        source_note = patient.get("note") or (
            f"Laptop webcam ({patient.get('name') or 'Patient'} · {patient.get('room') or '—'})"
        )
        if live and live.get("connected"):
            cam_status = "CONNECTED · live frames updating"
        elif active_sess is not None:
            cam_status = "Session ACTIVE but camera feed not connected"
        else:
            cam_status = "Not monitoring — open /camera-station on the laptop"
    else:
        source_label = "PRERECORDED VIDEO"
        vpath = resolve_video_path(patient)
        source_note = patient.get("note") or (str(vpath) if vpath else "Video missing")
        cam_status = "Video file available" if vpath else "Video file missing"

    pill_class = {
        "SAFE": "pd-pill pd-pill-safe",
        "ATTENTION": "pd-pill pd-pill-attention",
        "ALERT": "pd-pill pd-pill-alert",
    }.get(status, "pd-pill")
    pill = {
        "SAFE": ("#DCFCE7", CV["safe"]),
        "ATTENTION": ("#FEF9C3", "#A16207"),
        "ALERT": ("#FEE2E2", CV["alert"]),
    }.get(status, ("#F1F5F9", CV["muted"]))

    sessions = store.list_sessions(patient_id, limit=12)
    latest = sessions[0] if sessions else None
    events = store.list_recent(patient_id, limit=30)
    counts = store.count_events_by_status(patient_id)
    today_secs = _activity_seconds_today(store, patient_id)
    monitored_today = sum(today_secs.values())

    # Activity bars
    max_sec = max(today_secs.values()) if today_secs else 0.0
    activity_cards = []
    for name in ("sitting", "standing", "walking", "running", "lying", "unknown"):
        sec = today_secs.get(name, 0.0)
        pct = (100.0 * sec / max_sec) if max_sec > 0 else 0.0
        activity_cards.append(
            f"""
            <div class="pd-act">
              <div class="pd-act-top"><span>{name.title()}</span><b>{_fmt_dur(sec)}</b></div>
              <div class="pd-act-bar"><i style="width:{pct:.1f}%"></i></div>
            </div>
            """
        )
    if monitored_today <= 0:
        if source_type == "video":
            activity_body = (
                '<div class="pd-empty"><b>Video is linked but not analyzed yet.</b><br/>'
                "Tap <b>Run room check</b> below to generate the session summary, "
                "alerts (if any), and a playable recording.</div>"
            )
        else:
            activity_body = (
                '<div class="pd-empty">No monitored activity recorded for this patient today. '
                "Run a room check on this page to create a session.</div>"
            )
    else:
        activity_body = (
            f'<div class="pd-muted">Total monitored today (from saved sessions): '
            f"<b>{_fmt_dur(monitored_today)}</b></div>"
            f'<div class="pd-acts">{"".join(activity_cards)}</div>'
        )

    # Current issues only — resolved events + snapshots move to History.
    current_events = [e for e in events if e.get("event_status") != "RESOLVED"]
    if not current_events:
        event_rows = (
            '<div class="pd-empty">No open alerts for this patient. '
            "Resolved alerts and their snapshots are under <b>History</b>.</div>"
        )
    else:
        bits = []
        for e in current_events:
            eid = e.get("id")
            focus = " pd-focus" if focus_event_id and int(eid) == int(focus_event_id) else ""
            title = _event_title(e.get("event_type"))
            snap = _snapshot_abs(e.get("snapshot_path"))
            snap_bit = " · snapshot on file" if snap else " · no snapshot file"
            bits.append(
                f"""
                <div class="pd-row{focus}" data-event-id="{eid}">
                  <div>
                    <div class="pd-row-title">#{eid} · {title}</div>
                    <div class="pd-row-sub">
                      {_rel_time(e.get('created_at'))} · {e.get('event_status')}
                      {snap_bit}
                      <br>{html_lib.escape(e.get('reason_text') or e.get('action_text') or '')}
                    </div>
                  </div>
                  <div class="pd-chip">{e.get('event_status')}</div>
                </div>
                """
            )
        event_rows = (
            '<div class="pd-muted" style="margin-bottom:8px">'
            "Open / acknowledged only — resolve moves the case to <b>History</b>."
            "</div>"
            + "".join(bits)
        )

    # Snapshots for current (non-resolved) alerts only — served via API (no data-URL tabs).
    snaps = []
    for e in current_events:
        path = _snapshot_abs(e.get("snapshot_path"))
        if not path:
            continue
        eid = int(e.get("id") or 0)
        if not eid:
            continue
        snap_src = html_lib.escape(event_snapshot_url(eid, patient_id), quote=True)
        snaps.append(
            f"""
            <div class="pd-snap">
              <button type="button" class="cv-snap-zoom" aria-label="Enlarge snapshot">
                <img class="pd-snap-img" src="{snap_src}" alt="event {eid} snapshot"/>
              </button>
              <div class="pd-snap-meta">
                <b>{_event_title(e.get('event_type'))}</b><br>
                {_rel_time(e.get('created_at'))} · event #{eid}<br>
                <span class="pd-muted">Tap image to enlarge</span>
              </div>
            </div>
            """
        )
        if len(snaps) >= 8:
            break
    snap_body = (
        f'<div class="pd-snaps">{"".join(snaps)}</div>'
        if snaps
        else (
            '<div class="pd-empty">No open-alert snapshots. '
            "Resolved snapshots are kept under <b>History</b>.</div>"
        )
    )

    # Monitoring results
    most = "—"
    if monitored_today > 0:
        most = max(today_secs.items(), key=lambda kv: kv[1])[0].title()
    results = f"""
      <div class="pd-grid3">
        <div class="pd-stat"><div class="pd-stat-l">Monitored today</div><div class="pd-stat-n">{_fmt_dur(monitored_today)}</div></div>
        <div class="pd-stat"><div class="pd-stat-l">Open / Ack</div><div class="pd-stat-n">{counts['OPEN'] + counts['ACKNOWLEDGED']}</div></div>
        <div class="pd-stat"><div class="pd-stat-l">Resolved</div><div class="pd-stat-n">{counts['RESOLVED']}</div></div>
      </div>
      <div class="pd-muted" style="margin-top:10px">
        Most observed activity today: <b>{most}</b> · Saved sessions: <b>{len(sessions)}</b><br>
        These are monitoring estimates from room checks — not clinical assessments.
      </div>
    """

    if source_type == "live" and (active_sess is not None or (live and live.get("session_id"))):
        sid = (live or {}).get("session_id") or (active_sess or {}).get("id")
        elapsed = float((live or {}).get("elapsed_s") or (active_sess or {}).get("duration_s") or 0)
        conf = (live or {}).get("confidence")
        conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
        act = ((live or {}).get("activity") or st.get("activity") or "—").replace("_", " ").title()
        conn = "CONNECTED" if (live or {}).get("connected") else "WAITING FOR CAMERA"
        latest_block = (
            f"<b>ACTIVE</b> live session #{sid} · {conn}<br>"
            f"Elapsed {_fmt_dur(elapsed)} · Activity {act} (conf {conf_s})<br>"
            "Live video appears in the preview below while the laptop Camera Station is streaming. "
            "This page never opens the webcam."
        )
    elif latest:
        sess_st = latest.get("session_status") or "COMPLETED"
        latest_block = (
            f"Latest session #{latest.get('id')} · <b>{sess_st}</b> · "
            f"{_fmt_dur(float(latest.get('duration_s') or 0))} · "
            f"{latest.get('final_status') or '—'} · {latest.get('final_activity') or '—'}"
        )
        if sess_st == "INTERRUPTED":
            latest_block += "<br>Session ended unexpectedly — durations reflect recorded segments only."
    else:
        latest_block = (
            "No saved session yet — start Live Camera on the laptop (Ana) "
            "or Run room check for prerecorded rooms."
        )

    if show_alerts:
        alert_block = f"""
      <div class="pd-sec">
        <h2>Current alerts</h2>
        {event_rows}
      </div>

      <div class="pd-sec">
        <h2>Current alert snapshots</h2>
        {snap_body}
      </div>
        """
    else:
        alert_block = """
      <div class="pd-sec">
        <div class="pd-muted">
          Alert details, snapshots, Care Response, and Watch recording appear when you open
          this patient from an <b>alert banner</b> or the Alert Center.
        </div>
      </div>
        """

    return f"""
    <div id="careview-root" class="pd-page">
      <style>
        /* Colors also reinforced in ui.py APP_CSS (#careview-shell) — Gradio .prose * overrides plain rules. */
        .pd-page{{padding:18px 22px 10px;background:#F1F5F9!important;color:#0F172A!important;font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif}}
        .pd-back{{display:inline-block;margin-bottom:12px;color:#2563EB!important;font-weight:650;font-size:13px}}
        .pd-head{{background:#FFFFFF!important;border:1px solid #E2E8F0!important;border-radius:18px;padding:18px;margin-bottom:14px}}
        .pd-name{{font-size:26px;font-weight:750;letter-spacing:-.4px;color:#0F172A!important}}
        .pd-sub{{margin-top:4px;color:#64748B!important;font-size:13px}}
        .pd-pill{{display:inline-block;margin-top:10px;padding:5px 12px;border-radius:999px;font-size:11px;font-weight:750}}
        .pd-pill-safe{{background:#DCFCE7!important;color:#16A34A!important}}
        .pd-pill-attention{{background:#FEF9C3!important;color:#A16207!important}}
        .pd-pill-alert{{background:#FEE2E2!important;color:#DC2626!important}}
        .pd-meta{{margin-top:12px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px;font-size:13px;color:#334155!important}}
        .pd-meta b{{color:#1E293B!important}}
        .pd-sec{{background:#FFFFFF!important;border:1px solid #E2E8F0!important;border-radius:18px;padding:16px;margin-bottom:12px}}
        .pd-sec h2{{margin:0 0 10px;font-size:16px;color:#1E293B!important;font-weight:700}}
        .pd-muted{{color:#64748B!important;font-size:12px}}
        .pd-empty{{padding:12px;color:#64748B!important;font-size:13px;background:#F8FAFC!important;border-radius:12px}}
        .pd-acts{{display:grid;gap:8px;margin-top:10px}}
        .pd-act-top{{display:flex;justify-content:space-between;font-size:12px;color:#334155!important;margin-bottom:4px}}
        .pd-act-top b{{color:#1E293B!important}}
        .pd-act-bar{{height:8px;background:#E2E8F0;border-radius:999px;overflow:hidden}}
        .pd-act-bar i{{display:block;height:100%;background:#2563EB;border-radius:999px}}
        .pd-row{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #F1F5F9}}
        .pd-row:last-child{{border-bottom:0}}
        .pd-row-title{{font-size:13px;font-weight:650;color:#1E293B!important}}
        .pd-row-sub{{margin-top:3px;font-size:12px;color:#64748B!important;line-height:1.45}}
        .pd-chip{{padding:4px 10px;border-radius:999px;background:#F1F5F9!important;font-size:10px;font-weight:700;color:#475569!important;white-space:nowrap;max-width:42%;text-align:right}}
        .pd-chip-rec-ok{{background:#DCFCE7!important;color:#166534!important}}
        .pd-focus{{background:#EFF6FF!important;border-radius:12px;padding:10px}}
        .pd-grid3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
        .pd-stat{{background:#F8FAFC!important;border:1px solid #E2E8F0!important;border-radius:14px;padding:12px}}
        .pd-stat-l{{font-size:11px;color:#64748B!important}}
        .pd-stat-n{{margin-top:4px;font-size:20px;font-weight:750;color:#0F172A!important}}
        .pd-snaps{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}}
        .pd-snap{{border:1px solid #E2E8F0!important;border-radius:12px;padding:10px;background:#F8FAFC!important}}
        .pd-snap-img{{width:100%;height:120px;object-fit:cover;border-radius:8px;display:block}}
        .cv-snap-zoom{{appearance:none;border:0;padding:0;margin:0 0 8px;background:transparent;width:100%;
          cursor:zoom-in;display:block;border-radius:8px;overflow:hidden}}
        .pd-snap-meta{{font-size:12px;color:#334155!important;line-height:1.45}}
        .pd-snap-meta b{{color:#1E293B!important}}
        .pd-player-panel{{background:#0F172A;border-radius:14px;overflow:hidden;padding:0}}
        .pd-player-panel .pd-empty{{margin:0;border-radius:14px;background:#F8FAFC!important}}
        .pd-player{{display:block;width:100%;max-height:min(56vh,420px);background:#000;aspect-ratio:16/9;object-fit:contain}}
        .pd-player-info{{margin-top:10px}}
        @media(max-width:800px){{.pd-meta,.pd-grid3,.pd-snaps{{grid-template-columns:1fr}}.pd-player{{max-height:42vh}}}}
      </style>
      <div class="pd-back">Patient details · {patient['name']}</div>
      <div class="pd-head">
        <div class="pd-name">{patient['name']}</div>
        <div class="pd-sub">ID {patient['id']} · {patient['room']}</div>
        <span class="{pill_class}" style="background:{pill[0]}!important;color:{pill[1]}!important">{status}</span>
        <div class="pd-meta">
          <div><b>Current activity</b><br>{activity}</div>
          <div><b>Last updated</b><br>{updated}</div>
          <div><b>Source</b><br>{source_label}</div>
          <div><b>Source detail</b><br>{source_note}</div>
          <div><b>Connection / file</b><br>{cam_status}</div>
          <div><b>Why this status</b><br>{reason}</div>
        </div>
      </div>

      <div class="pd-sec">
        <h2>Current monitoring session</h2>
        <div class="pd-muted">{latest_block}</div>
        <div class="pd-muted" style="margin-top:8px">
          {"The <b>Live Camera</b> panel above this page shows annotated frames from the laptop Camera Station."
           if source_type == "live" else
           "Use <b>Run room check</b> below for prerecorded rooms. Prerecorded video is never labeled live."}
        </div>
      </div>

      <div class="pd-sec">
        <h2>Today's activity summary</h2>
        {activity_body}
      </div>

      <div class="pd-sec">
        <h2>Activity and monitoring results</h2>
        {results}
      </div>

      {alert_block}
    </div>
    """


def _event_choices(store: EventStore, patient_id: str):
    choices = []
    for e in store.list_recent(patient_id, limit=30):
        if e.get("event_status") == "RESOLVED":
            continue
        choices.append((f"#{e['id']} {_event_title(e.get('event_type'))}", int(e["id"])))
    return choices


def open_patient_details(user, patient_id: str, store: EventStore, focus_event_id=None):
    """Navigate Home/Alerts → patient details (filtered)."""
    import gradio as gr
    from .care_home import _latest_snapshot_rgb

    patient = _assert_access(user, patient_id)
    focus_val = None
    if focus_event_id not in (None, ""):
        try:
            focus_val = int(focus_event_id)
        except (TypeError, ValueError):
            focus_val = None
    alert_context = focus_val is not None
    html = build_patient_details_html(
        store, user, patient_id,
        focus_event_id=focus_val,
        alert_context=alert_context,
    )
    if alert_context:
        note = (
            f"**{patient['name']}** · {patient['room']} · reviewing alert. "
            "Use **Care Response** / **Watch recording** below."
        )
    elif patient.get("source_type") == "live":
        note = (
            f"**{patient['name']}** · {patient['room']} — "
            "patient info and live feed. Open an **alert banner** to respond."
        )
    else:
        note = (
            f"**{patient['name']}** · {patient['room']} — "
            "patient info. Use **Run room check**, or open an **alert** to respond."
        )
    preview = _latest_snapshot_rgb(store, patient_id)
    live_banner = (
        f"**{patient['name']}** · {patient['room']} · "
        f"{'Live camera' if patient.get('source_type') == 'live' else 'Prerecorded video'}"
    )
    try:
        from .live_hub import get_shared_hub

        hub = get_shared_hub()
        if hub is not None and patient_id == hub.patient_id:
            live_frame = hub.get_preview(patient_id)
            if live_frame is not None:
                preview = live_frame
            snap = hub.snapshot()
            if snap.get("has_fresh_frame"):
                live_banner = (
                    f"### Live Camera · **LIVE**\n"
                    f"Session `#{snap.get('session_id')}` · **{snap.get('status')}** · "
                    f"{(snap.get('activity') or '—').replace('_', ' ').title()}"
                )
            elif patient.get("source_type") == "live":
                live_banner = (
                    "### Live Camera · **OFFLINE**\n"
                    "Waiting for fresh frames from `/camera-station` on the laptop."
                )
    except Exception:
        pass
    choices = _event_choices(store, patient_id)
    # Do not auto-select an alert when opening the patient card — only when
    # arriving from an alert banner (focus_val set).
    show_live = patient.get("source_type") == "live"
    return (
        html,
        note,
        preview,
        live_banner,
        patient_id,
        gr.update(choices=choices, value=focus_val),
        focus_val,
        gr.update(visible=True),   # details toolbar (Back / Run room check)
        gr.update(visible=show_live),  # live camera panel
        bool(alert_context),  # show Care Response + Watch recording
    )


def back_to_patients(user, store: EventStore, panel: str = "Home"):
    import gradio as gr
    from .care_home import refresh_dashboard

    cards, inbox, alerts = refresh_dashboard(user, store, panel=panel)
    return (
        cards,
        "Patient board — tap a card for details.",
        None,
        "Status / next action",
        "",  # clear patient id textbox so live timer stops targeting Ana
        alerts,
        None,
        gr.update(visible=False),  # details toolbar
        inbox,
        gr.update(visible=False),  # live panel
    )
