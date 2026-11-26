"""Patient details page for CareView (filtered to one patient)."""
from __future__ import annotations

import base64
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
from .playback import PlaybackError, session_meta_payload

_ACTIVITIES = ("sitting", "standing", "walking", "running", "lying", "unknown")

EMPTY_PLAYER_HTML = """
<div class="pd-player-panel" data-patient="" data-session="">
  <div class="pd-empty">Select a session with a ready recording to watch it here.
  The player stays empty until you choose one.</div>
</div>
"""

EMPTY_PLAYER_INFO_HTML = """
<div class="pd-player-info pd-muted">Session recording details appear after you select a playable session.
Older recordings may not play on every phone.</div>
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
    """Cleared session dropdown + player (used on patient switch / back)."""
    import gradio as gr

    return (
        gr.update(choices=[], value=None),
        EMPTY_PLAYER_HTML,
        EMPTY_PLAYER_INFO_HTML,
        None,
    )


def playback_panel_for_patient(
    store: EventStore,
    user,
    patient_id: str,
) -> tuple[Any, str, str, None]:
    """Populate session dropdown for the open patient; player stays empty (no auto-load)."""
    import gradio as gr

    _assert_access(user, patient_id)
    choices = session_playback_choices(store, patient_id)
    return (
        gr.update(choices=choices, value=None),
        EMPTY_PLAYER_HTML,
        EMPTY_PLAYER_INFO_HTML,
        None,
    )


def build_playback_player_html(
    *,
    patient_id: str,
    session_id: int,
    video_url: str,
) -> str:
    """Trusted P2 video_url only — never invent paths."""
    safe_url = html_lib.escape(video_url, quote=True)
    safe_pid = html_lib.escape(patient_id, quote=True)
    return f"""
<div class="pd-player-panel" data-patient="{safe_pid}" data-session="{int(session_id)}">
  <video class="pd-player" controls playsinline preload="metadata"
    src="{safe_url}"></video>
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
  </div>
</div>
"""


def select_session_playback(
    user,
    patient_id: str,
    session_id,
    store: EventStore,
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

    player = build_playback_player_html(
        patient_id=pid, session_id=sid, video_url=str(meta["video_url"]),
    )
    info = build_playback_info_html(store, user, pid, sid, meta)
    return player, info, sid


def build_patient_details_html(
    store: EventStore,
    user,
    patient_id: str,
    *,
    focus_event_id: int | None = None,
) -> str:
    """HTML for one patient's details page (server-filtered)."""
    patient = _assert_access(user, patient_id)
    reconcile_patient_status(store, patient_id)
    status, st, active = effective_status(store, patient_id)
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

    # Sessions (recording availability labels — playback controls live in Gradio widgets)
    if not sessions:
        if source_type == "video":
            session_rows = (
                '<div class="pd-empty">No sessions yet — use <b>Run room check</b> '
                "to analyze the linked room video and unlock <b>Watch recording</b>.</div>"
            )
        else:
            session_rows = (
                '<div class="pd-empty">No saved monitoring sessions yet for this patient.</div>'
            )
    else:
        bits = []
        for s in sessions:
            secs = s.get("activity_seconds") or {}
            top = ""
            if secs:
                lab, _ = max(secs.items(), key=lambda kv: kv[1])
                top = f" · mostly {str(lab).replace('_', ' ')}"
            sess_st = s.get("session_status") or "COMPLETED"
            final_act = (s.get("final_activity") or "—")
            if isinstance(final_act, str):
                final_act = final_act.replace("_", " ").title()
            avail = recording_availability(store, s)
            rec_label = avail["label"]
            chip = "pd-chip pd-chip-rec-ok" if avail["playable"] else "pd-chip"
            when = _fmt_session_when(s.get("started_at"))
            dur_s = float(s.get("recording_duration_s") or s.get("duration_s") or 0)
            bits.append(
                f"""
                <div class="pd-row">
                  <div>
                    <div class="pd-row-title">Session #{s.get('id')} · {sess_st} · {when}</div>
                    <div class="pd-row-sub">
                      {_fmt_dur(dur_s)} · {_rel_time(s.get('started_at'))} ·
                      {s.get('room_id') or patient.get('room') or ''} ·
                      final activity {final_act} · {s.get('event_count', 0)} events{top}<br>
                      Recording: <b>{html_lib.escape(rec_label)}</b>
                    </div>
                  </div>
                  <div class="{chip}">{html_lib.escape(rec_label)}</div>
                </div>
                """
            )
        session_rows = (
            '<div class="pd-muted" style="margin-bottom:8px">'
            "Use <b>Session recording</b> below the page tools to play a ready recording. "
            "Availability here does not guarantee every older file plays on every phone."
            "</div>"
            + "".join(bits)
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

    # Snapshots for current (non-resolved) alerts only
    snaps = []
    for e in current_events:
        path = _snapshot_abs(e.get("snapshot_path"))
        if not path:
            continue
        try:
            rel = str(path.relative_to(config.ROOT)).replace("\\", "/")
        except ValueError:
            rel = str(path)
        thumb = ""
        try:
            raw = path.read_bytes()
            if len(raw) <= 1_500_000:
                b64 = base64.b64encode(raw).decode("ascii")
                mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                thumb = (
                    f'<a href="data:{mime};base64,{b64}" target="_blank" rel="noopener">'
                    f'<img class="pd-snap-img" src="data:{mime};base64,{b64}" alt="event {e.get("id")}"/></a>'
                )
        except OSError:
            thumb = ""
        snaps.append(
            f"""
            <div class="pd-snap">
              {thumb or '<div class="pd-empty">Image unavailable</div>'}
              <div class="pd-snap-meta">
                <b>{_event_title(e.get('event_type'))}</b><br>
                {_rel_time(e.get('created_at'))} · event #{e.get('id')}<br>
                <span class="pd-muted">{rel}</span>
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
        .pd-snap-img{{width:100%;height:120px;object-fit:cover;border-radius:8px;display:block;margin-bottom:8px}}
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
        <h2>Monitoring session history</h2>
        {session_rows}
      </div>

      <div class="pd-sec">
        <h2>Current alerts</h2>
        {event_rows}
      </div>

      <div class="pd-sec">
        <h2>Current alert snapshots</h2>
        {snap_body}
      </div>

      <div class="pd-sec">
        <h2>Activity and monitoring results</h2>
        {results}
      </div>
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
    html = build_patient_details_html(
        store, user, patient_id, focus_event_id=focus_val,
    )
    if focus_val is not None:
        note = (
            f"**{patient['name']}** · {patient['room']} · event #{focus_val}. "
            "Use **Handle alert** or **Watch recording** below."
        )
    elif patient.get("source_type") == "live":
        note = (
            f"**{patient['name']}** · {patient['room']} — "
            "live feed below when Camera Station is on. "
            "Open **Watch recording** / **Handle alert** if needed."
        )
    else:
        note = (
            f"**{patient['name']}** · {patient['room']} — "
            "**Run room check**, or open **Watch recording** / **Handle alert** below."
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
    if focus_val is None and choices:
        focus_val = choices[0][1]
    show_live = patient.get("source_type") == "live"
    return (
        html,
        note,
        preview,
        live_banner,
        patient_id,
        gr.update(choices=choices, value=focus_val),
        focus_val,
        gr.update(visible=True),   # details toolbar
        gr.update(visible=show_live),  # live camera panel (Ana)
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
