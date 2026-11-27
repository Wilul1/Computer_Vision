"""CareView-style Care Home helpers (Phase L2) — dashboard HTML + room checks."""
from __future__ import annotations

import html as html_lib
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from . import config
from .alert_actions import format_decision
from .alert_center import (
    build_alert_center_html,
    event_display_title,
    list_caregiver_active_alerts,
    list_caregiver_patient_alerts,
    outcome_label,
    parse_alert_bundle,
    session_ready_for_event,
)
from .care_demo import get_caregiver, get_patient, patients_for, resolve_video_path
from .event_store import ACTIVE_STATUSES, EventStore
from .monitoring import MonitoringSession
from .playback import event_snapshot_url, recordings_root


def _attach_source_video_recording(
    store: EventStore,
    *,
    session_id: int,
    patient_id: str,
    source_path: Path,
    started_at: str,
    ended_at: str,
    duration_s: float,
) -> dict | None:
    """Copy the room-check source clip under outputs/recordings/ as READY playback."""
    src = Path(source_path)
    if not src.is_file():
        return None
    out_dir = recordings_root() / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"session_{int(session_id)}_{stamp}.mp4"
    try:
        shutil.copy2(src, dest)
    except OSError:
        return None

    width = height = 0
    fps = 0.0
    cap = cv2.VideoCapture(str(dest))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    try:
        size = int(dest.stat().st_size)
    except OSError:
        size = 0

    # Store path relative to project root when possible (portable across machines).
    try:
        stored = str(dest.resolve().relative_to(config.ROOT.resolve())).replace("\\", "/")
    except ValueError:
        stored = str(dest.resolve())

    return store.update_session_recording(
        int(session_id),
        recording_path=stored,
        recording_status="READY",
        recording_started_at=started_at,
        recording_ended_at=ended_at,
        recording_duration_s=float(duration_s or 0),
        recording_fps=fps or None,
        recording_width=width or None,
        recording_height=height or None,
        recording_codec="copy",
        recording_bytes=size,
        recording_error="",
    )

# CareView reference palette
CV = {
    "safe": "#16A34A",
    "attention": "#EAB308",
    "alert": "#DC2626",
    "navy": "#1A2B4B",
    "blue": "#2563EB",
    "muted": "#64748B",
    "card": "#FFFFFF",
    "bg": "#F1F5F9",
    "line": "#E2E8F0",
}

# Open events can keep a patient elevated even if the last frame looked fine.
_ALERT_EVENTS = frozenset({"possible_fall"})
_ATTENTION_EVENTS = frozenset({
    "inactive", "out_of_view", "partial_body", "camera_unavailable",
})
_STATUS_RANK = {"SAFE": 0, "ATTENTION": 1, "ALERT": 2}


def _rel_time(iso: str | None) -> str:
    if not iso:
        return "Not checked"
    return iso.replace("T", " ")[:16] + " UTC"


def _greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _event_title(event_type: str | None) -> str:
    """Short caregiver-facing title (possible fall — never 'confirmed')."""
    return event_display_title(event_type, short=True)


def _parse_iso(iso: str | None):
    raw = (iso or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_hist_when(iso: str | None) -> str:
    dt = _parse_iso(iso)
    if dt is None:
        return _rel_time(iso)
    return dt.strftime("%b %d, %Y · %I:%M %p").replace(" 0", " ")


def _response_secs(event: dict) -> float | None:
    start = _parse_iso(event.get("created_at"))
    end = _parse_iso(event.get("resolved_at") or event.get("acknowledged_at"))
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _fmt_response(secs: float | None) -> str:
    if secs is None:
        return "—"
    s = int(round(secs))
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m}m {rem:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _initials(name: str) -> str:
    parts = [p for p in str(name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _build_resolved_history_html(store: EventStore, caregiver_id: str) -> tuple[str, int]:
    """History board: clickable event list + detail/snapshot panel (no icons)."""
    # Lazy import — patient_details already imports care_home helpers.
    from .patient_details import (
        build_incident_timeline_html,
        incident_marker_payload,
    )

    patients = patients_for(caregiver_id)
    if not patients:
        empty = (
            '<div class="cv-alert-empty">No patients yet. '
            "Resolved alerts will appear here.</div>"
        )
        return empty, 0

    pids = [p["id"] for p in patients]
    by_id = {p["id"]: p for p in patients}
    resolved: list[dict] = []
    for e in store.list_for_patients(pids, limit=80):
        if e.get("event_status") == "RESOLVED":
            resolved.append(e)
    if not resolved:
        empty = (
            '<div class="cv-alert-empty">No resolved events yet. '
            "Resolve an open alert and it appears here with its snapshot and recording.</div>"
        )
        return empty, 0

    # Newest first for the timeline.
    resolved.sort(key=lambda e: int(e.get("id") or 0), reverse=True)
    total = len(resolved)
    fall_n = sum(1 for e in resolved if e.get("event_type") == "possible_fall")
    resp_vals = [s for e in resolved if (s := _response_secs(e)) is not None]
    avg_resp = (sum(resp_vals) / len(resp_vals)) if resp_vals else None
    first_id = str(resolved[0].get("id") or "")

    list_rows: list[str] = []
    detail_panels: list[str] = []
    for e in resolved[:40]:
        eid = str(e.get("id") or "")
        pid = str(e.get("patient_id") or "")
        p = by_id.get(pid) or get_patient(pid) or {}
        pname = str(p.get("name") or pid or "Patient")
        room = str(p.get("room") or "—")
        title = _event_title(e.get("event_type"))
        when = _fmt_hist_when(e.get("created_at"))
        is_fall = e.get("event_type") == "possible_fall"
        selected = " is-selected" if eid == first_id else ""
        active = " is-active" if eid == first_id else ""
        reason = (
            str(e.get("reason_text") or e.get("action_text") or "").strip()
            or (
                "The system detected a possible sudden fall."
                if is_fall
                else "Resolved caregiver alert."
            )
        )
        out_title = outcome_label(e.get("handle_outcome")) if e.get("handle_outcome") else "—"
        note = str(e.get("handle_note") or "").strip() or "—"
        who = str(e.get("handled_by") or "caregiver")
        resolved_when = _fmt_hist_when(e.get("resolved_at") or e.get("created_at"))
        resp = _fmt_response(_response_secs(e))
        snap_rel = str(e.get("snapshot_path") or "").strip()
        snap_path = Path(snap_rel) if snap_rel else None
        if snap_path is not None and not snap_path.is_absolute():
            snap_path = config.ROOT / snap_path
        has_snap = bool(snap_path and snap_path.is_file())
        snap_count = 1 if has_snap else 0
        if has_snap:
            snap_src = html_lib.escape(event_snapshot_url(int(eid), pid), quote=True)
            snap_html = (
                f'<button type="button" class="cv-snap-zoom cv-hist2-snap-zoom" '
                f'aria-label="Enlarge snapshot">'
                f'<img class="cv-hist2-snap" src="{snap_src}" alt="Event {html_lib.escape(eid)} snapshot">'
                f"</button>"
            )
        else:
            snap_html = '<div class="cv-hist2-snap-empty">No snapshot on file</div>'
        ready = session_ready_for_event(store, e)
        if ready:
            sid_i = int(ready["session_id"])
            sess_row = store.get_session(sid_i) or {}
            marker = incident_marker_payload(e, sess_row)
            timeline = build_incident_timeline_html(marker)
            video_src = html_lib.escape(
                f"/api/sessions/{sid_i}/video?patient_id={pid}",
                quote=True,
            )
            video_html = (
                f'<div class="pd-player-panel cv-hist2-player" data-session="{sid_i}">'
                f'<video class="cv-hist2-video pd-player" controls playsinline preload="metadata" '
                f'src="{video_src}"></video>'
                f"{timeline}"
                f'<div class="cv-hist2-row-time" style="margin-top:8px">'
                f"Session #{sid_i} · issue recording</div>"
                f"</div>"
            )
            video_stat = "Ready"
        else:
            video_html = (
                '<div class="cv-hist2-snap-empty">No session recording linked to this alert.</div>'
            )
            video_stat = "None"
        blurb = (
            "The system detected a possible sudden fall."
            if is_fall
            else reason
        )
        list_rows.append(
            f"""
            <button type="button" class="cv-hist2-row{selected}"
              data-hist-select="{html_lib.escape(eid)}"
              data-patient-id="{html_lib.escape(pid)}"
              data-event-id="{html_lib.escape(eid)}">
              <div class="cv-hist2-avatar">{html_lib.escape(_initials(pname))}</div>
              <div class="cv-hist2-row-body">
                <div class="cv-hist2-row-name">{html_lib.escape(pname)}</div>
                <div class="cv-hist2-row-title">{html_lib.escape(title)}</div>
                <div class="cv-hist2-row-time">{html_lib.escape(when)}</div>
              </div>
              <span class="cv-hist2-pill">Resolved</span>
            </button>
            """
        )
        detail_panels.append(
            f"""
            <div class="cv-hist2-detail{active}" data-hist-detail="{html_lib.escape(eid)}">
              <div class="cv-hist2-detail-h">Event Details</div>
              <div class="cv-hist2-alert{' cv-hist2-alert-fall' if is_fall else ''}">
                <div class="cv-hist2-alert-title">{html_lib.escape(title)}</div>
                <div class="cv-hist2-alert-meta">{html_lib.escape(when)}</div>
                <div class="cv-hist2-alert-blurb">{html_lib.escape(blurb)}</div>
                <div class="cv-hist2-alert-stats">
                  <div><span>Snapshots</span><b>{snap_count} image{'s' if snap_count != 1 else ''}</b></div>
                  <div><span>Recording</span><b>{video_stat}</b></div>
                  <div><span>Response time</span><b>{html_lib.escape(resp)}</b></div>
                  <div><span>Status</span><b class="ok">Resolved</b></div>
                </div>
              </div>
              <div class="cv-hist2-card">
                <div class="cv-hist2-card-h">Patient</div>
                <div class="cv-hist2-patient">
                  <div class="cv-hist2-avatar lg">{html_lib.escape(_initials(pname))}</div>
                  <div>
                    <div class="cv-hist2-row-name">{html_lib.escape(pname)}</div>
                    <div class="cv-hist2-row-time">Room {html_lib.escape(room)}</div>
                  </div>
                </div>
              </div>
              <div class="cv-hist2-card">
                <div class="cv-hist2-card-h">Caregiver response</div>
                <div class="cv-hist2-kv"><span>Outcome</span><b>{html_lib.escape(out_title)}</b></div>
                <div class="cv-hist2-kv"><span>Care notes</span><b>{html_lib.escape(note)}</b></div>
                <div class="cv-hist2-kv"><span>Resolved by</span><b>{html_lib.escape(who)}</b></div>
                <div class="cv-hist2-kv"><span>Resolved at</span><b>{html_lib.escape(resolved_when)}</b></div>
              </div>
              <div class="cv-hist2-card">
                <div class="cv-hist2-card-h">Event snapshots</div>
                <div class="cv-hist2-snaps">{snap_html}</div>
              </div>
              <div class="cv-hist2-card">
                <div class="cv-hist2-card-h">Session recording</div>
                {video_html}
              </div>
            </div>
            """
        )

    html = f"""
    <style>
      .cv-hist2{{display:grid;gap:12px}}
      .cv-hist2-summary{{font-size:13px;color:#64748B;font-weight:650;padding:2px 2px 4px}}
      .cv-hist2-summary b{{color:#0F172A;font-weight:750}}
      .cv-hist2-summary .fall{{color:#B91C1C;font-weight:750}}
      .cv-hist2-grid{{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);gap:14px;align-items:start}}
      .cv-hist2-panel{{background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:14px;min-height:280px}}
      .cv-hist2-panel-h{{font-size:15px;font-weight:750;color:#0F172A;margin:0 0 12px}}
      .cv-hist2-list{{display:grid;gap:8px;max-height:min(70vh,720px);overflow:auto}}
      .cv-hist2-row{{display:flex;gap:12px;align-items:center;width:100%;text-align:left;
        padding:12px;border:1px solid #E2E8F0;border-radius:14px;background:#fff;cursor:pointer;
        color:#0F172A;font:inherit}}
      .cv-hist2-row:hover{{border-color:#93C5FD;background:#F8FAFC}}
      .cv-hist2-row.is-selected{{border-color:#2563EB;background:#EFF6FF;box-shadow:0 0 0 1px #2563EB inset}}
      .cv-hist2-avatar{{width:40px;height:40px;border-radius:999px;background:#DBEAFE;color:#1D4ED8;
        display:grid;place-items:center;font-size:12px;font-weight:750;flex:0 0 auto}}
      .cv-hist2-avatar.lg{{width:48px;height:48px;font-size:14px}}
      .cv-hist2-row-body{{flex:1;min-width:0}}
      .cv-hist2-row-name{{font-size:13px;font-weight:700;color:#0F172A}}
      .cv-hist2-row-title{{margin-top:2px;font-size:12px;font-weight:650;color:#B91C1C}}
      .cv-hist2-row-time{{margin-top:2px;font-size:11px;color:#64748B}}
      .cv-hist2-pill{{padding:4px 10px;border-radius:999px;background:#DCFCE7;color:#166534;
        font-size:10px;font-weight:750;white-space:nowrap}}
      .cv-hist2-detail{{display:none}}
      .cv-hist2-detail.is-active{{display:block}}
      .cv-hist2-detail-h{{font-size:15px;font-weight:750;color:#0F172A;margin:0 0 12px}}
      .cv-hist2-alert{{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:14px;margin-bottom:12px}}
      .cv-hist2-alert-fall{{background:#FEF2F2;border-color:#FECACA}}
      .cv-hist2-alert-title{{font-size:16px;font-weight:750;color:#0F172A}}
      .cv-hist2-alert-fall .cv-hist2-alert-title{{color:#7F1D1D}}
      .cv-hist2-alert-meta{{margin-top:4px;font-size:12px;color:#64748B}}
      .cv-hist2-alert-blurb{{margin-top:8px;font-size:13px;color:#334155;line-height:1.45}}
      .cv-hist2-alert-stats{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}}
      .cv-hist2-alert-stats span{{display:block;font-size:10px;font-weight:700;color:#94A3B8;margin-bottom:2px}}
      .cv-hist2-alert-stats b{{font-size:13px;color:#0F172A}}
      .cv-hist2-alert-stats b.ok{{color:#15803D}}
      .cv-hist2-card{{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:14px;margin-bottom:12px}}
      .cv-hist2-card-h{{font-size:13px;font-weight:750;color:#0F172A;margin:0 0 10px}}
      .cv-hist2-patient{{display:flex;gap:12px;align-items:center}}
      .cv-hist2-kv{{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-top:1px solid #F1F5F9;font-size:12px}}
      .cv-hist2-kv:first-of-type{{border-top:0;padding-top:0}}
      .cv-hist2-kv span{{color:#64748B}}
      .cv-hist2-kv b{{color:#0F172A;font-weight:650;text-align:right;max-width:65%}}
      .cv-hist2-snaps{{display:grid;gap:10px}}
      .cv-hist2-snap{{width:100%;max-height:260px;object-fit:cover;border-radius:12px;border:1px solid #E2E8F0;background:#F8FAFC;display:block}}
      .cv-hist2-snap-zoom{{appearance:none;border:0;padding:0;margin:0;background:transparent;width:100%;
        cursor:zoom-in;display:block;border-radius:12px;overflow:hidden}}
      .cv-hist2-video{{width:100%;max-height:280px;border-radius:12px;background:#0F172A;display:block}}
      .cv-hist2-snap-empty{{padding:28px;text-align:center;color:#94A3B8;font-size:13px;background:#F8FAFC;border-radius:12px}}
      @media(max-width:900px){{
        .cv-hist2-grid{{grid-template-columns:1fr}}
        .cv-hist2-list-panel{{order:1}}
        .cv-hist2-detail-panel{{order:2}}
        .cv-hist2-alert-stats{{grid-template-columns:1fr}}
        .cv-hist2-panel{{min-height:0}}
      }}
    </style>
    <div class="cv-hist2" id="cv-hist2" data-hist-active="{html_lib.escape(first_id)}">
      <div class="cv-hist2-summary">
        <b>{total}</b> closed · <span class="fall">{fall_n} falls</span> ·
        avg response {html_lib.escape(_fmt_response(avg_resp))}
      </div>
      <div class="cv-hist2-grid">
        <div class="cv-hist2-panel cv-hist2-list-panel">
          <div class="cv-hist2-panel-h">Closed cases</div>
          <div class="cv-hist2-list">{"".join(list_rows)}</div>
        </div>
        <div class="cv-hist2-panel cv-hist2-detail-panel">
          {"".join(detail_panels)}
        </div>
      </div>
    </div>
    """
    return html, total


def _status_from_event_type(event_type: str) -> str:
    if event_type in _ALERT_EVENTS:
        return "ALERT"
    if event_type in _ATTENTION_EVENTS:
        return "ATTENTION"
    return "SAFE"


def effective_status(store: EventStore, patient_id: str) -> tuple[str, dict, list[dict]]:
    """Board status = live/last signal, elevated by still-open events.

    Without open events, a stale ALERT/ATTENTION in patient_state alone must not
    keep the card red after the caregiver already resolved everything.
    """
    st = store.get_patient_state(patient_id) or {}
    active = store.list_active(patient_id)
    status = st.get("status") or "SAFE"
    if status not in _STATUS_RANK:
        status = "SAFE"

    # Prefer live hub status when this is the live patient with a fresh frame.
    try:
        from .live_hub import get_shared_hub

        hub = get_shared_hub()
        if hub is not None and hub.patient_id == patient_id:
            snap = hub.snapshot()
            if snap.get("has_fresh_frame"):
                live_st = snap.get("status") or "SAFE"
                if live_st in _STATUS_RANK:
                    status = live_st
    except Exception:
        pass

    for e in active:
        cand = _status_from_event_type(e.get("event_type") or "")
        if _STATUS_RANK[cand] > _STATUS_RANK[status]:
            status = cand

    # No open findings + not live-fresh → don't keep sticky ALERT from old state.
    if not active and status in ("ALERT", "ATTENTION"):
        live_fresh = False
        try:
            from .live_hub import get_shared_hub

            hub = get_shared_hub()
            if hub is not None and hub.patient_id == patient_id:
                live_fresh = bool(hub.snapshot().get("has_fresh_frame"))
        except Exception:
            live_fresh = False
        if not live_fresh:
            status = "SAFE"
    return status, st, active


def reconcile_patient_status(store: EventStore, patient_id: str) -> str:
    """Persist board status (open events elevate; clears sticky ALERT when none)."""
    status, st, _active = effective_status(store, patient_id)
    store.upsert_patient_state(
        patient_id,
        status=status,
        activity=st.get("activity") or "",
        reason=st.get("reason") or "",
        action=st.get("action") or "",
    )
    return status


def _status_counts(store: EventStore, caregiver_id: str) -> dict[str, int]:
    counts = {"SAFE": 0, "ATTENTION": 0, "ALERT": 0}
    for p in patients_for(caregiver_id):
        status, _, _ = effective_status(store, p["id"])
        counts[status] += 1
    return counts


def _open_alert_total(store: EventStore, caregiver_id: str) -> int:
    return sum(store.count_open(p["id"]) for p in patients_for(caregiver_id))


def _open_events_for_caregiver(store: EventStore, caregiver_id: str, limit: int = 8) -> list[dict]:
    """Active findings for caregiver (flat), prioritized (High → Medium → Low)."""
    return list_caregiver_active_alerts(store, caregiver_id, limit=limit, status="Active")


def _open_patient_alerts_for_caregiver(
    store: EventStore, caregiver_id: str, limit: int = 20,
) -> list[dict]:
    """Active patient-level alert groups for caregiver (P4.1)."""
    return list_caregiver_patient_alerts(store, caregiver_id, limit=limit, status="Active")


def normalize_panel(panel: str | None) -> str:
    raw = (panel or "Home").strip()
    for name in ("Home", "Alerts", "History", "Settings"):
        if raw.lower() == name.lower():
            return name
    return "Home"


def build_dashboard_html(
    store: EventStore,
    caregiver_id: str,
    caregiver_name: str = "caregiver1",
    panel: str | None = "Home",
    alert_filters: dict | None = None,
) -> str:
    """CareView main panel HTML (nav is a Gradio Radio — Gradio strips HTML onclick)."""
    panel = normalize_panel(panel)
    for p in patients_for(caregiver_id):
        reconcile_patient_status(store, p["id"])
    counts = _status_counts(store, caregiver_id)
    open_n = _open_alert_total(store, caregiver_id)
    patient_n = len(patients_for(caregiver_id))

    cards = []
    for p in patients_for(caregiver_id):
        status, st, active = effective_status(store, p["id"])
        activity = (st.get("activity") or "").replace("_", " ").title()
        # Keep subtitle short for phone rows (mockup-style).
        if active and status != "SAFE":
            sub = _event_title(active[0].get("event_type"))
        elif activity and activity.lower() not in ("", "unknown"):
            sub = activity
        elif p.get("source_type") == "live":
            sub = "Live camera"
        elif p.get("source_type") == "video":
            sub = "Demo video"
        else:
            sub = "No monitoring yet"
        pill = {
            "SAFE": ("#DCFCE7", CV["safe"], "Safe"),
            "ATTENTION": ("#FEF9C3", "#A16207", "Attention"),
            "ALERT": ("#FEE2E2", CV["alert"], "Alert"),
        }.get(status, ("#F1F5F9", CV["muted"], status.title()))
        cards.append(
            f"""
            <div class="cv-patient-card cv-clickable" data-patient-id="{p['id']}"
                 role="button" tabindex="0" title="Open {p['name']}">
              <div class="cv-patient-row">
                <div class="cv-avatar">{html_lib.escape(str(p.get('initials', p['name'][:1])))}</div>
                <div class="cv-patient-who">
                  <div class="cv-patient-name">{html_lib.escape(str(p['name']))}</div>
                  <div class="cv-patient-room">{html_lib.escape(str(p['room']))} · {html_lib.escape(sub)}</div>
                </div>
                <span class="cv-pill" style="background:{pill[0]};color:{pill[1]}">{pill[2]}</span>
              </div>
              <div class="cv-open-link">Open patient →</div>
            </div>
            """
        )
    if not cards:
        cards.append(
            """
            <div class="cv-patient-card" style="min-height:120px">
              <div class="cv-patient-name">No patients yet</div>
              <div class="cv-patient-room">Add a name and room under Settings.</div>
              <div class="cv-card-hint">Settings → Add patient</div>
            </div>
            """
        )

    def _patient_alert_row(g: dict) -> str:
        """Home Recent Alerts: one row per patient (P4.1 bundle)."""
        pname = g.get("patient_name") or g.get("patient_id") or "Patient"
        room = g.get("room") or ""
        title = g.get("short_title") or _event_title(
            (g.get("primary_event") or {}).get("event_type")
        )
        eid = int(g.get("primary_event_id") or 0)
        n = int(g.get("finding_count") or 1)
        token = g.get("bundle_token") or str(eid)
        extra = f" · {n} findings" if n > 1 else ""
        status = g.get("overall_status") or "OPEN"
        accent = "#B91C1C"
        return f"""
            <div class="cv-alert-row cv-clickable" data-event-id="{eid}"
                 data-patient-id="{g.get('patient_id') or ''}"
                 data-bundle="{token}" role="button" tabindex="0"
                 title="Review patient alert ({n} finding{'s' if n != 1 else ''})">
              <div class="cv-alert-thumb" style="background:#FEE2E2;color:#DC2626">#{eid}</div>
              <div class="cv-alert-body">
                <div class="cv-alert-title">{pname} — {room}</div>
                <div class="cv-alert-desc" style="color:{accent}">{title}{extra}</div>
                <div class="cv-alert-time">{_rel_time(g.get('latest_created_at'))} · {status}</div>
              </div>
              <div class="cv-alert-open" style="color:{accent};background:#FEE2E2">Open</div>
            </div>
            """

    open_groups = _open_patient_alerts_for_caregiver(store, caregiver_id, limit=20)
    alerts_html = [_patient_alert_row(g) for g in open_groups]
    if not alerts_html:
        alerts_html = ['<div class="cv-alert-empty">No open alerts right now.</div>']
    # Patient-level open count for Home (one row == one patient alert).
    open_patient_n = len(open_groups)

    history_body, hist_n = _build_resolved_history_html(store, caregiver_id)

    settings_rooms = []
    for p in patients_for(caregiver_id):
        src = "Live camera" if p.get("source_type") == "live" else "Prerecorded room video"
        settings_rooms.append(
            f'<div class="cv-alert-row"><div class="cv-alert-body">'
            f'<div class="cv-alert-title">{p["name"]} — {p["room"]}</div>'
            f'<div class="cv-alert-time">{src}</div></div></div>'
        )

    greet = _greeting()
    if open_patient_n == 0 and counts["ALERT"] == 0 and counts["ATTENTION"] == 0 and patient_n:
        home_sub = f"{patient_n} patients · all safe right now"
    else:
        home_sub = (
            f"{patient_n} patients · {counts['SAFE']} safe · "
            f"{counts['ATTENTION']} attention · {counts['ALERT']} alert"
            + (f" · {open_patient_n} open" if open_patient_n else "")
        )
    titles = {
        "Home": (
            greet,
            home_sub,
        ),
        "Alerts": (
            "Alerts",
            f"{open_patient_n} open patient case{'s' if open_patient_n != 1 else ''} · "
            f"{open_n} finding{'s' if open_n != 1 else ''}",
        ),
        "History": (
            "History",
            f"{hist_n} closed case{'s' if hist_n != 1 else ''}",
        ),
        "Settings": ("Settings", f"Signed in as {caregiver_name}"),
    }
    hello, sub = titles[panel]
    filters = alert_filters or {}

    if panel == "Home":
        body = f"""
          <div class="cv-stats cv-stats-compact">
            <div class="cv-stat cv-stat-safe">
              <div class="cv-stat-n">{counts['SAFE']}</div><div class="cv-stat-l">Safe</div>
            </div>
            <div class="cv-stat cv-stat-att">
              <div class="cv-stat-n">{counts['ATTENTION']}</div><div class="cv-stat-l">Attention</div>
            </div>
            <div class="cv-stat cv-stat-alert">
              <div class="cv-stat-n">{counts['ALERT']}</div><div class="cv-stat-l">Alert</div>
            </div>
          </div>
          <div class="cv-section-head">
            <h2>My patients</h2>
            <span class="cv-link">{patient_n} rooms</span>
          </div>
          <div class="cv-patients">{''.join(cards)}</div>
          <div class="cv-section-head" style="margin-top:18px">
            <h2>Open alerts</h2>
            <span class="cv-link">{open_patient_n} open</span>
          </div>
          <div class="cv-alerts">{''.join(alerts_html[:4])}</div>
        """
    elif panel == "Alerts":
        body = build_alert_center_html(
            store,
            caregiver_id,
            priority=filters.get("priority"),
            patient_id=filters.get("patient_id"),
            room=filters.get("room"),
            status=filters.get("status"),
            event_type=filters.get("event_type"),
            category=filters.get("category"),
            date_from=filters.get("date_from"),
            date_to=filters.get("date_to"),
        )
    elif panel == "History":
        body = history_body
    else:
        if not settings_rooms:
            settings_rooms = [
                '<div class="cv-alert-empty">No rooms yet — add a patient below.</div>'
            ]
        body = f"""
          <div class="cv-alerts" style="margin-bottom:14px">
            <div class="cv-alert-row"><div class="cv-alert-body">
              <div class="cv-alert-title">Account</div>
              <div class="cv-alert-time">{caregiver_name} · demo login caregiver / care123</div>
            </div></div>
          </div>
          <div class="cv-section-head"><h2>Your patients</h2></div>
          <div class="cv-alerts">{''.join(settings_rooms)}</div>
          <p class="cv-muted" style="margin-top:12px">
            Use <b>Add patient</b> below for a name and room. Optional live camera via
            laptop <b>/camera-station</b>.
          </p>
        """

    return f"""
    <div id="careview-root" class="cv-app cv-app-mainonly">
      <section class="cv-main">
        <div class="cv-top">
          <div>
            <div class="cv-hello">{hello}</div>
            <div class="cv-sub">{sub}</div>
          </div>
        </div>
        {body}
      </section>
    </div>
    """


def switch_panel(user, panel, store: EventStore, alert_filters: dict | None = None):
    if not user:
        return (
            '<div id="careview-root"><p style="color:#64748B;padding:12px">Sign in first.</p></div>'
        )
    return build_dashboard_html(
        store, user["id"], user.get("name", "caregiver1"), panel=panel,
        alert_filters=alert_filters,
    )



def login(username: str, password: str, store: EventStore):
    import gradio as gr

    user = get_caregiver((username or "").strip(), password or "")
    empty = pd.DataFrame(columns=["id", "patient", "time", "type", "status", "action", "snapshot"])
    if not user:
        return (
            None,
            "Login failed. Use **caregiver** / **care123**.",
            """
            <div id="careview-root" class="cv-app">
              <aside class="cv-rail">
                <div class="cv-brand"><div class="cv-brand-mark">⌂</div>
                <div><div class="cv-brand-name">CareView</div>
                <div class="cv-brand-sub">Care Home Monitor</div></div></div>
                <nav class="cv-nav">
                  <div class="cv-nav-item cv-nav-active">Home</div>
                  <div class="cv-nav-item">Alerts</div>
                  <div class="cv-nav-item">History</div>
                  <div class="cv-nav-item">Settings</div>
                </nav>
                <div class="cv-nav-foot">Logout</div>
              </aside>
              <section class="cv-main">
                <div class="cv-hello">Sign in failed</div>
                <div class="cv-sub">Use caregiver / care123</div>
              </section>
            </div>
            """,
            empty,
            gr.update(value="Home"),
            gr.update(choices=[], value=None),
        )
    dash = build_dashboard_html(store, user["id"], user.get("name", "caregiver1"), panel="Home")
    inbox = _inbox_df(store, user["id"])
    alert_choices = gr_open_event_choices(store, user["id"])
    alert_val = alert_choices[0][1] if alert_choices else None
    n = len(patients_for(user["id"]))
    ready = (
        f"Signed in as **{user.get('name', 'caregiver1')}**. "
        + (f"{n} patient{'s' if n != 1 else ''} on your board."
           if n else "No patients yet — open **Settings** and add one.")
    )
    return (
        user,
        ready,
        dash,
        inbox,
        gr.update(value="Home"),
        gr.update(choices=alert_choices, value=alert_val),
    )


def refresh_dashboard(
    user, store: EventStore, panel: str | None = "Home",
    alert_filters: dict | None = None,
):
    import gradio as gr

    empty = pd.DataFrame(columns=["id", "patient", "time", "type", "status", "action", "snapshot"])
    if not user:
        return (
            '<div id="careview-root"><p style="color:#64748B;padding:12px">Sign in first.</p></div>',
            empty,
            gr.update(choices=[], value=None),
        )
    alert_choices = gr_open_event_choices(store, user["id"])
    alert_val = alert_choices[0][1] if alert_choices else None
    return (
        build_dashboard_html(
            store, user["id"], user.get("name", "caregiver1"), panel=panel,
            alert_filters=alert_filters,
        ),
        _inbox_df(store, user["id"]),
        gr.update(choices=alert_choices, value=alert_val),
    )


def normalize_alert_filters(
    priority=None, patient_id=None, status=None, event_type=None,
    date_from=None, date_to=None,
) -> dict:
    """Normalize Gradio filter widgets into kwargs for Alert Center."""
    def _clean(val, *, prefixes=("All",)):
        if val in (None, ""):
            return None
        s = str(val).strip()
        if not s:
            return None
        for p in prefixes:
            if s.lower().startswith(p.lower()):
                return None
        return s

    st = _clean(status)
    # UI label "All active" → treat as Active (OPEN+ACK).
    if st and st.lower() == "active":
        st = "Active"
    return {
        "priority": _clean(priority),
        "patient_id": _clean(patient_id),
        "status": st,
        "event_type": _clean(event_type, prefixes=("All", "All types")),
        "date_from": _clean(date_from),
        "date_to": _clean(date_to),
    }


def gr_patient_choices(caregiver_id: str | None):
    if not caregiver_id:
        return []
    return [(f"{p['name']} ({p['room']})", p["id"]) for p in patients_for(caregiver_id)]


def gr_open_event_choices(store: EventStore, caregiver_id: str | None):
    """Dropdown labels for patient alert bundles — values are reviewed-ID tokens."""
    if not caregiver_id:
        return []
    choices = []
    for g in _open_patient_alerts_for_caregiver(store, caregiver_id, limit=20):
        name = g.get("patient_name") or g.get("patient_id") or "Patient"
        title = g.get("short_title") or _event_title(
            (g.get("primary_event") or {}).get("event_type")
        )
        pri = g.get("priority") or ""
        n = int(g.get("finding_count") or 1)
        extra = f" (+{n - 1} more)" if n > 1 else ""
        token = g.get("bundle_token") or str(g.get("primary_event_id") or "")
        choices.append((f"[{pri}] {name} · {title}{extra}", token))
    return choices


def _inbox_df(store: EventStore, caregiver_id: str) -> pd.DataFrame:
    ids = [p["id"] for p in patients_for(caregiver_id)]
    raws = store.list_for_patients(ids, limit=30)
    enriched = []
    for raw in raws:
        p = get_patient(raw["patient_id"])
        name = p["name"] if p else raw["patient_id"]
        enriched.append([
            raw.get("id"),
            name,
            raw.get("created_at", ""),
            raw.get("event_type", ""),
            raw.get("event_status", ""),
            raw.get("action_text", ""),
            raw.get("snapshot_path") or "",
        ])
    headers = ["id", "patient", "time", "type", "status", "action", "snapshot"]
    return pd.DataFrame(enriched, columns=headers)


def prepare_video_patient_demo(
    store: EventStore,
    patient_id: str,
    *,
    new_pipeline,
    lock,
    progress=None,
) -> dict:
    """Analyze a video patient's clip now — session, alerts, snapshot, playback ready."""
    patient = get_patient(patient_id)
    if not patient:
        raise ValueError(f"Patient not found: {patient_id}")
    if patient.get("source_type") != "video":
        raise ValueError(f"{patient.get('name') or patient_id} is not a video demo patient.")
    path = resolve_video_path(patient)
    if path is None:
        raise FileNotFoundError(
            f"Video missing for {patient['name']}: {patient.get('video_path')}"
        )

    monitor = MonitoringSession(
        patient_id=patient["id"], source=f"room:{patient['room']}", store=store,
    )
    labels: list[str] = []
    last = None
    annotated_rgb = None
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    events_before = len(store.list_recent(patient["id"], limit=500))
    with lock:
        pipe = new_pipeline(live=False)
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        # Demo speed: ~10 processed frames/sec of source video (still enough for fall rules).
        stride = max(1, int(round(fps / 10.0)))
        sec_per_processed = stride / fps
        i = 0
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    r = pipe.process(frame, i / fps)
                    mon = monitor.update(r, r.get("annotated"))
                    last = mon
                    labels.append(r["activity"])
                    if r.get("annotated") is not None:
                        annotated_rgb = cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB)
                    processed += 1
                    if progress is not None and processed % 3 == 0:
                        progress(
                            min((i + 1) / max(n, 1), 0.99),
                            desc=f"{patient['name']} {i + 1}/{n}",
                        )
                i += 1
        finally:
            cap.release()
            pipe.close()

    if last is None:
        raise RuntimeError(f"Could not read the room video for {patient['name']}.")
    if progress is not None:
        progress(1.0, desc=f"{patient['name']} ready")

    ended_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    duration_s = processed * sec_per_processed
    label_counts = dict(Counter(labels))
    activity_seconds = {
        lab: count * sec_per_processed for lab, count in label_counts.items()
    }
    events_after = len(store.list_recent(patient["id"], limit=500))
    new_events = max(0, events_after - events_before)

    store.upsert_patient_state(
        patient["id"],
        status=last["status"],
        activity=last.get("activity") or (labels[-1] if labels else ""),
        reason=last.get("reason", ""),
        action=last.get("action", ""),
    )
    board_status = reconcile_patient_status(store, patient["id"])
    preview_path = None
    for e in store.list_recent(patient["id"], limit=5):
        if e.get("snapshot_path"):
            preview_path = e.get("snapshot_path")
            break
    session = store.create_session(
        patient_id=patient["id"],
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        source_type="video",
        source=f"room:{patient['room']}",
        final_status=board_status,
        final_activity=last.get("activity") or "",
        reason_text=last.get("reason", ""),
        event_count=new_events,
        activity_seconds=activity_seconds,
        label_counts=label_counts,
        preview_path=preview_path,
        session_status="COMPLETED",
        room_id=patient.get("room") or "",
    )
    playable = False
    if session is not None:
        rec = _attach_source_video_recording(
            store,
            session_id=int(session["id"]),
            patient_id=patient["id"],
            source_path=path,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
        )
        playable = bool(rec and (rec.get("recording_status") or "").upper() == "READY")
    return {
        "patient": patient,
        "status": board_status,
        "last": last,
        "duration_s": duration_s,
        "new_events": new_events,
        "playable": playable,
        "annotated_rgb": annotated_rgb,
        "session": session,
    }


def run_room_check(
    user, patient_id: str, store: EventStore, new_pipeline, lock,
    progress=None, panel: str | None = "Home",
):
    import gradio as gr

    if not user:
        raise gr.Error("Sign in first.")
    patient = get_patient(patient_id)
    if not patient or patient["caregiver_id"] != user["id"]:
        raise gr.Error("Patient not found or not assigned to you.")

    if patient["source_type"] == "live":
        st = store.get_patient_state(patient["id"])
        preview = _latest_snapshot_rgb(store, patient["id"])
        note = (
            f"**{patient['name']}** ({patient['room']}) uses the **laptop live camera**.\n\n"
            "1. On the laptop open **/camera-station**\n"
            "2. Start camera (full body)\n"
            "3. Return here and refresh"
        )
        if st:
            note += (
                f"\n\nLast known: **{st.get('status')}** · `{st.get('activity')}` · {st.get('reason')}"
            )
        cards, inbox, alerts = refresh_dashboard(user, store, panel=panel)
        return note, preview, format_decision({
            "status": (st or {}).get("status") or "SAFE",
            "reason": (st or {}).get("reason") or "Awaiting live camera",
            "action": (st or {}).get("action") or "Start Camera Station on the laptop",
        }, activity=(st or {}).get("activity")), cards, inbox, alerts, patient["id"]

    if patient["source_type"] != "video":
        cards, inbox, alerts = refresh_dashboard(user, store, panel=panel)
        note = (
            f"**{patient['name']}** ({patient['room']}) has no camera/video linked yet.\n\n"
            "Patient info is saved. Attach live camera (Settings → add with live camera) "
            "or use Camera Station when this room is the live patient."
        )
        return (
            note, None,
            format_decision({
                "status": "SAFE",
                "reason": "No monitoring source linked",
                "action": "Add live camera or wait for monitoring",
            }),
            cards, inbox, alerts, patient["id"],
        )

    try:
        prepared = prepare_video_patient_demo(
            store, patient["id"], new_pipeline=new_pipeline, lock=lock, progress=progress,
        )
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    last = prepared["last"]
    board_status = prepared["status"]
    play_line = (
        "Recording **READY** — open **Watch recording** on the patient page."
        if prepared["playable"]
        else "Session summary saved (recording copy unavailable)."
    )
    note = (
        f"**Case ready: {patient['name']}** ({patient['room']})\n\n"
        f"**{board_status}** · {last.get('reason')}\n\n"
        f"Session saved ({prepared['duration_s']:.0f}s monitored from processed frames).\n\n"
        f"{play_line}"
    )
    decision_md = format_decision(
        {**last, "status": board_status},
        activity=last.get("activity"),
    )
    cards, inbox, alerts = refresh_dashboard(user, store, panel=panel)
    return note, prepared["annotated_rgb"], decision_md, cards, inbox, alerts, patient["id"]


def review_alert(user, event_id, store: EventStore, new_pipeline, lock, progress=None, panel="Home"):
    """Open the patient case for an alert and focus that event id / bundle."""
    import gradio as gr

    if not user:
        raise gr.Error("Sign in first.")
    if event_id is None or event_id == "":
        raise gr.Error("Pick an open alert first.")
    primary, _ids = parse_alert_bundle(event_id)
    if primary is None:
        raise gr.Error("Pick a valid patient alert first.")
    eid = int(primary)
    row = store.get(eid)
    if row is None:
        raise gr.Error(f"Event {eid} not found.")
    if row.get("event_status") == "RESOLVED":
        raise gr.Error(f"Event {eid} is already resolved — check History.")
    patient_id = row["patient_id"]
    note, frame, mon, cards, inbox, alerts, pid = run_room_check(
        user, patient_id, store, new_pipeline, lock, progress=progress, panel=panel,
    )
    title = _event_title(row.get("event_type"))
    note = f"**Reviewing event #{eid}** — {title}\n\n" + note
    return note, frame, mon, cards, inbox, alerts, pid, eid


def sync_live_patient_state(store: EventStore, patient_id: str, decision: dict, activity: str):
    store.upsert_patient_state(
        patient_id,
        status=decision.get("status", "SAFE"),
        activity=activity or "",
        reason=decision.get("reason", ""),
        action=decision.get("action", ""),
    )
    reconcile_patient_status(store, patient_id)


def _latest_snapshot_rgb(store: EventStore, patient_id: str):
    from . import config

    for e in store.list_recent(patient_id, limit=8):
        rel = e.get("snapshot_path") or ""
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = config.ROOT / path
        if not path.exists():
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None
