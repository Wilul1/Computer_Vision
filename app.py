"""Activity monitor app (browser UI).

    python app.py            # then open http://127.0.0.1:7860

Three tabs: upload a video, upload a photo, or use the webcam live. It is the
same pipeline as src/run.py and src/evaluate.py (src/pipeline.py), with the
parameters tuned on the val split. YOLO and the pose models are loaded ONCE at
start-up and shared, not reloaded per click or per frame.

Phase 1A: monitoring status + SQLite events + snapshots shown in Gradio.
Phase L2: Care Home board (demo caregiver + 3 patients).
"""
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.alert_actions import decide, format_decision  # noqa: E402
from src.alert_center import (  # noqa: E402
    caregiver_owns_event,
)
from src.care_home import (  # noqa: E402
    build_dashboard_html,
    login as care_login_fn,
    normalize_alert_filters,
    reconcile_patient_status,
    refresh_dashboard,
    run_room_check,
    switch_panel as care_switch_panel_fn,
    sync_live_patient_state,
)
from src.camera_station import CameraStationError, ServerCameraStation  # noqa: E402
from src.live_hub import LIVE_PATIENT_ID, LiveMonitorHub  # noqa: E402
from src.patient_details import (  # noqa: E402
    back_to_patients as care_back_fn,
    empty_playback_panel,
    open_patient_details as care_open_details_fn,
    playback_panel_for_patient,
    select_session_playback,
)
from src.classifier import IGNORED  # noqa: E402
from src.detector import PersonDetector  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.monitoring import EVENT_TABLE_HEADERS, MonitoringSession, events_table  # noqa: E402
from src.params import Params  # noqa: E402
from src.patient_roster import reset_roster_singleton  # noqa: E402
from src.pipeline import MonitorPipeline  # noqa: E402
from src.session_summary import session_blurb, session_details  # noqa: E402
from ui import APP_CSS, CAREGIVER_APP_CSS, build_caregiver_ui, build_ui  # noqa: E402
from ui_camera_station import (  # noqa: E402
    CAMERA_STATION_CSS,
    build_camera_station_ui,
    camera_station_footer_html,
    camera_station_header_html,
)


PARAMS = Params.load()
print("Loading models once...")
DETECTOR = PersonDetector()
DETECTOR(np.zeros((360, 640, 3), np.uint8))   # warm-up, so the first click is not 4 s slower
LOCK = threading.Lock()       # the shared YOLO model is used by one request at a time
EXAMPLES = ROOT / "tests" / "cases"
EVENT_STORE = EventStore()
# Caregiver-managed roster (empty until they add patients). Same DB file as events.
PATIENT_ROSTER = reset_roster_singleton(EVENT_STORE.db_path)
# Live Camera Station patient id (assign via Add patient → live camera).
LIVE_PATIENT = LIVE_PATIENT_ID
LAB_PATIENT = "lab-session"
LIVE_HUB = LiveMonitorHub(EVENT_STORE, patient_id=LIVE_PATIENT)


def new_pipeline(*, live: bool = False) -> MonitorPipeline:
    """Fresh tracking state around the shared YOLO model.

    live=True only for webcam: skips MOG2/camera-flow to raise live FPS.
    Video/photo Analyze keep live=False (same path as evaluate).
    """
    pipe = MonitorPipeline(PARAMS, use_yolo=False, live=live)
    pipe.detector = DETECTOR
    return pipe


# One-time wipe → Alex (live) + Casey with room-check data already prepared.
_WIPE_FLAG = EVENT_STORE.db_path.parent / ".roster_cleared_v3"
if not _WIPE_FLAG.exists():
    from src.care_demo import seed_careview_pack
    from src.care_home import prepare_video_patient_demo

    EVENT_STORE.wipe_all_monitoring_data()
    PATIENT_ROSTER.clear_all()
    try:
        seed_careview_pack(caregiver_id="cg1")
        print("Seeding Casey ready-demo data (analyzing fall clip)...")
        prepared = prepare_video_patient_demo(
            EVENT_STORE, "p-casey", new_pipeline=new_pipeline, lock=LOCK,
        )
        print(
            f"Caregiver board ready — Alex (live) + Casey "
            f"({prepared.get('status')}, {prepared.get('new_events')} alerts)."
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Caregiver board wiped; ready demo skipped: {exc}")
    _WIPE_FLAG.write_text("cleared_v3\n", encoding="utf-8")

# Every launch: close yesterday's leftover OPEN/ACK alerts so they don't beep / sticky-banner.
_stale_n = EVENT_STORE.resolve_stale_active_events()
if _stale_n:
    from src.care_demo import patients_for as _patients_for

    print(f"Closed {_stale_n} stale alert(s) from before today (UTC).")
    for _p in _patients_for("cg1"):
        reconcile_patient_status(EVENT_STORE, _p["id"])


# Server-side webcam owner for /camera-station (avoids browser getUserMedia / LAN HTTP block).
CAMERA_STATION = ServerCameraStation(
    LIVE_HUB,
    new_pipeline=new_pipeline,
    lock=LOCK,
    camera_index=0,
    # Cap ~15; actual rate follows CPU. Live pipeline is slimmed to cut lag.
    target_fps=15.0,
    on_frame=lambda mon, activity: sync_live_patient_state(
        EVENT_STORE, LIVE_PATIENT, mon, activity,
    ),
)


def _events_df(patient_id: str = LIVE_PATIENT) -> pd.DataFrame:
    rows = events_table(EVENT_STORE.list_recent(patient_id, limit=15))
    return pd.DataFrame(rows, columns=EVENT_TABLE_HEADERS)


def timeline(times: list[float], labels: list[str], min_len: float = 0.5) -> pd.DataFrame:
    """Per-frame labels -> segments [start, end, activity]; runs shorter than
    min_len seconds are merged into the previous segment."""
    segs = []
    for t, lab in zip(times, labels):
        if segs and segs[-1][2] == lab:
            segs[-1][1] = t
        else:
            segs.append([t, t, lab])
    merged = []
    for s in segs:
        if merged and (s[1] - s[0] < min_len or s[2] == merged[-1][2]):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return pd.DataFrame([{"start (s)": round(a, 1), "end (s)": round(b, 1), "activity": c} for a, b, c in merged])


def summary_table(labels: list[str], fps_source: float) -> pd.DataFrame:
    counts = Counter(labels)
    total = sum(counts.values())
    return pd.DataFrame([{"activity": k, "seconds": round(v / fps_source, 1), "share": f"{v / total:.0%}"}
                         for k, v in counts.most_common()])


# ---------------------------------------------------------------- video
def analyse_video(path, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a video first.")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out_path = Path(tempfile.mkdtemp()) / "annotated.mp4"
    writer, labels, base_labels, times, ms, events, confs = None, [], [], [], [], 0, []
    fall_events = inactive_events = 0
    inactive_seconds = 0.0
    zone_out = zone_partial = 0
    last_decision = decide(r={"activity": "standing"})
    monitor = MonitoringSession(patient_id=LAB_PATIENT, source="video", store=EVENT_STORE)
    with LOCK:
        pipe = new_pipeline()
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / fps
                t0 = time.perf_counter()
                r = pipe.process(frame, t)
                ms.append((time.perf_counter() - t0) * 1000)
                mon = monitor.update(r, r.get("annotated"))
                last_decision = mon
                if writer is None:
                    h, w = r["annotated"].shape[:2]
                    writer = imageio_ffmpeg.write_frames(str(out_path), (w, h), fps=fps, codec="libx264",
                                                         pix_fmt_out="yuv420p", macro_block_size=1,
                                                         output_params=["-crf", "26"])
                    writer.send(None)
                writer.send(np.ascontiguousarray(r["annotated"][:, :, ::-1]))
                labels.append(r["activity"])
                base_labels.append(r["activity_baseline"])
                confs.append(r["confidence"])
                times.append(t)
                events = r["motion_events"]
                fall_events = r.get("fall_events", 0)
                inactive_events = r.get("inactive_events", 0)
                inactive_seconds = r.get("inactive_seconds", 0.0)
                zone_out = r.get("zone_out_frames", 0)
                zone_partial = r.get("zone_partial_frames", 0)
                i += 1
                if i % 10 == 0:
                    progress(min(i / n, 1.0), desc=f"frame {i}/{n}")
        finally:
            cap.release()
            pipe.close()
            if writer is not None:
                writer.close()
    if not labels:
        raise gr.Error("Could not read any frames from this video.")
    voting = [l for l in labels if l not in IGNORED]
    main = Counter(voting).most_common(1)[0][0] if voting else labels[-1]
    warm = ms[10:] or ms
    duration_s = len(labels) / fps
    blurb = session_blurb(
        labels, duration_s=duration_s, fall_events=fall_events,
        inactive_events=inactive_events, inactive_seconds=inactive_seconds,
        zone_out_frames=zone_out, zone_partial_frames=zone_partial, fps=fps,
    )
    details = session_details(
        labels, duration_s=duration_s, fall_events=fall_events,
        inactive_events=inactive_events, inactive_seconds=inactive_seconds,
        zone_out_frames=zone_out, zone_partial_frames=zone_partial, fps=fps,
        extra={
            "main activity": main,
            "confidence (share of frames)": round(sum(l == main for l in voting) / max(len(voting), 1), 2),
            "baseline (MOG2 blob) says": Counter(base_labels).most_common(1)[0][0],
            "motion events": events,
            "frames": len(labels),
            "video length (s)": round(duration_s, 1),
            "processing (ms per frame, median)": round(float(np.median(warm)), 1),
            "processing FPS": round(1000 / float(np.median(warm)), 1),
            "monitoring status (end of clip)": last_decision.get("status"),
            "monitoring reason": last_decision.get("reason"),
            "next action": last_decision.get("action"),
        },
    )
    notes = []
    if fall_events:
        notes.append(f"FALL ×{fall_events}")
    if inactive_events:
        notes.append(f"INACTIVE ×{inactive_events}")
    if zone_out or zone_partial:
        notes.append("ZONE issues")
    note = ("  ·  " + "  ·  ".join(notes)) if notes else ""
    mon_md = format_decision(last_decision, activity=main)
    headline = (f"### {main.upper()}  ·  {details['confidence (share of frames)']:.0%} of frames  ·  "
                f"{details['processing (ms per frame, median)']} ms/frame ({details['processing FPS']} FPS)"
                f"{note}\n\n{blurb}\n\n{mon_md}")
    return (
        str(out_path),
        headline,
        summary_table(labels, fps),
        timeline(times, labels),
        details,
        mon_md,
        _events_df(),
    )


# ---------------------------------------------------------------- image
def analyse_image(image):
    if image is None:
        raise gr.Error("Upload a photo first.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with LOCK:
        pipe = new_pipeline()
        try:
            t0 = time.perf_counter()
            r = pipe.process(bgr, 0.0)
            ms = (time.perf_counter() - t0) * 1000
        finally:
            pipe.close()
    # Photo: status only (no temporal fall/inactive); still show framing status.
    mon = decide(r)
    fmt = lambda v: None if v is None else round(float(v), 2)  # noqa: E731
    details = {
        "activity": r["activity"],
        "people found (YOLO)": len(r["people"]),
        "person confidence (YOLO)": round(r["people"][0]["conf"], 2) if r["people"] else None,
        "knee angle (deg)": fmt(r.get("knee_angle")), "hip angle (deg)": fmt(r.get("hip_angle")),
        "thigh ratio": fmt(r.get("thigh_ratio")), "torso tilt (deg)": fmt(r.get("torso_angle")),
        "processing (ms)": round(ms, 1),
        "monitoring status": mon["status"],
        "monitoring reason": mon["reason"],
        "next action": mon["action"],
        "note": "A single photo has no motion, so it can only show standing, sitting or lying.",
    }
    headline = (
        f"### {r['activity'].upper()}  ·  {len(r['people'])} person(s)  ·  {ms:.0f} ms\n\n"
        + format_decision(mon, activity=r["activity"])
    )
    return cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), headline, details


# ---------------------------------------------------------------- webcam
def _close_webcam_state(state):
    """Release the per-session pipeline so the next Start is a clean slate."""
    if isinstance(state, dict):
        pipe = state.get("pipe")
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
    return None


def _live_patient_label() -> str:
    """Name · Room for the roster patient assigned to this Camera Station."""
    return PATIENT_ROSTER.live_camera_label()


def _live_patient_name() -> str:
    patient = PATIENT_ROSTER.get_live_patient()
    if patient and (patient.get("name") or "").strip():
        return patient["name"].strip()
    return "the live patient"


def _station_status_md(snap: dict | None = None) -> str:
    snap = snap or LIVE_HUB.snapshot()
    conn = "CONNECTED" if snap.get("connected") else "DISCONNECTED"
    sid = snap.get("session_id") or "—"
    elapsed = float(snap.get("elapsed_s") or 0)
    mins, secs = divmod(int(elapsed), 60)
    act = (snap.get("activity") or "—").replace("_", " ").title()
    conf = snap.get("confidence")
    conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
    label = _live_patient_label()
    return (
        f"**Camera Station · {label}**  \n"
        f"Session `#{sid}` · **{snap.get('status') or 'SAFE'}** · Camera **{conn}**  \n"
        f"Activity: **{act}** ({conf_s}) · Elapsed **{mins}m {secs:02d}s**"
    )


def webcam_step(frame, state):
    """Camera Station: browser streams laptop webcam → one CV pipeline → shared hub.

    Caregiver browsers only poll LIVE_HUB.get_preview() — they never open the camera.
    """
    empty_events = _events_df()
    LIVE_HUB.ensure_session()
    if frame is None:
        mon = LIVE_HUB.monitor().update(None, None, camera_unavailable=True)
        LIVE_HUB.mark_camera_unavailable(mon)
        sync_live_patient_state(EVENT_STORE, LIVE_PATIENT, mon, "")
        return (
            None,
            "### Camera frame missing — check Start camera / browser permission",
            mon["panel_markdown"],
            _station_status_md(),
            empty_events,
            state if isinstance(state, dict) else None,
        )
    if state is None:
        state = {
            "pipe": new_pipeline(live=True),
            "start": time.perf_counter(),
            "labels": [],
            "fall_events": 0,
            "inactive_events": 0,
            "inactive_seconds": 0.0,
            "zone_out": 0,
            "zone_partial": 0,
        }
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    with LOCK:
        t0 = time.perf_counter()
        r = state["pipe"].process(bgr, time.perf_counter() - state["start"])
        ms = (time.perf_counter() - t0) * 1000
    mon = LIVE_HUB.monitor().update(r, r.get("annotated"))
    annotated_rgb = cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB)
    snap = LIVE_HUB.ingest(
        annotated_rgb=annotated_rgb,
        activity=r.get("activity"),
        confidence=r.get("confidence"),
        decision=mon,
        camera_unavailable=False,
        inference_ms=ms,
    )
    sync_live_patient_state(EVENT_STORE, LIVE_PATIENT, mon, r.get("activity", ""))
    state["labels"].append(r["activity"])
    state["fall_events"] = r.get("fall_events", 0)
    state["inactive_events"] = r.get("inactive_events", 0)
    state["inactive_seconds"] = r.get("inactive_seconds", 0.0)
    state["zone_out"] = r.get("zone_out_frames", 0)
    state["zone_partial"] = r.get("zone_partial_frames", 0)
    flags = []
    if r.get("fall_alert"):
        flags.append("**POSSIBLE FALL**")
    elif r.get("inactive_alert"):
        flags.append("**INACTIVE**")
    elif r.get("zone_alert"):
        flags.append(f"**{r.get('zone_label') or 'ZONE'}**")
    flag = ("  ·  " + "  ·  ".join(flags)) if flags else ""
    text = (f"### {r['activity'].upper()}  ·  {r['confidence']:.0%}  ·  "
            f"{ms:.0f} ms/frame ({1000 / max(ms, 1):.0f} FPS)"
            f"{flag}")
    live_fps = 12.0
    dur = len(state["labels"]) / live_fps
    summary = (
        _station_status_md(snap)
        + "\n\n### Session so far\n"
        + session_blurb(
            state["labels"], duration_s=dur,
            fall_events=state["fall_events"],
            inactive_events=state["inactive_events"],
            inactive_seconds=state["inactive_seconds"],
            zone_out_frames=state["zone_out"],
            zone_partial_frames=state["zone_partial"],
            fps=live_fps,
        )
    )
    return (
        annotated_rgb,
        text,
        mon["panel_markdown"],
        summary,
        _events_df(),
        state,
    )


def stop_webcam(state):
    """Stop Camera Station stream and finalize the ACTIVE live session."""
    finished = LIVE_HUB.stop(interrupted=False)
    summary = "### Camera off — no session data yet."
    panel = format_decision(decide(camera_unavailable=True))
    if finished is not None:
        dur = float(finished.get("duration_s") or 0)
        acts = finished.get("activity_seconds") or {}
        act_bits = ", ".join(
            f"{k}: {v:.0f}s" for k, v in sorted(acts.items(), key=lambda kv: -kv[1])
        ) or "no activity segments"
        who = _live_patient_name()
        panel = format_decision({
            "status": finished.get("final_status") or "SAFE",
            "reason": finished.get("reason_text") or "Session completed",
            "action": f"Review session on {who}'s patient page",
        }, activity=finished.get("final_activity"))
        summary = (
            f"### Session #{finished.get('id')} · **{finished.get('session_status')}**\n"
            f"Duration **{dur:.0f}s** · Events **{finished.get('event_count', 0)}**\n\n"
            f"Activity: {act_bits}\n\n"
            f"Caregiver phone can open {who}'s details to review this session."
        )
    elif isinstance(state, dict) and state.get("labels"):
        live_fps = 12.0
        dur = len(state["labels"]) / live_fps
        blurb = session_blurb(
            state["labels"], duration_s=dur,
            fall_events=state.get("fall_events", 0),
            inactive_events=state.get("inactive_events", 0),
            inactive_seconds=state.get("inactive_seconds", 0.0),
            zone_out_frames=state.get("zone_out", 0),
            zone_partial_frames=state.get("zone_partial", 0),
            fps=live_fps,
        )
        summary = f"### Session summary\n{blurb}"
    _close_webcam_state(state)
    # Clear patient live connection signal without forcing SAFE if open events remain.
    st = EVENT_STORE.get_patient_state(LIVE_PATIENT) or {}
    EVENT_STORE.upsert_patient_state(
        LIVE_PATIENT,
        status=st.get("status") or "SAFE",
        activity=st.get("activity") or "",
        reason="Live camera stopped",
        action=st.get("action") or f"Open {_live_patient_name()}'s details to review the session",
    )
    reconcile_patient_status(EVENT_STORE, LIVE_PATIENT)
    return (
        gr.skip(),
        None,
        "### Camera Station stopped — session saved.",
        panel,
        summary,
        _events_df(),
        None,
    )


def _open_alerts_for_live() -> list[dict]:
    """OPEN alerts for the live Camera Station patient (created during streaming)."""
    from src.alert_center import list_caregiver_active_alerts

    rows = list_caregiver_active_alerts(
        EVENT_STORE, "cg1", limit=8, status="OPEN", patient_id=LIVE_PATIENT,
    )
    return rows


def _live_banner_md(snap: dict | None = None) -> str:
    """Caregiver-facing LIVE / STALE / OFFLINE banner for the live patient's details."""
    snap = snap or LIVE_HUB.snapshot()
    age = snap.get("frame_age_s")
    age_s = f"{age:.1f}s ago" if isinstance(age, (int, float)) else "no frame yet"
    act = (snap.get("activity") or "—").replace("_", " ").title()
    conf = snap.get("confidence")
    conf_s = f"{conf:.0%}" if isinstance(conf, (int, float)) else "—"
    sid = snap.get("session_id") or "—"
    status = snap.get("status") or "—"
    live_state = snap.get("live_state") or "OFFLINE"
    # Never label STALE/OFFLINE frames as LIVE.
    seq = snap.get("frame_seq") or 0
    open_alerts = _open_alerts_for_live()
    fall_open = [a for a in open_alerts if a.get("event_type") == "possible_fall"]
    alert_block = ""
    if fall_open:
        top = fall_open[0]
        from src.alert_center import event_display_title

        title = event_display_title(top.get("event_type"), short=False)
        reason = (top.get("reason_text") or top.get("action_text") or "").strip()
        n = len(fall_open)
        # Frame may look SAFE while an OPEN fall alert still needs a check.
        alert_block = (
            f"\n\n### ⚠ POSSIBLE FALL ({n}) — **{title}**\n"
            f"{reason or 'Check this patient now.'}\n\n"
            "Raised **during live monitoring**. Open **Alerts** or use **Care response** below."
        )
        status = "ALERT"
    if (
        live_state == "LIVE"
        and CAMERA_STATION.running
        and snap.get("connected")
        and snap.get("has_fresh_frame")
    ):
        return (
            f"### Live Camera · **LIVE**\n"
            f"Session `#{sid}` · **{status}** · Activity **{act}** ({conf_s}) · "
            f"Updated {age_s} · seq `{seq}`\n\n"
            "Annotated frames come from the laptop Camera Station (view-only on this phone)."
            f"{alert_block}"
        )
    if CAMERA_STATION.running or snap.get("session_id") or snap.get("has_frame"):
        return (
            f"### Live Camera · **{live_state if live_state != 'LIVE' else 'STALE'}**\n"
            f"Session `#{sid}` · last activity **{act}** · {age_s}\n\n"
            "Monitoring may still be ACTIVE on the laptop, but this phone has no fresh frame. "
            "Confirm **Start Monitoring** is running on `/camera-station`."
            f"{alert_block}"
        )
    return (
        "### Live Camera · **OFFLINE**\n"
        "No ACTIVE Camera Station session. On the laptop open `/camera-station` and press "
        "**Start Monitoring**. This phone never opens the webcam."
        f"{alert_block}"
    )


def care_alert_watch(user, last_seen_id=0):
    """Poll OPEN alerts for a sticky interrupt banner (works while watching live).

    last_seen_id must come from a Textbox (not State) — Gradio Timer drops State.
    See `poll_urgent_watch` for last_seen encoding (+seed / -sticky).
    """
    from src.alert_center import poll_urgent_watch

    user = _care_user(user)
    if not user:
        return "", "0"
    interrupt_id = 0
    try:
        interrupt_id = int(LIVE_HUB.snapshot().get("interrupt_event_id") or 0)
    except Exception:
        interrupt_id = 0
    html, seen = poll_urgent_watch(
        EVENT_STORE,
        user["id"],
        last_seen_id,
        interrupt_event_id=interrupt_id,
    )
    if interrupt_id > 0 and html:
        try:
            LIVE_HUB.clear_interrupt_event(interrupt_id)
        except Exception:
            pass
    return html, seen


def care_live_tick(patient_id, last_seq=0):
    """Single caregiver poll: embedded JPEG HTML + banner + status.

    Embeds JPEG as a data-URI when seq advances so Messenger/WebView cannot keep
    a frozen cached `/api/live/*.jpg`. Skips the HTML payload when seq is unchanged.
    Uses a Textbox patient id so Gradio Timer reliably sees the live patient id.
    """
    pid = (patient_id or "").strip() if isinstance(patient_id, str) else ""
    try:
        prev_seq = int(last_seq or 0)
    except (TypeError, ValueError):
        prev_seq = 0
    if pid != LIVE_PATIENT:
        # Not the live patient — do not touch the live panel outputs / keep seq.
        return gr.skip(), gr.skip(), gr.skip(), prev_seq
    snap = LIVE_HUB.snapshot()
    seq = int(snap.get("frame_seq") or 0)
    banner = _live_banner_md(snap)
    # Caregiver status strip stays light (no perf clutter on the phone).
    mon = _station_status_md(snap)
    if not snap.get("has_fresh_frame"):
        empty = LIVE_HUB.preview_img_html(
            pid,
            height=72,
            empty_message="No live frame — start Camera Station on the laptop (details stay above).",
        )
        return empty, banner, mon, 0
    if seq == prev_seq and prev_seq > 0:
        # Same frame — skip image HTML; banner still refreshes age/status.
        return gr.skip(), banner, mon, prev_seq
    # embed_jpeg=True defeats aggressive in-app browser image caches.
    html = LIVE_HUB.preview_img_html(pid, seq=seq, embed_jpeg=True)
    return html, banner, mon, seq


def care_live_preview(patient_id):
    """Lab / numpy preview path (resized RGB from hub cache)."""
    pid = (patient_id or "").strip() if isinstance(patient_id, str) else patient_id
    if pid != LIVE_PATIENT:
        return gr.skip()
    return LIVE_HUB.get_preview(pid)


def care_live_status(patient_id):
    pid = (patient_id or "").strip() if isinstance(patient_id, str) else patient_id
    if pid != LIVE_PATIENT:
        return gr.skip()
    return _live_banner_md()


def camera_station_status():
    """Camera Station status strip (no caregiver patient filter)."""
    return _station_status_md()


def _station_ui_bundle(*, skip_unchanged_images: bool = False):
    snap = CAMERA_STATION.snapshot_ui()
    hub = snap["hub"]
    err = snap.get("error") or ""
    err_md = f"**Camera error:** {err}" if err else ""
    diag = snap.get("diagnostics") or LIVE_HUB.diagnostics_md()
    label = _live_patient_label()
    if CAMERA_STATION.running:
        status = _station_status_md(hub) + "\n\n" + diag
    else:
        status = (
            _station_status_md(hub)
            + "\n\nCamera Station is **stopped**. Press **Start Monitoring** to open the laptop webcam."
            + "\n\n" + diag
        )
    raw = snap.get("raw")
    ann = snap.get("annotated")
    if skip_unchanged_images and not snap.get("ui_changed", True):
        # Avoid re-encoding the same numpy frames through Gradio every timer tick.
        raw = gr.skip()
        ann = gr.skip()
    return (
        camera_station_header_html(label),
        raw,
        ann,
        snap.get("headline") or "### Camera Station",
        snap.get("panel") or status,
        status,
        err_md,
        gr.update(value=_events_df(), label=f"Session events ({label})"),
        camera_station_footer_html(label),
    )


def station_start(camera_index):
    """Start server-side OpenCV capture for the live Camera Station patient."""
    try:
        idx = int(camera_index) if camera_index is not None else 0
    except (TypeError, ValueError):
        idx = 0
    try:
        msg = CAMERA_STATION.start(idx)
    except CameraStationError as exc:
        title, raw, ann, head, panel, status, _err, events, footer = _station_ui_bundle()
        return (
            title, raw, ann,
            f"### Could not start camera\n{exc}",
            panel, status, f"**Camera error:** {exc}", events, footer,
        )
    # Sync board patient state while station runs (worker updates hub; refresh once).
    st = LIVE_HUB.snapshot()
    sync_live_patient_state(
        EVENT_STORE, LIVE_PATIENT,
        {"status": st.get("status") or "SAFE", "reason": st.get("reason") or msg, "action": "Monitoring"},
        st.get("activity") or "",
    )
    title, raw, ann, head, panel, status, err, events, footer = _station_ui_bundle()
    return title, raw, ann, f"### {msg}\n\n{head}", panel, status, err, events, footer


def station_stop():
    """Stop server-side capture and finalize the ACTIVE live session."""
    finished = CAMERA_STATION.stop(interrupted=False)
    who = _live_patient_name()
    st = EVENT_STORE.get_patient_state(LIVE_PATIENT) or {}
    EVENT_STORE.upsert_patient_state(
        LIVE_PATIENT,
        status=st.get("status") or "SAFE",
        activity=st.get("activity") or "",
        reason="Live camera stopped",
        action=st.get("action") or f"Open {who}'s details to review the session",
    )
    reconcile_patient_status(EVENT_STORE, LIVE_PATIENT)
    title, raw, ann, head, panel, status, err, events, footer = _station_ui_bundle()
    if finished is not None:
        # Re-read session so recording_* columns from finalize are visible.
        row = EVENT_STORE.get_session(int(finished["id"])) or finished
        dur = float(row.get("duration_s") or 0)
        acts = row.get("activity_seconds") or {}
        act_bits = ", ".join(
            f"{k}: {v:.0f}s" for k, v in sorted(acts.items(), key=lambda kv: -kv[1])
        ) or "no activity segments"
        rec_st = row.get("recording_status") or "NONE"
        rec_path = row.get("recording_path") or "—"
        rec_err = row.get("recording_error") or ""
        rec_line = f"Recording **{rec_st}** · `{rec_path}`"
        if rec_err:
            rec_line += f" · {rec_err}"
        panel = format_decision({
            "status": row.get("final_status") or "SAFE",
            "reason": row.get("reason_text") or "Session completed",
            "action": f"Review session on {who}'s patient page",
        }, activity=row.get("final_activity"))
        head = (
            f"### Session #{row.get('id')} · **{row.get('session_status')}**\n"
            f"Duration **{dur:.0f}s** · Events **{row.get('event_count', 0)}**\n\n"
            f"Activity: {act_bits}\n\n{rec_line}"
        )
        status = head
    else:
        head = "### Camera Station stopped"
        panel = format_decision(decide(camera_unavailable=True))
    return title, raw, ann, head, panel, status, err, events, footer


def station_tick():
    """Timer poll for Camera Station preview images + status."""
    return _station_ui_bundle(skip_unchanged_images=True)


def acknowledge_event(event_id, *, handled_by=None, note=None):
    """Lab path: ACK by event id (no caregiver roster check).

    Caregiver UI must use care_ack(), which enforces patient assignment.
    """
    if event_id is None or event_id == "":
        return _events_df(), "Enter an event id to acknowledge."
    try:
        eid = int(event_id)
    except (TypeError, ValueError):
        return _events_df(), "Event id must be a number."
    row = EVENT_STORE.acknowledge(eid, handled_by=handled_by, note=note)
    if row is None:
        return _events_df(), f"Event {eid} not found."
    return _events_df(), (
        f"Event {eid} → **{row['event_status']}**"
        + (f" · noted by **{row.get('handled_by')}**" if row.get("handled_by") else "")
        + ". Alert stays active until Resolve."
    )


def resolve_event(event_id, *, handled_by=None, outcome=None, note=None):
    """Lab path: resolve by event id (no caregiver roster check)."""
    from src.alert_center import outcome_label

    if event_id is None or event_id == "":
        return _events_df(), "Enter an event id to resolve."
    try:
        eid = int(event_id)
    except (TypeError, ValueError):
        return _events_df(), "Event id must be a number."
    row = EVENT_STORE.resolve(
        eid, handled_by=handled_by, outcome=outcome, note=note,
    )
    if row is None:
        return _events_df(), f"Event {eid} not found."
    bits = [f"Event {eid} → **RESOLVED**"]
    if row.get("handle_outcome"):
        bits.append(f"outcome **{outcome_label(row.get('handle_outcome'))}**")
    if row.get("handled_by"):
        bits.append(f"by **{row.get('handled_by')}**")
    return _events_df(), " · ".join(bits) + "."


def care_login(username, password):
    return care_login_fn(username, password, EVENT_STORE)


def _filters_from_widgets(priority=None, patient=None, status=None, event_type=None,
                          date_from=None, date_to=None):
    return normalize_alert_filters(
        priority=priority, patient_id=patient, status=status, event_type=event_type,
        date_from=date_from, date_to=date_to,
    )


def care_refresh(user, panel="Home", patient_id=None,
                 priority=None, patient_filter=None, status=None, event_type=None,
                 date_from=None, date_to=None):
    user = _care_user(user)
    pid = (patient_id or "").strip() if isinstance(patient_id, str) else patient_id
    if pid:
        html, _n, _p, _m, _pid, alerts, _eid, _vis, _live, *_ = care_open_details_fn(
            user, pid, EVENT_STORE,
        )
        from src.care_home import _inbox_df
        return html, _inbox_df(EVENT_STORE, user["id"]), alerts
    filters = _filters_from_widgets(
        priority, patient_filter, status, event_type, date_from, date_to,
    )
    return refresh_dashboard(user, EVENT_STORE, panel=panel, alert_filters=filters)


def care_filter_alerts(user, panel, priority, patient_filter, status, event_type,
                       date_from, date_to):
    """Rebuild board (Alert Center when on Alerts) with filter widgets."""
    user = _care_user(user)
    if not user:
        return (
            '<div id="careview-root"><p style="color:#64748B;padding:12px">Sign in first.</p></div>'
        )
    filters = _filters_from_widgets(
        priority, patient_filter, status, event_type, date_from, date_to,
    )
    return build_dashboard_html(
        EVENT_STORE, user["id"], user.get("name", "caregiver1"),
        panel=panel or "Home", alert_filters=filters,
    )


def care_switch_panel(user, panel, priority=None, patient_filter=None, status=None,
                      event_type=None, date_from=None, date_to=None):
    """Nav Home/Alerts/History/Settings — leave patient details and hide live panel."""
    import gradio as gr

    user = _care_user(user)
    filters = _filters_from_widgets(
        priority, patient_filter, status, event_type, date_from, date_to,
    )
    html = care_switch_panel_fn(user, panel, EVENT_STORE, alert_filters=filters)
    label = (panel or "Home").strip() or "Home"
    note = f"Viewing **{label}**. Open a patient card for details and recordings."
    # Lab care page: no separate live_panel widget — clear details chrome + playback.
    return (
        html,
        note,
        None,  # cframe (Image)
        "Status / next action",
        None,  # clear patient state
        gr.update(visible=False),  # details toolbar (+ playback block)
        *empty_playback_panel(),
    )


def care_switch_panel_phone(user, panel, priority=None, patient_filter=None, status=None,
                            event_type=None, date_from=None, date_to=None):
    """Phone nav: also hide Live Camera panel so History/Alerts are visible."""
    import gradio as gr

    user = _care_user(user)
    filters = _filters_from_widgets(
        priority, patient_filter, status, event_type, date_from, date_to,
    )
    html = care_switch_panel_fn(user, panel, EVENT_STORE, alert_filters=filters)
    label = (panel or "Home").strip() or "Home"
    if label == "Alerts":
        note = "Alert Center — pick an alert, then use **Handle alert** below."
    elif label == "Settings":
        note = "Add a patient or use **Reset demo** for Alex (live) + Casey (ready)."
    elif label == "History":
        note = "Event history for your patients."
    else:
        note = f"**{label}** — tap a patient card when you need details."
    empty_live = (
        '<div class="cv-live-empty cv-live-empty-compact">'
        "Live feed appears here when Camera Station is running.</div>"
    )
    handle_vis, play_vis, filters_vis, settings_vis, events_vis = _tool_group_vis(
        details_open=False, panel=label, alert_context=False,
    )
    return (
        html,
        note,
        empty_live,
        f"### {label}",
        "",  # clear patient id so live timer stops
        gr.update(visible=False),  # details toolbar
        gr.update(visible=False),  # live panel
        0,  # live seq
        handle_vis,
        play_vis,
        filters_vis,
        settings_vis,
        events_vis,
        *empty_playback_panel(),
    )


def care_room_check(user, patient_id, panel="Home", progress=gr.Progress()):
    return run_room_check(
        user, patient_id, EVENT_STORE, new_pipeline, LOCK,
        progress=progress, panel=panel,
    )


def _care_user(user):
    """Card clicks can race auto-login State — fall back to demo caregiver."""
    if user:
        return user
    from src.care_demo import get_caregiver
    return get_caregiver("caregiver", "care123")


def _preview_html_for_caregiver(preview, patient_id: str) -> str:
    """Caregiver live panel uses HTML <img>, not Gradio Image/numpy."""
    pid = (patient_id or "").strip() if isinstance(patient_id, str) else ""
    if pid == LIVE_PATIENT:
        return LIVE_HUB.preview_img_html(pid)
    if isinstance(preview, np.ndarray) and preview.size:
        from src.live_hub import _encode_jpeg, _resize_preview

        small = _resize_preview(preview)
        jpeg, _ms = _encode_jpeg(small)
        if jpeg:
            import base64

            b64 = base64.b64encode(jpeg).decode("ascii")
            return (
                '<div class="cv-live-wrap" style="background:#0F172A;border-radius:12px;'
                'overflow:hidden;text-align:center">'
                f'<img src="data:image/jpeg;base64,{b64}" alt="Preview" '
                'style="width:100%;max-height:320px;object-fit:contain;display:block;'
                'margin:0 auto"/></div>'
            )
    return (
        '<div class="cv-live-empty" style="padding:24px;text-align:center;color:#64748B">'
        "No preview</div>"
    )


def _playback_for(user, patient_id: str, focus_event_id=None):
    """Issue recording only — auto-loads the alert's linked session video."""
    user = _care_user(user)
    pid = (patient_id or "").strip()
    if not pid:
        return empty_playback_panel()
    return playback_panel_for_patient(
        EVENT_STORE, user, pid, focus_event_id=focus_event_id,
    )


def _tool_group_vis(
    *,
    details_open: bool,
    panel: str = "Home",
    alert_context: bool = False,
):
    """Visibility for caregiver tool cards under the board.

    - Care Response + Watch recording: only with alert context (banner / Alert Center)
      or on the Alerts board (no patient case open)
    - Alert filters: Alerts board only
    - Add patient: Settings only (never on a patient case)
    - Events log: History only
    """
    label = (panel or "Home").strip() or "Home"
    on_alerts = label == "Alerts"
    on_settings = label == "Settings"
    on_history = label == "History"
    show_response = (bool(details_open) and bool(alert_context)) or (
        on_alerts and not details_open
    )
    show_play = bool(details_open) and bool(alert_context)
    return (
        gr.update(visible=show_response),  # chandle_group (Care Response)
        gr.update(visible=show_play),  # cplay_group (Watch recording)
        gr.update(visible=on_alerts and not details_open),  # calert_filters
        gr.update(visible=on_settings and not details_open),  # csettings_tools
        gr.update(visible=on_history and not details_open),  # cevents_log
    )


def _as_caregiver_details(out):
    """Convert open_patient_details → caregiver live HTML + seq.

    open_patient_details returns 10 items; last is alert_context (bool).
    Caregiver UI uses the first 9 + seq; alert_context is handled by callers.
    """
    items = list(out)
    alert_context = bool(items[9]) if len(items) > 9 else False
    core = items[:9]
    pid = core[4] if len(core) > 4 else ""
    core[2] = _preview_html_for_caregiver(core[2], pid if isinstance(pid, str) else "")
    core.append(0)  # clive_seq
    return tuple(core), alert_context


def _lab_open_with_playback(user, patient_id: str, focus_event_id=None):
    """Lab care page: open details + issue recording (when opened from an alert)."""
    user = _care_user(user)
    base = care_open_details_fn(
        user, patient_id, EVENT_STORE, focus_event_id=focus_event_id,
    )
    # base: html…details_vis, live_vis, alert_context — lab uses first 8 + playback
    return tuple(base[:8]) + tuple(
        _playback_for(user, patient_id, focus_event_id=focus_event_id)
    )


def care_open_ana(user, panel="Home"):
    return _lab_open_with_playback(user, "p1-ana")


def care_open_pedro(user, panel="Home"):
    return _lab_open_with_playback(user, "p2-ben")


def care_open_maria(user, panel="Home"):
    return _lab_open_with_playback(user, "p3-cara")


def care_review_alert(user, event_id, panel="Home"):
    user = _care_user(user)
    if not user:
        raise gr.Error("Sign in first.")
    if event_id in (None, ""):
        raise gr.Error("Pick an open alert first.")
    try:
        eid = int(event_id)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Event id must be a number.") from exc
    row = caregiver_owns_event(EVENT_STORE, eid, user["id"])
    if row is None:
        raise gr.Error("Event not found or not assigned to you.")
    return _lab_open_with_playback(user, row["patient_id"], focus_event_id=eid)


def _phone_open_with_playback(user, patient_id: str, focus_event_id=None):
    user = _care_user(user)
    base = care_open_details_fn(
        user, patient_id, EVENT_STORE, focus_event_id=focus_event_id,
    )
    phone, alert_context = _as_caregiver_details(base)
    handle_vis, play_vis, filters_vis, settings_vis, events_vis = _tool_group_vis(
        details_open=True, panel="Home", alert_context=alert_context,
    )
    return phone + (
        handle_vis, play_vis, filters_vis, settings_vis, events_vis,
    ) + tuple(_playback_for(user, patient_id, focus_event_id=focus_event_id))


def care_open_patient_phone(user, patient_id, panel="Home"):
    """Open any assigned patient by id (from board click / Open patient)."""
    user = _care_user(user)
    pid = (patient_id or "").strip()
    if not pid:
        raise gr.Error("Pick a patient first.")
    from src.care_demo import get_patient

    patient = get_patient(pid)
    if not user or not patient or patient.get("caregiver_id") != user["id"]:
        raise gr.Error("Patient not found or not assigned to you.")
    return _phone_open_with_playback(user, pid)


# Keep legacy names as thin aliases (lab / old hooks).
def care_open_ana_phone(user, panel="Home"):
    return care_open_patient_phone(user, LIVE_PATIENT, panel=panel)


def care_open_pedro_phone(user, panel="Home"):
    return care_open_patient_phone(user, "p2-ben", panel=panel)


def care_open_maria_phone(user, panel="Home"):
    return care_open_patient_phone(user, "p3-cara", panel=panel)


def care_add_patient(user, name, room, use_live_camera=False, panel="Settings"):
    """Create a patient for the signed-in caregiver; refresh board."""
    import gradio as gr
    from src.care_demo import add_patient, patients_for
    from src.care_home import gr_open_event_choices

    user = _care_user(user)
    if not user:
        return (
            '<div id="careview-root"><p style="padding:12px;color:#64748B">Sign in first.</p></div>',
            "Sign in first.",
            gr.update(),
            gr.update(choices=[("All patients", "All")], value="All"),
        )
    # Radio returns "yes"/"no"; Checkbox returned bool — accept both.
    if isinstance(use_live_camera, str):
        live = use_live_camera.strip().lower() in ("yes", "true", "1", "live")
    else:
        live = bool(use_live_camera)
    try:
        row = add_patient(
            name=name or "",
            room=room or "",
            caregiver_id=user["id"],
            source_type="live" if live else "none",
        )
    except ValueError as exc:
        html = build_dashboard_html(
            EVENT_STORE, user["id"], user.get("name", "caregiver1"), panel=panel or "Settings",
        )
        return (
            html, f"**Could not add patient:** {exc}",
            gr.update(), gr.update(), gr.update(),
        )
    html = build_dashboard_html(
        EVENT_STORE, user["id"], user.get("name", "caregiver1"), panel="Home",
    )
    note = (
        f"Added **{row['name']}** · {row['room']}. "
        + ("Linked to laptop Camera Station." if row.get("source_type") == "live"
           else "Open the card anytime from Home.")
    )
    alert_choices = gr_open_event_choices(EVENT_STORE, user["id"])
    alert_val = alert_choices[0][1] if alert_choices else None
    patient_choices = [("All patients", "All")] + [
        (f"{p['name']} — {p['room']}", p["id"]) for p in patients_for(user["id"])
    ]
    return (
        html,
        note,
        gr.update(choices=alert_choices, value=alert_val),
        gr.update(choices=patient_choices, value="All"),
        gr.update(value="Home"),
    )


def care_load_demo_pack(user, panel="Settings"):
    """Wipe board data and load Alex (live) + Casey (ready demo)."""
    import gradio as gr
    from src.care_demo import patients_for, seed_careview_pack
    from src.care_home import gr_open_event_choices

    user = _care_user(user)
    if not user:
        return (
            '<div id="careview-root"><p style="padding:12px;color:#64748B">Sign in first.</p></div>',
            "Sign in first.",
            gr.update(),
            gr.update(choices=[("All patients", "All")], value="All"),
            gr.update(),
        )
    try:
        from src.care_home import prepare_video_patient_demo

        EVENT_STORE.wipe_all_monitoring_data()
        rows = seed_careview_pack(caregiver_id=user["id"])
        prepared = prepare_video_patient_demo(
            EVENT_STORE, "p-casey", new_pipeline=new_pipeline, lock=LOCK,
        )
    except FileNotFoundError as exc:
        html = build_dashboard_html(
            EVENT_STORE, user["id"], user.get("name", "caregiver1"),
            panel=panel or "Settings",
        )
        return html, f"**Demo pack missing files:** {exc}", gr.update(), gr.update(), gr.update()
    except (ValueError, RuntimeError) as exc:
        html = build_dashboard_html(
            EVENT_STORE, user["id"], user.get("name", "caregiver1"),
            panel=panel or "Settings",
        )
        return html, f"**Could not load demo pack:** {exc}", gr.update(), gr.update(), gr.update()

    html = build_dashboard_html(
        EVENT_STORE, user["id"], user.get("name", "caregiver1"), panel="Home",
    )
    names = ", ".join(f"**{r['name']}** ({r['room']})" for r in rows)
    note = (
        f"Board wiped. Loaded demo: {names}.\n\n"
        f"**Casey is ready** — {prepared.get('status')} · "
        f"{prepared.get('new_events')} alert(s) · playback "
        f"{'READY' if prepared.get('playable') else 'pending'}.\n\n"
        "1. Open **Casey** / **Alerts** to review the fall case\n"
        "2. **Alex** — start `/camera-station` for live laptop monitoring"
    )
    alert_choices = gr_open_event_choices(EVENT_STORE, user["id"])
    alert_val = alert_choices[0][1] if alert_choices else None
    patient_choices = [("All patients", "All")] + [
        (f"{p['name']} · {p['room']}", p["id"]) for p in patients_for(user["id"])
    ]
    return (
        html,
        note,
        gr.update(choices=alert_choices, value=alert_val),
        gr.update(choices=patient_choices, value="All"),
        gr.update(value="Home"),
    )


def care_review_alert_phone(user, event_id, panel="Home"):
    user = _care_user(user)
    if not user:
        raise gr.Error("Sign in first.")
    if event_id in (None, ""):
        raise gr.Error("Pick an open alert first.")
    try:
        eid = int(event_id)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Event id must be a number.") from exc
    row = caregiver_owns_event(EVENT_STORE, eid, user["id"])
    if row is None:
        raise gr.Error("Event not found or not assigned to you.")
    return _phone_open_with_playback(user, row["patient_id"], focus_event_id=eid)


def care_back_to_patients(user, panel="Home"):
    """Lab back: 9 board outputs + cleared playback (no live_panel widget)."""
    base = care_back_fn(_care_user(user), EVENT_STORE, panel=panel)
    # base: cards, note, preview, mon, pid, alerts, eid, details_vis, inbox, live_vis
    return tuple(base[:9]) + tuple(empty_playback_panel())


def care_back_to_patients_phone(user, panel="Home"):
    """Phone back: board + empty live HTML + seq + tool vis + cleared playback."""
    base = list(care_back_fn(_care_user(user), EVENT_STORE, panel=panel))
    base[2] = (
        '<div class="cv-live-empty cv-live-empty-compact">'
        "Live feed appears here when Camera Station is running.</div>"
    )
    handle_vis, play_vis, filters_vis, settings_vis, events_vis = _tool_group_vis(
        details_open=False, panel=panel, alert_context=False,
    )
    # cards…live_vis (10) + seq + handle/play/filters/settings/events + playback (4)
    return tuple(base) + (
        0, handle_vis, play_vis, filters_vis, settings_vis, events_vis,
    ) + tuple(empty_playback_panel())


def care_run_details(user, patient_id, panel="Home", progress=gr.Progress()):
    """Run room check for the open patient, then reload their details page."""
    user = _care_user(user)
    if not patient_id:
        raise gr.Error("Open a patient first.")
    note, frame, mon, _cards, inbox, _alerts, pid = care_room_check(
        user, patient_id, panel=panel, progress=progress,
    )
    html, _note, prev, banner, pid2, alerts, eid, vis, _live_vis, *_ = care_open_details_fn(
        user, pid, EVENT_STORE,
    )
    base = (
        html, note, frame if frame is not None else prev, banner, pid2, alerts, eid, vis,
        inbox,
    )
    return base + tuple(_playback_for(user, pid2))


def care_run_details_phone(user, patient_id, panel="Home", progress=gr.Progress()):
    user = _care_user(user)
    if not patient_id:
        raise gr.Error("Open a patient first.")
    note, frame, mon, _cards, inbox, _alerts, pid = care_room_check(
        user, patient_id, panel=panel, progress=progress,
    )
    # If room check opened findings, jump into alert context on the first one.
    active = EVENT_STORE.list_active(pid)
    focus = int(active[0]["id"]) if active else None
    opened = care_open_details_fn(user, pid, EVENT_STORE, focus_event_id=focus)
    phone, alert_context = _as_caregiver_details(opened)
    phone = list(phone)
    # Prefer room-check preview frame when present.
    if frame is not None:
        phone[2] = _preview_html_for_caregiver(frame, pid)
    phone[1] = note
    live_vis = opened[8]
    seq = phone[9] if len(phone) > 9 else 0
    handle_vis, play_vis, filters_vis, settings_vis, events_vis = _tool_group_vis(
        details_open=True, panel=panel, alert_context=alert_context,
    )
    # html…eid, details, inbox, live, seq, handle, play, filters, settings, events, playback
    core = list(phone[:8]) + [
        inbox, live_vis, seq, handle_vis, play_vis, filters_vis, settings_vis, events_vis,
    ]
    return tuple(core) + tuple(_playback_for(user, pid, focus_event_id=focus))


def care_select_session(user, patient_id, session_id):
    """Load / clear the embedded P2 player for the currently open patient only."""
    user = _care_user(user)
    pid = (patient_id or "").strip() if isinstance(patient_id, str) else ""
    # Stale Gradio events: no open patient → always clear (never keep prior video).
    if not pid:
        _dd, player, info, sid = empty_playback_panel()
        return player, info, sid
    return select_session_playback(user, pid, session_id, EVENT_STORE)


def care_response_preview(user, event_id):
    """Refresh Care Response incident card when the selected patient alert changes."""
    from src.alert_center import (
        build_care_response_summary_html,
        caregiver_owns_event,
        find_patient_alert_group,
        parse_alert_bundle,
    )

    user = _care_user(user)
    if not user or event_id in (None, ""):
        return build_care_response_summary_html(None), None
    primary, reviewed = parse_alert_bundle(event_id)
    if primary is None:
        return build_care_response_summary_html(None), None
    owned = caregiver_owns_event(EVENT_STORE, primary, user["id"])
    if owned is None:
        return build_care_response_summary_html(None), None
    group = find_patient_alert_group(
        EVENT_STORE,
        user["id"],
        primary_event_id=primary,
        patient_id=str(owned.get("patient_id") or ""),
        reviewed_ids=reviewed or [primary],
    )
    # Prefer the reviewed ID snapshot so confirmation matches what will be closed.
    if group is not None and reviewed:
        group = dict(group)
        group["child_event_ids"] = list(reviewed)
        group["finding_count"] = len(reviewed)
        group["bundle_token"] = (
            f"{primary}|{','.join(str(i) for i in reviewed)}"
        )
    return build_care_response_summary_html(owned, group=group), primary


def _care_bundle_auth(user, bundle_value):
    """Validate caregiver ownership for a reviewed bundle. Returns (primary, ids, patient_id, err)."""
    from src.alert_center import (
        caregiver_owns_event,
        caregiver_owns_patient,
        find_patient_alert_group,
        parse_alert_bundle,
    )

    primary, reviewed = parse_alert_bundle(bundle_value)
    if primary is None or not reviewed:
        return None, [], None, "Select a patient alert first."
    owned = caregiver_owns_event(EVENT_STORE, primary, user["id"])
    if owned is None:
        return None, [], None, "Not allowed: event not found or not assigned to you."
    pid = str(owned.get("patient_id") or "")
    if not caregiver_owns_patient(user["id"], pid):
        return None, [], None, "Not allowed: patient not assigned to you."

    # Plain event id (no ``|`` token): expand to the current patient alert group so
    # Resolve closes fall + inactivity together when the UI only passed the primary id.
    # Explicit multi-id tokens stay frozen (late-arriving findings are not added).
    raw = str(bundle_value or "")
    if "|" not in raw and len(reviewed) == 1:
        group = find_patient_alert_group(
            EVENT_STORE,
            user["id"],
            primary_event_id=primary,
            patient_id=pid,
        )
        if group and group.get("child_event_ids"):
            reviewed = [int(x) for x in group["child_event_ids"]]

    # Re-check every reviewed id belongs to this patient + caregiver (reject foreign).
    for eid in reviewed:
        row = EVENT_STORE.get(int(eid))
        if row is None:
            continue
        if str(row.get("patient_id") or "") != pid:
            return None, [], None, "Not allowed: reviewed findings include another patient."
        if not caregiver_owns_patient(user["id"], str(row.get("patient_id") or "")):
            return None, [], None, "Not allowed: reviewed findings include another patient."
    return primary, reviewed, pid, None


def care_ack(user, event_id, panel="Home", patient_id=None, outcome=None, handle_note=None):
    """Caregiver ACK for a reviewed patient bundle (OPEN children only).

    Does not trust browser-supplied patient_id for authorization.
    Lab tools keep using acknowledge_event() without roster checks.
    """
    from src.care_home import _inbox_df

    user = _care_user(user)
    if not user:
        empty = refresh_dashboard(None, EVENT_STORE)
        return empty[1], "Sign in first.", empty[0], empty[2]
    on_details = bool(patient_id and str(patient_id).strip())
    primary, reviewed, pid, err = _care_bundle_auth(user, event_id)
    if err:
        note = err
        if on_details:
            html, _n, _p, _m, _pid, alerts, _eid, _vis, _live, *_ = care_open_details_fn(
                user, patient_id, EVENT_STORE,
            )
            return _inbox_df(EVENT_STORE, user["id"]), note, html, alerts
        cards, inbox, alerts = refresh_dashboard(user, EVENT_STORE, panel=panel)
        return inbox, note, cards, alerts
    who = user.get("name") or user.get("username") or "caregiver"
    result = EVENT_STORE.acknowledge_events_for_patient(
        pid, reviewed, handled_by=who, note=handle_note,
    )
    if not result.get("ok"):
        note = "Not allowed: reviewed findings include another patient."
        cards, inbox, alerts = refresh_dashboard(user, EVENT_STORE, panel=panel)
        return inbox, note, cards, alerts
    n_up = len(result.get("updated") or [])
    n_skip = len(result.get("skipped") or [])
    note = (
        f"Acknowledged **{n_up}** reviewed finding{'s' if n_up != 1 else ''} "
        f"for this patient"
        + (f" · {n_skip} skipped" if n_skip else "")
        + ". Alert stays active until Resolve. Does **not** force SAFE."
    )
    reconcile_patient_status(EVENT_STORE, pid)
    if on_details:
        html, _n, _p, _m, _pid, alerts, _eid, _vis, _live, *_ = care_open_details_fn(
            user, pid, EVENT_STORE, focus_event_id=primary,
        )
        return _inbox_df(EVENT_STORE, user["id"]), note, html, alerts
    cards, inbox, alerts = refresh_dashboard(user, EVENT_STORE, panel=panel)
    return inbox, note, cards, alerts


def care_resolve(user, event_id, panel="Home", patient_id=None, outcome=None, handle_note=None):
    """Caregiver resolve for a reviewed patient bundle — then opens History."""
    from src.alert_center import outcome_label
    from src.care_demo import get_patient
    from src.care_home import _inbox_df

    user = _care_user(user)
    stay = (panel or "Home").strip() or "Home"
    if not user:
        empty = refresh_dashboard(None, EVENT_STORE)
        return empty[1], "Sign in first.", empty[0], empty[2], stay
    on_details = bool(patient_id and str(patient_id).strip())
    primary, reviewed, pid, err = _care_bundle_auth(user, event_id)
    if err:
        note = err
        if on_details:
            html, _n, _p, _m, _pid, alerts, _eid, _vis, _live, *_ = care_open_details_fn(
                user, patient_id, EVENT_STORE,
            )
            return _inbox_df(EVENT_STORE, user["id"]), note, html, alerts, stay
        cards, inbox, alerts = refresh_dashboard(user, EVENT_STORE, panel=stay)
        return inbox, note, cards, alerts, stay
    who = user.get("name") or user.get("username") or "caregiver"
    out = (outcome or "checked_ok")
    if isinstance(out, (list, tuple)) and out:
        out = out[0]
    out_code = str(out).strip() or "checked_ok"
    n_review = len(reviewed)
    pname = (get_patient(pid) or {}).get("name") or pid
    result = EVENT_STORE.resolve_events_for_patient(
        pid,
        reviewed,
        handled_by=who,
        outcome=out_code,
        note=handle_note,
    )
    if not result.get("ok"):
        note = "Not allowed: reviewed findings include another patient."
        cards, inbox, alerts = refresh_dashboard(user, EVENT_STORE, panel=stay)
        return inbox, note, cards, alerts, stay
    n_up = len(result.get("updated") or [])
    n_skip = len(result.get("skipped") or [])
    note = (
        f"Resolved **{n_up}** of {n_review} reviewed finding{'s' if n_review != 1 else ''} "
        f"for **{pname}**"
        + (f" · outcome **{outcome_label(out_code)}**" if out_code else "")
        + (f" · {n_skip} skipped" if n_skip else "")
        + ". New findings that arrived after this review stay open."
    )
    reconcile_patient_status(EVENT_STORE, pid)
    cards, inbox, alerts = refresh_dashboard(user, EVENT_STORE, panel="History")
    note = (
        f"{note}  \nOpened in **History** — each finding is listed separately with its own snapshot."
    )
    return inbox, note, cards, alerts, "History"


# Browser-side: Gradio leaves getUserMedia running after the stream UI stops.
STOP_CAMERA_JS = """
() => {
  document.querySelectorAll('video').forEach(v => {
    const s = v.srcObject;
    if (s && s.getTracks) {
      s.getTracks().forEach(t => { try { t.stop(); } catch (e) {} });
    }
    v.srcObject = null;
  });
}
"""

# Click Gradio's built-in webcam / record controls inside the live preview card.
START_CAMERA_JS = """
() => {
  const root = document.querySelector('.live-camera-viewer') || document;
  const pick = (sel) => root.querySelector(sel) || document.querySelector(sel);
  const labels = ['webcam', 'Webcam', 'camera', 'Camera', 'record', 'Record'];
  const buttons = Array.from(document.querySelectorAll('button, [role="button"]'));
  for (const b of buttons) {
    const t = ((b.getAttribute('aria-label') || '') + ' ' + (b.innerText || '')).toLowerCase();
    if (labels.some(l => t.includes(l.toLowerCase()))) {
      // Prefer controls near the live preview
      if (!root.contains(b) && root !== document) continue;
      b.click();
      return [];
    }
  }
  // Fallback: any webcam-looking control on the page
  for (const b of buttons) {
    const t = ((b.getAttribute('aria-label') || '') + ' ' + (b.innerText || '')).toLowerCase();
    if (t.includes('webcam') || t.includes('camera')) { b.click(); break; }
  }
  return [];
}
"""


# ---------------------------------------------------------------- layout (separate Gradio mounts)
caregiver_demo = build_caregiver_ui(
    care_login, care_refresh,
    care_ack, care_resolve,
    care_switch_panel_phone,
    care_open_ana_phone, care_open_pedro_phone, care_open_maria_phone,
    care_review_alert_phone,
    care_back_to_patients_phone, care_run_details_phone,
    care_live_tick,
    care_initial_html=build_dashboard_html(EVENT_STORE, "cg1", "caregiver1", panel="Home"),
    care_select_session=care_select_session,
    care_filter_alerts=care_filter_alerts,
    care_open_patient=care_open_patient_phone,
    care_add_patient=care_add_patient,
    care_load_demo_pack=care_load_demo_pack,
    care_alert_watch=care_alert_watch,
    care_response_preview=care_response_preview,
)

camera_demo = build_camera_station_ui(
    station_start, station_stop, station_tick,
    initial_patient_label=_live_patient_label(),
)

# CV lab remains available for demos / debugging (not the caregiver phone UI).
lab_demo = build_ui(
    analyse_video, analyse_image, webcam_step, stop_webcam,
    STOP_CAMERA_JS, START_CAMERA_JS, EXAMPLES,
    acknowledge_event, resolve_event,
    care_login, care_refresh, care_room_check,
    care_ack, care_resolve,
    care_switch_panel,
    care_open_ana, care_open_pedro, care_open_maria, care_review_alert,
    care_back_to_patients, care_run_details,
    care_live_preview, care_live_status,
    care_initial_html=build_dashboard_html(EVENT_STORE, "cg1", "caregiver1", panel="Home"),
    care_select_session=care_select_session,
)

# Backward-compatible name for imports/tests that expect `demo`.
demo = caregiver_demo

def _landing_html() -> str:
    import html as html_lib

    label = html_lib.escape(_live_patient_label())
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CareView</title>
<style>
  body{{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:#F1F5F9;color:#0F172A}}
  main{{max-width:560px;margin:0 auto;padding:40px 20px}}
  h1{{font-size:28px;margin:0 0 8px}}
  p{{color:#64748B;line-height:1.5}}
  a.card{{display:block;margin:14px 0;padding:18px;border-radius:16px;background:#fff;
    border:1px solid #E2E8F0;text-decoration:none;color:inherit;box-shadow:0 1px 2px rgba(15,23,42,.04)}}
  a.card b{{display:block;font-size:16px;margin-bottom:4px;color:#0F172A}}
  a.card span{{font-size:13px;color:#64748B}}
  a.card.primary{{border-color:#2563EB;background:#EFF6FF}}
  .note{{margin-top:24px;font-size:12px;color:#94A3B8}}
</style></head><body><main>
  <h1>CareView</h1>
  <p>Choose the interface for this device. Both share the same monitoring backend.</p>
  <a class="card primary" href="/caregiver">
    <b>Caregiver app</b>
    <span>Phone or desktop home. Patients, alerts, History, live feed view. No webcam controls.</span>
  </a>
  <a class="card" href="/camera-station">
    <b>Camera Station — {label}</b>
    <span>Laptop only — opens the webcam in Python (works on LAN HTTP).</span>
  </a>
  <a class="card" href="/lab">
    <b>CV lab tools</b>
    <span>Old video / photo activity analysis (optional development UI).</span>
  </a>
  <p class="note">Local demo only · Gradio public share disabled · same Wi‑Fi for phone access</p>
</main></body></html>
"""


def create_server_app():
    """FastAPI host with separate Caregiver + Camera Station Gradio mounts."""
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, RedirectResponse

    from src.playback import register_playback_routes

    api = FastAPI(title="CareView")

    @api.get("/")
    def root_to_caregiver():
        """Default entry is CareView caregiver Home — not the old lab UI."""
        return RedirectResponse(url="/caregiver", status_code=302)

    @api.get("/start", response_class=HTMLResponse)
    def start_chooser():
        return _landing_html()

    @api.get("/home")
    def home_alias():
        return RedirectResponse(url="/caregiver")

    def _jpeg_response(patient_id: str, *, require_seq: int | None = None):
        data, meta = LIVE_HUB.get_preview_jpeg(patient_id)
        if data is None:
            return Response(status_code=404, content=b"", media_type="image/jpeg")
        seq = int(meta.get("seq") or 0)
        if require_seq is not None and seq != int(require_seq):
            # Still return latest bytes; unique URL is for cache busting only.
            pass
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "ETag": f'"{seq}-{meta.get("bytes") or len(data)}"',
                "X-Frame-Seq": str(seq),
                "X-Frame-Age": str(meta.get("age_s") if meta.get("age_s") is not None else ""),
                "X-Jpeg-Bytes": str(meta.get("bytes") or len(data)),
            },
        )

    @api.get("/api/live/{patient_id}.jpg")
    def live_preview_jpeg(patient_id: str):
        """Latest caregiver JPEG (encoded once per hub publish; ETag = frame seq)."""
        return _jpeg_response(patient_id)

    @api.get("/api/live/{patient_id}/frame/{seq}.jpg")
    def live_preview_jpeg_seq(patient_id: str, seq: int):
        """Path-busting JPEG URL for clients that ignore query-string cache busting."""
        return _jpeg_response(patient_id, require_seq=seq)

    @api.get("/api/live/{patient_id}/meta")
    def live_preview_meta(patient_id: str):
        """Lightweight JSON for diagnostics / clients that poll seq only."""
        if patient_id != LIVE_HUB.patient_id:
            return {"ok": False, "patient_id": patient_id, "live_state": "OFFLINE"}
        snap = LIVE_HUB.snapshot()
        return {
            "ok": True,
            "patient_id": patient_id,
            "frame_seq": snap.get("frame_seq"),
            "live_state": snap.get("live_state"),
            "has_fresh_frame": snap.get("has_fresh_frame"),
            "frame_age_s": snap.get("frame_age_s"),
            "jpeg_bytes": snap.get("jpeg_bytes"),
            "camera_fps": snap.get("camera_fps"),
            "ai_fps": snap.get("ai_fps"),
            "preview_fps": snap.get("preview_fps"),
            "last_infer_ms": snap.get("last_infer_ms"),
            "publish_lag_ms": snap.get("publish_lag_ms"),
            "frames_skipped": snap.get("frames_skipped"),
            "activity": snap.get("activity"),
            "status": snap.get("status"),
            "session_id": snap.get("session_id"),
        }

    # P2: read-only session recording playback (not authenticated — LAN demo).
    register_playback_routes(api, EVENT_STORE)

    gr.mount_gradio_app(
        api, caregiver_demo, path="/caregiver",
        theme=gr.themes.Base(), css=CAREGIVER_APP_CSS, footer_links=[],
    )
    gr.mount_gradio_app(
        api, camera_demo, path="/camera-station",
        theme=gr.themes.Base(), css=CAMERA_STATION_CSS, footer_links=[],
    )
    gr.mount_gradio_app(
        api, lab_demo, path="/lab",
        theme=gr.themes.Base(), css=APP_CSS, footer_links=["runs"],
    )
    return api


if __name__ == "__main__":
    import socket

    import uvicorn

    def _lan_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"

    lan = _lan_ip()
    print("CareView (same Wi‑Fi, no public share):")
    print(f"  Caregiver Home:  http://{lan}:7860/          → redirects to /caregiver")
    print(f"  Caregiver app:   http://{lan}:7860/caregiver   ← phone / group demo")
    print(f"  Camera Station:  http://{lan}:7860/camera-station  ← laptop webcam")
    print(f"  Interface pick:  http://{lan}:7860/start")
    print(f"  CV lab (old):    http://{lan}:7860/lab")
    print("Login: caregiver / care123 · Demo patients: Alex (live) + Casey (ready fall)")
    print("Note: Camera Station uses OpenCV (not browser getUserMedia) so LAN HTTP works.")
    uvicorn.run(create_server_app(), host="0.0.0.0", port=7860, log_level="info")
