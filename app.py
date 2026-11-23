"""Activity monitor app (browser UI).

    python app.py            # then open http://127.0.0.1:7860

Three tabs: upload a video, upload a photo, or use the webcam live. It is the
same pipeline as src/run.py and src/evaluate.py (src/pipeline.py), with the
parameters tuned on the val split. YOLO and the pose models are loaded ONCE at
start-up and shared, not reloaded per click or per frame.
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
from src.classifier import IGNORED  # noqa: E402
from src.detector import PersonDetector  # noqa: E402
from src.params import Params  # noqa: E402
from src.pipeline import MonitorPipeline  # noqa: E402
from src.session_summary import session_blurb, session_details  # noqa: E402
from ui import APP_CSS, build_ui  # noqa: E402

PARAMS = Params.load()
print("Loading models once...")
DETECTOR = PersonDetector()
DETECTOR(np.zeros((360, 640, 3), np.uint8))   # warm-up, so the first click is not 4 s slower
LOCK = threading.Lock()       # the shared YOLO model is used by one request at a time
EXAMPLES = ROOT / "tests" / "cases"


def new_pipeline(*, live: bool = False) -> MonitorPipeline:
    """Fresh tracking state around the shared YOLO model.

    live=True only for webcam: skips MOG2/camera-flow to raise live FPS.
    Video/photo Analyze keep live=False (same path as evaluate).
    """
    pipe = MonitorPipeline(PARAMS, use_yolo=False, live=live)
    pipe.detector = DETECTOR
    return pipe


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
    headline = (f"### {main.upper()}  ·  {details['confidence (share of frames)']:.0%} of frames  ·  "
                f"{details['processing (ms per frame, median)']} ms/frame ({details['processing FPS']} FPS)"
                f"{note}\n\n{blurb}")
    return str(out_path), headline, summary_table(labels, fps), timeline(times, labels), details


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
    fmt = lambda v: None if v is None else round(float(v), 2)  # noqa: E731
    details = {
        "activity": r["activity"],
        "people found (YOLO)": len(r["people"]),
        "person confidence (YOLO)": round(r["people"][0]["conf"], 2) if r["people"] else None,
        "knee angle (deg)": fmt(r.get("knee_angle")), "hip angle (deg)": fmt(r.get("hip_angle")),
        "thigh ratio": fmt(r.get("thigh_ratio")), "torso tilt (deg)": fmt(r.get("torso_angle")),
        "processing (ms)": round(ms, 1),
        "note": "A single photo has no motion, so it can only show standing, sitting or lying.",
    }
    headline = f"### {r['activity'].upper()}  ·  {len(r['people'])} person(s)  ·  {ms:.0f} ms"
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


def webcam_step(frame, state):
    """Called for every webcam frame the browser streams. One pipeline per
    browser session, kept in `state`, so tracking and smoothing carry over."""
    if frame is None:
        return None, "", "Start a session to build a summary.", _close_webcam_state(state)
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
    state["labels"].append(r["activity"])
    state["fall_events"] = r.get("fall_events", 0)
    state["inactive_events"] = r.get("inactive_events", 0)
    state["inactive_seconds"] = r.get("inactive_seconds", 0.0)
    state["zone_out"] = r.get("zone_out_frames", 0)
    state["zone_partial"] = r.get("zone_partial_frames", 0)
    flags = []
    if r.get("fall_alert"):
        flags.append("**FALL ALERT**")
    elif r.get("inactive_alert"):
        flags.append("**INACTIVE**")
    elif r.get("zone_alert"):
        flags.append(f"**{r.get('zone_label') or 'ZONE'}**")
    flag = ("  ·  " + "  ·  ".join(flags)) if flags else ""
    text = (f"### {r['activity'].upper()}  ·  {r['confidence']:.0%}  ·  "
            f"{ms:.0f} ms/frame ({1000 / max(ms, 1):.0f} FPS)"
            f"{flag}")
    # Live stream ~4 FPS (stream_every=0.25); approximate duration from label count.
    live_fps = 4.0
    dur = len(state["labels"]) / live_fps
    summary = "### Session so far\n" + session_blurb(
        state["labels"], duration_s=dur,
        fall_events=state["fall_events"],
        inactive_events=state["inactive_events"],
        inactive_seconds=state["inactive_seconds"],
        zone_out_frames=state["zone_out"],
        zone_partial_frames=state["zone_partial"],
        fps=live_fps,
    )
    return cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), text, summary, state


def stop_webcam(state):
    """Drop the pipeline and show a session summary. Do NOT replace the Camera
    Image with None — that breaks Gradio's webcam control until a full refresh."""
    summary = "### Camera off — no session data yet."
    if isinstance(state, dict) and state.get("labels"):
        live_fps = 4.0
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
        summary = f"### Session summary\n{blurb}\n\nClick **Start camera** for another run."
    _close_webcam_state(state)
    return (
        gr.skip(),
        None,
        "### Camera off — session ended.",
        summary,
        None,
    )


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


# ---------------------------------------------------------------- layout
demo = build_ui(
    analyse_video, analyse_image, webcam_step, stop_webcam,
    STOP_CAMERA_JS, START_CAMERA_JS, EXAMPLES,
)

if __name__ == "__main__":
    # Gradio 6 moved theme/css from Blocks() to launch(); without this the dark UI is unstyled.
    demo.launch(theme=gr.themes.Base(), css=APP_CSS, footer_links=["runs"])
