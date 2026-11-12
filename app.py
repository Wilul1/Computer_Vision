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

PARAMS = Params.load()
print("Loading models once...")
DETECTOR = PersonDetector()
DETECTOR(np.zeros((360, 640, 3), np.uint8))   # warm-up, so the first click is not 4 s slower
LOCK = threading.Lock()       # the shared YOLO model is used by one request at a time
EXAMPLES = ROOT / "tests" / "cases"


def new_pipeline() -> MonitorPipeline:
    """Fresh tracking state (pose tracker, MOG2 background, smoothing) around the
    already-loaded YOLO model."""
    pipe = MonitorPipeline(PARAMS, use_yolo=False)
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
    details = {
        "main activity": main,
        "confidence (share of frames)": round(sum(l == main for l in voting) / max(len(voting), 1), 2),
        "baseline (MOG2 blob) says": Counter(base_labels).most_common(1)[0][0],
        "motion events": events,
        "frames": len(labels),
        "video length (s)": round(len(labels) / fps, 1),
        "processing (ms per frame, median)": round(float(np.median(warm)), 1),
        "processing FPS": round(1000 / float(np.median(warm)), 1),
    }
    headline = (f"### {main.upper()}  ·  {details['confidence (share of frames)']:.0%} of frames  ·  "
                f"{details['processing (ms per frame, median)']} ms/frame ({details['processing FPS']} FPS)")
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
def webcam_step(frame, state):
    """Called for every webcam frame the browser streams. One pipeline per
    browser session, kept in `state`, so tracking and smoothing carry over."""
    if frame is None:
        return None, "", state
    if state is None:
        state = {"pipe": new_pipeline(), "start": time.perf_counter()}
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    with LOCK:
        t0 = time.perf_counter()
        r = state["pipe"].process(bgr, time.perf_counter() - state["start"])
        ms = (time.perf_counter() - t0) * 1000
    text = (f"### {r['activity'].upper()}  ·  {r['confidence']:.0%}  ·  motion events {r['motion_events']}  ·  "
            f"{ms:.0f} ms/frame")
    return cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), text, state


# ---------------------------------------------------------------- layout
with gr.Blocks(title="Activity Monitor") as demo:
    gr.Markdown("# Motion Detection and Activity Monitoring\n"
                "Detects a person and classifies **standing, sitting, walking, running or lying** "
                "(YOLO11n + MediaPipe Pose + rules tuned on validation data). "
                "Works best with a **fixed camera** and the **whole body in view**, 2–5 m away.")
    with gr.Tab("Video"):
        with gr.Row():
            vin = gr.Video(label="Video (mp4, avi, mov)", sources=["upload"])
            vout = gr.Video(label="Result")
        vbtn = gr.Button("Analyse video", variant="primary")
        vhead = gr.Markdown()
        with gr.Row():
            vsum = gr.Dataframe(label="Time per activity", interactive=False)
            vtl = gr.Dataframe(label="Timeline", interactive=False)
        vjson = gr.JSON(label="Details")
        examples = sorted(str(p) for p in EXAMPLES.glob("*.mp4"))
        if examples:
            gr.Examples(examples, inputs=vin, label="Test-case clips")
        vbtn.click(analyse_video, vin, [vout, vhead, vsum, vtl, vjson])
    with gr.Tab("Photo"):
        with gr.Row():
            iin = gr.Image(label="Photo", type="numpy", sources=["upload"])
            iout = gr.Image(label="Result")
        ibtn = gr.Button("Analyse photo", variant="primary")
        ihead = gr.Markdown()
        ijson = gr.JSON(label="Details")
        photos = sorted(str(p) for p in EXAMPLES.glob("*.jpg"))
        if photos:
            gr.Examples(photos, inputs=iin, label="Test-case photos")
        ibtn.click(analyse_image, iin, [iout, ihead, ijson])
    with gr.Tab("Webcam (live)"):
        gr.Markdown("Allow the browser to use the camera. Stand 2–3 m back so your whole body is in view.")
        with gr.Row():
            win = gr.Image(label="Camera", sources=["webcam"], streaming=True, type="numpy")
            wout = gr.Image(label="Result")
        whead = gr.Markdown()
        wstate = gr.State(None)
        win.stream(webcam_step, [win, wstate], [wout, whead, wstate], stream_every=0.1, concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
