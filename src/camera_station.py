"""Server-side Camera Station capture (OpenCV).

Browser getUserMedia is unavailable on plain HTTP LAN IPs (no secure context),
which is why Gradio shows image.no_webcam_support. The laptop process owns the
physical webcam via OpenCV instead — one owner, shared through LiveMonitorHub.

Capture and AI run on separate threads so inference never blocks the camera
reader (avoids OpenCV buffer backlog / multi-second preview lag).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from .live_hub import LiveMonitorHub, _resize_preview
from .session_recorder import SessionRecorder


class CameraStationError(RuntimeError):
    """Readable camera failure for the Camera Station UI."""


class ServerCameraStation:
    """Owns the laptop webcam; publishes processed frames to LiveMonitorHub."""

    def __init__(
        self,
        hub: LiveMonitorHub,
        *,
        new_pipeline: Callable[..., Any],
        lock: threading.Lock,
        camera_index: int = 0,
        target_fps: float = 15.0,
        on_frame: Callable[[dict, str], None] | None = None,
        ui_preview_max_w: int = 480,
        ui_preview_max_h: int = 270,
        record_fps: float = 10.0,
        record_queue_size: int = 30,
    ):
        self.hub = hub
        self.new_pipeline = new_pipeline
        self.lock = lock
        self.camera_index = int(camera_index)
        self.target_fps = float(target_fps)
        self.on_frame = on_frame
        self.ui_preview_max_w = int(ui_preview_max_w)
        self.ui_preview_max_h = int(ui_preview_max_h)
        self.record_fps = float(record_fps)
        self.record_queue_size = int(record_queue_size)
        self._capture_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._owner = threading.Lock()
        self._running = False
        self._last_error = ""
        # Latest capture slot (overwrite-only; never a queue).
        self._cap_lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._latest_capture_ts: float | None = None
        self._latest_capture_seq: int = 0
        self._processed_capture_seq: int = 0
        self._raw_rgb: np.ndarray | None = None  # small UI preview
        self._annotated_rgb: np.ndarray | None = None  # small UI preview
        self._ui_seq: int = -1
        self._raw_preview_seq: int = 0
        self._raw_ui_seq: int = -1
        self._ui_raw_cache: np.ndarray | None = None
        self._ui_ann_cache: np.ndarray | None = None
        self._headline = "Camera Station Idle."
        self._panel = "Press **Start Monitoring** To Open The Laptop Webcam."
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_skipped = 0
        self._t0 = 0.0
        self._infer_sum_ms = 0.0
        self._last_infer_ms = 0.0
        self._capture_fps = 0.0
        self._process_fps = 0.0
        self._preview_lag_ms: float | None = None
        self._recorder: SessionRecorder | None = None
        self._recording_snap: dict[str, Any] = {}

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> str:
        return self._last_error

    def snapshot_ui(self) -> dict[str, Any]:
        """UI bundle. Skip heavy copies when hub/UI seq is unchanged."""
        hub = self.hub.snapshot()
        # Prefer annotated publish seq; fall back to capture seq for raw-only updates.
        seq = int(hub.get("frame_seq") or 0) * 1_000_000 + int(self._latest_capture_seq)
        raw = self._raw_rgb
        ann = self._annotated_rgb
        changed = seq != self._ui_seq
        if changed:
            self._ui_seq = seq
            raw_out = None if raw is None else np.ascontiguousarray(raw)
            ann_out = None if ann is None else np.ascontiguousarray(ann)
            self._ui_raw_cache = raw_out
            self._ui_ann_cache = ann_out
        else:
            raw_out = self._ui_raw_cache
            ann_out = self._ui_ann_cache
        rec = self._recorder.snapshot() if self._recorder is not None else dict(self._recording_snap)
        return {
            "running": self._running,
            "error": self._last_error,
            "raw": raw_out,
            "annotated": ann_out,
            "headline": self._headline,
            "panel": self._panel,
            "hub": hub,
            "frames": self._frames_processed,
            "frames_captured": self._frames_captured,
            "skipped": self._frames_skipped,
            "ui_changed": changed,
            "ui_seq": seq,
            "capture_fps": self._capture_fps,
            "process_fps": self._process_fps,
            "preview_lag_ms": self._preview_lag_ms,
            "recording": rec,
            "diagnostics": self._diagnostics_md(hub, rec),
        }

    def snapshot_raw_preview(self) -> tuple[np.ndarray | None, bool]:
        """Return one new raw preview, independently of the slower AI UI poll."""
        with self._cap_lock:
            seq = self._raw_preview_seq
            frame = self._raw_rgb
            if frame is None or seq == self._raw_ui_seq:
                return None, False
            self._raw_ui_seq = seq
            return np.ascontiguousarray(frame), True

    def _diagnostics_md(
        self, hub: dict[str, Any] | None = None, rec: dict[str, Any] | None = None,
    ) -> str:
        hub = hub or self.hub.snapshot()
        rec = rec if rec is not None else (
            self._recorder.snapshot() if self._recorder is not None else self._recording_snap
        )
        lag = self._preview_lag_ms
        lag_s = f"{lag:.0f}" if isinstance(lag, (int, float)) else "—"
        r_status = str(rec.get("recording_status") or "NONE").replace("_", " ").title()
        live_state = str(hub.get("live_state") or "Unknown").replace("_", " ").title()
        r_dur = float(rec.get("recording_duration_s") or 0.0)
        r_drop = int(rec.get("recording_frames_dropped") or 0)
        r_fps = rec.get("recording_fps") or self.record_fps
        r_err = rec.get("recording_error") or ""
        warn = ""
        if rec.get("recording_warn_30m") or r_dur >= 30 * 60:
            warn = "  \n⚠️ Recording Has Exceeded **30 Minutes** (Still Recording — Not Auto-Stopped)."
        err_line = f"  \nRecording Error: {r_err}" if r_err else ""
        return (
            f"**Station Performance** · Capture `{self._capture_fps:.1f} FPS` · "
            f"AI `{self._process_fps:.1f} FPS` · Inference `{self._last_infer_ms:.0f} ms` · "
            f"Preview Lag `{lag_s} ms`  \n"
            f"Captured `{self._frames_captured}` · Processed `{self._frames_processed}` · "
            f"Skipped `{self._frames_skipped}` · Hub Sequence `{hub.get('frame_seq')}` · "
            f"JPEG `{hub.get('jpeg_bytes') or 0} B` · State **{live_state}**  \n"
            f"**Recording** · **{r_status}** · `{r_dur:.0f}s` · Target `{r_fps:g} FPS` · "
            f"Dropped `{r_drop}`"
            f"{warn}{err_line}"
        )

    def start(self, camera_index: int | None = None) -> str:
        """Start capture. Raises CameraStationError with a readable reason."""
        if not self._owner.acquire(blocking=False):
            raise CameraStationError(
                "Camera Station is already starting or stopping. Wait a moment and try again."
            )
        try:
            if self._running:
                return "Monitoring already active — laptop webcam is in use by this station."
            if camera_index is not None:
                self.camera_index = int(camera_index)
            self._last_error = ""
            self._stop.clear()
            cap = self._open_capture(self.camera_index)
            # Probe one frame before starting worker threads.
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise CameraStationError(
                    f"No camera image from device index {self.camera_index}. "
                    "Check that a webcam is connected and not used by another app."
                )
            with self._cap_lock:
                self._latest_bgr = frame
                self._latest_capture_ts = time.time()
                self._latest_capture_seq = 1
                self._processed_capture_seq = 0
            self._raw_rgb = _resize_preview(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                self.ui_preview_max_w,
                self.ui_preview_max_h,
            )
            with self._cap_lock:
                self._raw_preview_seq = 1
                self._raw_ui_seq = -1
            self._running = True
            self._frames_captured = 1
            self._frames_processed = 0
            self._frames_skipped = 0
            self._infer_sum_ms = 0.0
            self._t0 = time.perf_counter()
            self._ui_seq = -1
            sid = self.hub.ensure_session()
            # Recording uses raw capture frames; failure must not block monitoring.
            self._recorder = None
            self._recording_snap = {}
            try:
                h, w = frame.shape[:2]
                recorder = SessionRecorder(
                    patient_id=self.hub.patient_id,
                    session_id=int(sid),
                    width=w,
                    height=h,
                    fps=self.record_fps,
                    queue_size=self.record_queue_size,
                )
                rec_meta = recorder.start()
                self._recording_snap = dict(rec_meta)
                if rec_meta.get("recording_status") == "RECORDING":
                    self._recorder = recorder
                    self._persist_recording_meta(rec_meta)
                    # Seed with the probe frame so timeline starts immediately.
                    recorder.submit(frame, time.perf_counter())
                else:
                    self._persist_recording_meta(rec_meta)
            except Exception as exc:
                self._recording_snap = {
                    "recording_status": "FAILED",
                    "recording_error": str(exc),
                    "recording_session_id": sid,
                }
                try:
                    self._persist_recording_meta(self._recording_snap)
                except Exception:
                    pass
            self._capture_thread = threading.Thread(
                target=self._capture_loop, args=(cap,), name="careview-capture", daemon=True,
            )
            self._process_thread = threading.Thread(
                target=self._process_loop, name="careview-ai", daemon=True,
            )
            self._capture_thread.start()
            self._process_thread.start()
            rec_st = (self._recording_snap or {}).get("recording_status") or "NONE"
            return (
                f"Monitoring started on camera index **{self.camera_index}** "
                f"(capture + AI threads · recording **{rec_st}**)."
            )
        except CameraStationError as exc:
            self._last_error = str(exc)
            self._running = False
            raise
        except Exception as exc:
            self._last_error = str(exc)
            self._running = False
            raise CameraStationError(self._format_open_error(exc)) from exc
        finally:
            self._owner.release()

    def stop(self, *, interrupted: bool = False) -> dict[str, Any] | None:
        if not self._running and self._capture_thread is None and self._process_thread is None:
            # Still finalize any leftover recorder, then session.
            self._finalize_recorder(interrupted=interrupted)
            return self.hub.stop(interrupted=interrupted)
        self._stop.set()
        if self._recorder is not None:
            self._recorder.stop_accepting()
        for thread in (self._capture_thread, self._process_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=8.0)
        self._capture_thread = None
        self._process_thread = None
        self._running = False
        # Recording status must be known before session finalize.
        rec_meta = self._finalize_recorder(interrupted=interrupted)
        finished = self.hub.stop(interrupted=interrupted)
        with self._cap_lock:
            self._latest_bgr = None
        self._raw_rgb = None
        self._annotated_rgb = None
        self._ui_raw_cache = None
        self._ui_ann_cache = None
        self._ui_seq = -1
        r_status = str((rec_meta or {}).get("recording_status") or "NONE").replace("_", " ").title()
        self._headline = "### Camera Station Stopped"
        self._panel = (
            f"Session Finalized. Recording **{r_status}**. "
            "Press **Start Monitoring** For Another Run."
        )
        return finished

    def _persist_recording_meta(self, meta: dict[str, Any]) -> None:
        sid = meta.get("recording_session_id") or self.hub.snapshot().get("session_id")
        if sid is None:
            return
        self.hub.store.update_session_recording(
            int(sid),
            recording_path=meta.get("recording_path"),
            recording_status=meta.get("recording_status"),
            recording_started_at=meta.get("recording_started_at"),
            recording_ended_at=meta.get("recording_ended_at"),
            recording_duration_s=meta.get("recording_duration_s"),
            recording_fps=meta.get("recording_fps"),
            recording_width=meta.get("recording_width"),
            recording_height=meta.get("recording_height"),
            recording_codec=meta.get("recording_codec"),
            recording_bytes=meta.get("recording_bytes"),
            recording_error=meta.get("recording_error") or "",
        )

    def _finalize_recorder(self, *, interrupted: bool = False) -> dict[str, Any]:
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            meta = dict(self._recording_snap or {})
            if interrupted and meta.get("recording_status") in ("RECORDING", "FINALIZING", "NONE", None, ""):
                meta["recording_status"] = "INCOMPLETE"
                meta["recording_error"] = meta.get("recording_error") or "Recording interrupted"
                try:
                    self._persist_recording_meta(meta)
                except Exception:
                    pass
            self._recording_snap = meta
            return meta
        try:
            if interrupted:
                meta = recorder.mark_incomplete("Recording interrupted")
            else:
                meta = recorder.finalize(interrupted=False)
        except Exception as exc:
            meta = {
                "recording_status": "FAILED",
                "recording_error": str(exc),
                "recording_session_id": recorder.session_id,
                "recording_patient_id": recorder.patient_id,
            }
        self._recording_snap = dict(meta)
        try:
            self._persist_recording_meta(meta)
        except Exception:
            pass
        return meta

    def _open_capture(self, index: int) -> cv2.VideoCapture:
        # Prefer DirectShow on Windows for built-in / USB webcams.
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise CameraStationError(
                f"Could not open camera index {index}. "
                "Possible causes: no camera detected, permission denied by Windows, "
                "or the camera is already in use by another application."
            )
        # MJPG + modest size: USB cams deliver newer frames with less decode lag.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        # Prefer latest frame over deep buffers (reduces capture→display lag).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    @staticmethod
    def _format_open_error(exc: Exception) -> str:
        msg = str(exc) or exc.__class__.__name__
        return (
            f"Camera initialization failed: {msg}. "
            "Check device connection, Windows camera privacy settings, "
            "and that no other app holds the webcam."
        )

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        """Continuously read the webcam into a single latest-frame slot."""
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._last_error = (
                        "Camera disconnected or frame grab failed while monitoring."
                    )
                    self._headline = "### Camera Unavailable"
                    mon = self.hub.monitor().update(None, None, camera_unavailable=True)
                    self.hub.mark_camera_unavailable(mon)
                    self._panel = mon.get("panel_markdown") or self._last_error
                    if self._stop.wait(0.25):
                        break
                    continue
                now = time.time()
                mono = time.perf_counter()
                with self._cap_lock:
                    # Drop unread frames by overwrite — counted later by process loop.
                    self._latest_bgr = frame
                    self._latest_capture_ts = now
                    self._latest_capture_seq += 1
                    cap_seq = self._latest_capture_seq
                self._frames_captured += 1
                wall = max(time.perf_counter() - self._t0, 1e-6)
                self._capture_fps = self._frames_captured / wall
                # Raw recording submit is non-blocking (bounded queue / drop-oldest).
                if self._recorder is not None:
                    try:
                        self._recorder.submit(frame, mono)
                    except Exception:
                        pass
                # Local raw preview updates immediately (not gated on AI).
                raw_small = _resize_preview(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    self.ui_preview_max_w,
                    self.ui_preview_max_h,
                )
                with self._cap_lock:
                    self._raw_rgb = raw_small
                    self._raw_preview_seq = cap_seq
                self._preview_lag_ms = max(0.0, (time.time() - now) * 1000.0)
                # Tiny yield so AI / Gradio can run; avoid busy-spin if camera is fast.
                if self._stop.wait(0.001):
                    break
                _ = cap_seq  # silence lint; seq stored under lock
        except Exception as exc:
            self._last_error = f"Capture thread error: {exc}"
            self._headline = "### Camera Station Error"
            self._panel = self._last_error
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _take_latest(self) -> tuple[np.ndarray | None, float | None, int, int]:
        """Copy the newest unread capture. Returns (bgr, ts, seq, skipped)."""
        with self._cap_lock:
            if self._latest_bgr is None:
                return None, None, self._latest_capture_seq, 0
            seq = self._latest_capture_seq
            if seq == self._processed_capture_seq:
                return None, None, seq, 0
            skipped = max(0, seq - self._processed_capture_seq - 1)
            self._processed_capture_seq = seq
            frame = self._latest_bgr.copy()
            ts = self._latest_capture_ts
            return frame, ts, seq, skipped

    def _process_loop(self) -> None:
        """AI + hub publish at target_fps using the newest capture only."""
        pipe = None
        interval = 1.0 / max(self.target_fps, 0.5)
        try:
            pipe = self.new_pipeline(live=True)
            t_start = time.perf_counter()
            while not self._stop.is_set():
                loop_t0 = time.perf_counter()
                frame, capture_ts, _cap_seq, skipped = self._take_latest()
                if frame is None:
                    # Wait briefly for the next capture.
                    if self._stop.wait(0.01):
                        break
                    continue
                if skipped:
                    self._frames_skipped += skipped
                with self.lock:
                    t0 = time.perf_counter()
                    r = pipe.process(frame, time.perf_counter() - t_start)
                    ms = (time.perf_counter() - t0) * 1000.0
                self._last_infer_ms = ms
                self._infer_sum_ms += ms
                mon = self.hub.monitor().update(r, r.get("annotated"))
                # Fresh possible_fall → caregiver interrupt (banner + beep).
                if mon.get("event_created") and mon.get("event"):
                    self.hub.note_interrupt_event(
                        mon["event"].get("id"),
                        mon.get("event_type") or (mon.get("event") or {}).get("event_type"),
                    )
                annotated = r.get("annotated")
                if annotated is not None:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                else:
                    annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._frames_processed += 1
                wall = max(time.perf_counter() - self._t0, 1e-6)
                self._process_fps = self._frames_processed / wall
                publish_lag_ms = None
                if capture_ts is not None:
                    publish_lag_ms = max(0.0, (time.time() - capture_ts) * 1000.0)
                    self._preview_lag_ms = publish_lag_ms
                self.hub.ingest(
                    annotated_rgb=annotated_rgb,
                    activity=r.get("activity"),
                    confidence=r.get("confidence"),
                    decision=mon,
                    camera_unavailable=False,
                    inference_ms=ms,
                    frames_skipped=skipped,
                    camera_fps=self._capture_fps,
                    ai_fps=self._process_fps,
                    capture_ts=capture_ts,
                    publish_lag_ms=publish_lag_ms,
                )
                if self.on_frame is not None:
                    try:
                        self.on_frame(mon, r.get("activity") or "")
                    except Exception:
                        pass
                # Small annotated preview for Gradio (avoid shipping full-res every tick).
                self._annotated_rgb = _resize_preview(
                    annotated_rgb, self.ui_preview_max_w, self.ui_preview_max_h,
                )
                self._last_error = ""
                act = (r.get("activity") or "unknown").replace("_", " ").title()
                conf = float(r.get("confidence") or 0)
                flags = []
                if r.get("fall_alert"):
                    flags.append("Possible Fall")
                elif r.get("inactive_alert"):
                    flags.append("Inactive")
                elif r.get("zone_alert"):
                    flags.append(str(r.get("zone_label") or "Zone").replace("_", " ").title())
                flag = (" · " + " · ".join(flags)) if flags else ""
                # Show real process FPS, not instantaneous 1000/infer_ms (was misleading).
                self._headline = (
                    f"### {act} · {conf:.0%} · {ms:.0f} ms/Inference · "
                    f"AI {self._process_fps:.1f} FPS · Camera {self._capture_fps:.1f} FPS{flag}"
                )
                rec = self._recorder.snapshot() if self._recorder is not None else self._recording_snap
                self._panel = (
                    (mon.get("panel_markdown") or "")
                    + "\n\n"
                    + self._diagnostics_md(rec=rec)
                )
                elapsed = time.perf_counter() - loop_t0
                # Only rate-limit when ahead of target; never add sleep when already lagging.
                delay = interval - elapsed
                if delay > 0.002:
                    self._stop.wait(delay)
        except Exception as exc:
            self._last_error = f"AI thread error: {exc}"
            self._headline = "### Camera Station Error"
            self._panel = self._last_error
        finally:
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
            # Stop capture too if AI exits unexpectedly; stop() still joins both.
            self._stop.set()
            self._running = False
