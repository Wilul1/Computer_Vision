"""Shared live monitoring hub for CareView.

Camera Station owns the laptop webcam via server-side OpenCV (see
src/camera_station.py), runs CV, and publishes the latest processed frame here.
Caregiver browsers only poll this hub — they never open the webcam.

Latest-frame cache only: one annotated frame (+ one JPEG) per active patient,
never an unbounded queue. JPEG is encoded once per publish and reused until
the next frame.
"""
from __future__ import annotations

import base64
import threading
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .care_demo import get_patient
from .event_store import EventStore
from .monitoring import MonitoringSession

LIVE_PATIENT_ID = "p1-ana"
# Caregiver preview: small JPEG, encoded once per published frame.
PREVIEW_MAX_W = 640
PREVIEW_MAX_H = 360
JPEG_QUALITY = 70
STALE_MAX_AGE_S = 8.0

_SHARED: LiveMonitorHub | None = None


def set_shared_hub(hub: "LiveMonitorHub") -> None:
    global _SHARED
    _SHARED = hub


def get_shared_hub() -> "LiveMonitorHub | None":
    return _SHARED


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resize_preview(rgb: np.ndarray, max_w: int = PREVIEW_MAX_W, max_h: int = PREVIEW_MAX_H) -> np.ndarray:
    """Downscale for caregiver preview while preserving aspect ratio."""
    if not isinstance(rgb, np.ndarray) or rgb.size == 0:
        return rgb
    h, w = rgb.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 0.999:
        return np.ascontiguousarray(rgb)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> tuple[bytes, float]:
    """Encode RGB → JPEG once. Returns (bytes, encode_ms)."""
    t0 = time.perf_counter()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    ms = (time.perf_counter() - t0) * 1000.0
    if not ok or buf is None:
        return b"", ms
    return buf.tobytes(), ms


class ActivityTracker:
    """Aggregate consecutive same-label intervals; flush segments on change."""

    def __init__(self, store: EventStore, patient_id: str, session_id: int):
        self.store = store
        self.patient_id = patient_id
        self.session_id = session_id
        self._label: str | None = None
        self._start: float | None = None
        self._conf_sum = 0.0
        self._conf_n = 0
        self._lock = threading.Lock()

    def observe(self, label: str | None, confidence: float | None = None) -> None:
        lab = (label or "unknown").lower()
        now = time.monotonic()
        conf = float(confidence) if confidence is not None else None
        with self._lock:
            if self._label is None:
                self._label = lab
                self._start = now
                self._conf_sum = conf or 0.0
                self._conf_n = 1 if conf is not None else 0
                return
            if lab == self._label:
                if conf is not None:
                    self._conf_sum += conf
                    self._conf_n += 1
                return
            self._flush_locked(now)
            self._label = lab
            self._start = now
            self._conf_sum = conf or 0.0
            self._conf_n = 1 if conf is not None else 0

    def finalize(self) -> None:
        with self._lock:
            self._flush_locked(time.monotonic())
            self._label = None
            self._start = None

    def _flush_locked(self, end_mono: float) -> None:
        if self._label is None or self._start is None:
            return
        duration = max(0.0, end_mono - self._start)
        # Ignore sub-frame noise; keep real short poses (>=0.4s).
        if duration < 0.4:
            return
        avg_conf = (self._conf_sum / self._conf_n) if self._conf_n else None
        ended = _utc_now()
        # Approximate start from duration (wall clock).
        try:
            end_dt = datetime.now(timezone.utc).replace(microsecond=0)
            start_dt = datetime.fromtimestamp(end_dt.timestamp() - duration, tz=timezone.utc)
            started = start_dt.replace(microsecond=0).isoformat()
            ended = end_dt.isoformat()
        except Exception:
            started = ended
        self.store.add_activity_segment(
            session_id=self.session_id,
            patient_id=self.patient_id,
            activity=self._label,
            started_at=started,
            ended_at=ended,
            duration_s=duration,
            confidence=avg_conf,
        )


class LiveMonitorHub:
    """Process-wide live session + latest annotated frame for caregiver poll."""

    def __init__(
        self,
        store: EventStore,
        patient_id: str = LIVE_PATIENT_ID,
        *,
        preview_max_w: int = PREVIEW_MAX_W,
        preview_max_h: int = PREVIEW_MAX_H,
        jpeg_quality: int = JPEG_QUALITY,
        stale_max_age_s: float = STALE_MAX_AGE_S,
    ):
        self.store = store
        self.patient_id = patient_id
        self.preview_max_w = int(preview_max_w)
        self.preview_max_h = int(preview_max_h)
        self.jpeg_quality = int(jpeg_quality)
        self.stale_max_age_s = float(stale_max_age_s)
        self._lock = threading.RLock()
        # Latest-frame cache (bounded: one slot).
        self._frame: np.ndarray | None = None  # resized RGB preview
        self._full_frame: np.ndarray | None = None  # full annotated RGB (snapshots/CV)
        self._frame_ts: float | None = None
        self._frame_seq: int = 0
        self._jpeg: bytes | None = None
        self._jpeg_b64: str | None = None
        self._jpeg_encode_ms: float = 0.0
        self._session_id: int | None = None
        self._started_mono: float | None = None
        self._activity: str = ""
        self._confidence: float | None = None
        self._status: str = "SAFE"
        self._reason: str = ""
        self._connected: bool = False
        self._tracker: ActivityTracker | None = None
        self._monitor: MonitoringSession | None = None
        self._events_at_start = 0
        # Perf counters (updated by Camera Station / ingest).
        self._last_infer_ms: float | None = None
        self._infer_ms_ema: float | None = None
        self._publish_count: int = 0
        self._jpeg_hits: int = 0
        self._jpeg_misses: int = 0
        self._frames_skipped: int = 0
        self._camera_fps: float | None = None
        self._ai_fps: float | None = None
        self._preview_polls: int = 0
        self._preview_deliveries: int = 0
        self._publish_t0: float | None = None
        self._capture_ts: float | None = None
        self._publish_lag_ms: float | None = None
        # Stale ACTIVE rows from a previous process → INTERRUPTED.
        self.store.interrupt_stale_sessions()
        set_shared_hub(self)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self._started_mono is not None and self._session_id is not None:
                elapsed = max(0.0, time.monotonic() - self._started_mono)
            age = None
            fresh = False
            if self._frame_ts is not None:
                age = max(0.0, time.time() - self._frame_ts)
                fresh = age <= self.stale_max_age_s and self._frame is not None
            live_state = self._live_state_locked(fresh)
            preview_fps = None
            if self._publish_t0 is not None and self._publish_count > 0:
                dt = max(time.perf_counter() - self._publish_t0, 1e-6)
                preview_fps = self._publish_count / dt
            return {
                "patient_id": self.patient_id,
                "active": self._session_id is not None and self._connected,
                "session_id": self._session_id,
                "connected": self._connected,
                "activity": self._activity,
                "confidence": self._confidence,
                "status": self._status,
                "reason": self._reason,
                "elapsed_s": elapsed,
                "updated_at": (
                    datetime.fromtimestamp(self._frame_ts, tz=timezone.utc).isoformat()
                    if self._frame_ts
                    else None
                ),
                "has_frame": self._frame is not None,
                "has_fresh_frame": fresh,
                "frame_age_s": age,
                "frame_seq": self._frame_seq,
                "live_state": live_state,
                "jpeg_bytes": len(self._jpeg) if self._jpeg else 0,
                "jpeg_encode_ms": self._jpeg_encode_ms,
                "preview_shape": (
                    list(self._frame.shape) if isinstance(self._frame, np.ndarray) else None
                ),
                "full_shape": (
                    list(self._full_frame.shape)
                    if isinstance(self._full_frame, np.ndarray)
                    else None
                ),
                "last_infer_ms": self._last_infer_ms,
                "infer_ms_ema": self._infer_ms_ema,
                "camera_fps": self._camera_fps,
                "ai_fps": self._ai_fps,
                "preview_fps": preview_fps,
                "frames_skipped": self._frames_skipped,
                "jpeg_cache_hits": self._jpeg_hits,
                "jpeg_cache_misses": self._jpeg_misses,
                "preview_polls": self._preview_polls,
                "preview_deliveries": self._preview_deliveries,
                "publish_count": self._publish_count,
                "capture_ts": self._capture_ts,
                "publish_lag_ms": self._publish_lag_ms,
            }

    def diagnostics_md(self) -> str:
        """Dev/station diagnostics — keep off the caregiver phone chrome."""
        s = self.snapshot()
        age = s.get("frame_age_s")
        age_s = f"{age:.1f}" if isinstance(age, (int, float)) else "—"
        cam = s.get("camera_fps")
        ai = s.get("ai_fps")
        prev = s.get("preview_fps")
        infer = s.get("last_infer_ms")
        ema = s.get("infer_ms_ema")
        return (
            f"**Perf** · state **{s.get('live_state')}** · seq `{s.get('frame_seq')}` · "
            f"age `{age_s}`s  \n"
            f"Camera FPS `{cam if cam is not None else '—'}` · "
            f"AI FPS `{ai if ai is not None else '—'}` · "
            f"preview FPS `{round(prev, 1) if prev is not None else '—'}`  \n"
            f"Infer `{round(infer, 1) if infer is not None else '—'} ms` "
            f"(ema `{round(ema, 1) if ema is not None else '—'} ms`) · "
            f"JPEG `{s.get('jpeg_bytes') or 0} B` · encode "
            f"`{round(s.get('jpeg_encode_ms') or 0, 2)} ms` · skipped `{s.get('frames_skipped')}`"
        )

    def _live_state_locked(self, fresh: bool) -> str:
        if fresh and self._connected and self._session_id is not None:
            return "LIVE"
        if self._session_id is not None or self._frame is not None:
            return "STALE"
        return "OFFLINE"

    def get_preview(
        self, patient_id: str | None = None, *, max_age_s: float | None = None,
    ) -> np.ndarray | None:
        """Latest resized RGB preview for caregiver view (copy)."""
        pid = patient_id or self.patient_id
        if pid != self.patient_id:
            return None
        age_limit = self.stale_max_age_s if max_age_s is None else float(max_age_s)
        with self._lock:
            self._preview_polls += 1
            if self._frame is None:
                return None
            if self._frame_ts and (time.time() - self._frame_ts) > age_limit:
                return None
            frame = self._frame
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return None
            out = np.ascontiguousarray(frame)
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            self._preview_deliveries += 1
            return out

    def get_full_frame(
        self, patient_id: str | None = None, *, max_age_s: float | None = None,
    ) -> np.ndarray | None:
        """Full-resolution annotated RGB (copy) for snapshots / CV needs."""
        pid = patient_id or self.patient_id
        if pid != self.patient_id:
            return None
        age_limit = self.stale_max_age_s if max_age_s is None else float(max_age_s)
        with self._lock:
            src = self._full_frame if self._full_frame is not None else self._frame
            if src is None:
                return None
            if self._frame_ts and (time.time() - self._frame_ts) > age_limit:
                return None
            out = np.ascontiguousarray(src)
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            return out

    def get_preview_jpeg(
        self,
        patient_id: str | None = None,
        *,
        since_seq: int | None = None,
        max_age_s: float | None = None,
    ) -> tuple[bytes | None, dict[str, Any]]:
        """Return cached JPEG bytes (no re-encode). None bytes if unchanged/stale.

        When since_seq equals the current sequence, returns (None, meta) so callers
        can skip re-sending the same image.
        """
        pid = patient_id or self.patient_id
        meta: dict[str, Any] = {
            "patient_id": pid,
            "seq": 0,
            "unchanged": False,
            "stale": False,
            "bytes": 0,
        }
        if pid != self.patient_id:
            return None, meta
        age_limit = self.stale_max_age_s if max_age_s is None else float(max_age_s)
        with self._lock:
            self._preview_polls += 1
            meta["seq"] = self._frame_seq
            if self._jpeg is None or self._frame is None:
                return None, meta
            age = (time.time() - self._frame_ts) if self._frame_ts else None
            meta["age_s"] = age
            if age is not None and age > age_limit:
                meta["stale"] = True
                return None, meta
            if since_seq is not None and int(since_seq) == self._frame_seq and self._frame_seq > 0:
                meta["unchanged"] = True
                self._jpeg_hits += 1
                meta["bytes"] = len(self._jpeg)
                return None, meta
            self._jpeg_hits += 1
            self._preview_deliveries += 1
            meta["bytes"] = len(self._jpeg)
            return self._jpeg, meta

    def preview_img_html(
        self,
        patient_id: str | None = None,
        *,
        seq: int | None = None,
        height: int = 320,
        empty_message: str = "Waiting for live frame from Camera Station…",
        embed_jpeg: bool = True,
    ) -> str:
        """HTML live preview.

        Default embeds a data-URI JPEG so in-app browsers (e.g. Messenger WebView)
        cannot freeze on a cached `/api/live/*.jpg` response. Sequence is included
        in data-seq for diagnostics. Optional URL mode uses a path that includes
        the seq so caches that ignore query strings still miss.
        """
        pid = patient_id or self.patient_id
        with self._lock:
            fresh = False
            if self._frame_ts is not None and self._frame is not None:
                age = max(0.0, time.time() - self._frame_ts)
                fresh = age <= self.stale_max_age_s
            use_seq = int(seq if seq is not None else self._frame_seq)
            jpeg = self._jpeg
            b64 = self._jpeg_b64
            age_s = (
                max(0.0, time.time() - self._frame_ts) if self._frame_ts is not None else None
            )
        if pid != self.patient_id or not fresh or jpeg is None:
            # Compact offline strip — avoid a full-viewport dark box that hides patient details.
            return (
                f'<div class="cv-live-empty cv-live-empty-compact" style="padding:12px 14px;text-align:center;'
                f'color:#64748B;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;'
                f'min-height:56px;max-height:88px;'
                f'display:flex;align-items:center;justify-content:center;font-size:13px">'
                f'{empty_message}</div>'
            )
        age_label = f"{age_s:.1f}s" if isinstance(age_s, (int, float)) else "—"
        if embed_jpeg:
            if not b64:
                b64 = base64.b64encode(jpeg).decode("ascii")
            src = f"data:image/jpeg;base64,{b64}"
        else:
            # Unique path segment defeats caches that ignore ?query.
            src = f"/api/live/{pid}/frame/{use_seq}.jpg"
        return (
            f'<div class="cv-live-wrap" data-seq="{use_seq}" data-age="{age_label}" '
            f'style="background:#0F172A;border-radius:12px;overflow:hidden;'
            f'text-align:center;min-height:{height}px">'
            f'<img src="{src}" alt="Live camera seq {use_seq}" width="640" height="360" '
            f'style="width:100%;max-height:{height}px;object-fit:contain;'
            f'display:block;margin:0 auto" loading="eager" decoding="async"/>'
            f'<div style="font-size:11px;color:#94A3B8;padding:4px 8px;text-align:left">'
            f'seq {use_seq} · age {age_label}</div>'
            f"</div>"
        )

    def ensure_session(self) -> int:
        """Create or reuse ACTIVE live session. Idempotent for repeated starts."""
        patient = get_patient(self.patient_id) or {}
        with self._lock:
            existing = self.store.get_active_session(self.patient_id)
            if existing is not None:
                self._session_id = int(existing["id"])
                if self._started_mono is None:
                    self._started_mono = time.monotonic()
                if self._tracker is None:
                    self._tracker = ActivityTracker(
                        self.store, self.patient_id, self._session_id,
                    )
                if self._monitor is None:
                    self._monitor = MonitoringSession(
                        patient_id=self.patient_id,
                        source="live",
                        store=self.store,
                        session_id=self._session_id,
                    )
                return self._session_id
            started = _utc_now()
            row = self.store.create_session(
                patient_id=self.patient_id,
                started_at=started,
                ended_at="",
                duration_s=0.0,
                source_type="live",
                source="laptop_webcam",
                final_status="",
                final_activity="",
                reason_text="Live monitoring started",
                event_count=0,
                session_status="ACTIVE",
                room_id=patient.get("room") or "Room 01",
            )
            self._session_id = int(row["id"])
            self._started_mono = time.monotonic()
            self._events_at_start = len(self.store.list_recent(self.patient_id, limit=500))
            self._tracker = ActivityTracker(self.store, self.patient_id, self._session_id)
            self._monitor = MonitoringSession(
                patient_id=self.patient_id,
                source="live",
                store=self.store,
                session_id=self._session_id,
            )
            self._connected = False
            return self._session_id

    def ingest(
        self,
        *,
        annotated_rgb: np.ndarray | None,
        activity: str | None,
        confidence: float | None,
        decision: dict,
        camera_unavailable: bool = False,
        inference_ms: float | None = None,
        frames_skipped: int = 0,
        camera_fps: float | None = None,
        ai_fps: float | None = None,
        capture_ts: float | None = None,
        publish_lag_ms: float | None = None,
    ) -> dict:
        """Replace the latest frame (bounded cache) + encode JPEG once."""
        with self._lock:
            if self._session_id is None:
                self.ensure_session()
            self._connected = not camera_unavailable and annotated_rgb is not None
            if frames_skipped:
                self._frames_skipped += int(frames_skipped)
            if camera_fps is not None:
                self._camera_fps = float(camera_fps)
            if ai_fps is not None:
                self._ai_fps = float(ai_fps)
            if inference_ms is not None:
                self._last_infer_ms = float(inference_ms)
                if self._infer_ms_ema is None:
                    self._infer_ms_ema = float(inference_ms)
                else:
                    self._infer_ms_ema = 0.8 * self._infer_ms_ema + 0.2 * float(inference_ms)
            if capture_ts is not None:
                self._capture_ts = float(capture_ts)
            if publish_lag_ms is not None:
                self._publish_lag_ms = float(publish_lag_ms)
            if annotated_rgb is not None:
                full = np.ascontiguousarray(annotated_rgb)
                if full.dtype != np.uint8:
                    full = np.clip(full, 0, 255).astype(np.uint8)
                preview = _resize_preview(full, self.preview_max_w, self.preview_max_h)
                jpeg, enc_ms = _encode_jpeg(preview, self.jpeg_quality)
                self._full_frame = full
                self._frame = preview
                self._jpeg = jpeg if jpeg else None
                self._jpeg_b64 = base64.b64encode(jpeg).decode("ascii") if jpeg else None
                self._jpeg_encode_ms = enc_ms
                self._jpeg_misses += 1  # new encode
                self._frame_ts = time.time()
                self._frame_seq += 1
                self._publish_count += 1
                if self._publish_t0 is None:
                    self._publish_t0 = time.perf_counter()
            self._activity = activity or ""
            self._confidence = confidence
            self._status = decision.get("status") or self._status
            self._reason = decision.get("reason") or ""
            if self._tracker is not None and not camera_unavailable:
                self._tracker.observe(activity, confidence)
            return self.snapshot()

    def mark_camera_unavailable(self, decision: dict) -> None:
        with self._lock:
            self._connected = False
            self._status = decision.get("status") or "ATTENTION"
            self._reason = decision.get("reason") or "Camera unavailable"
            self._activity = ""

    def stop(self, *, interrupted: bool = False) -> dict | None:
        """Finalize ACTIVE session → COMPLETED or INTERRUPTED."""
        with self._lock:
            sid = self._session_id
            if sid is None:
                self._reset_runtime()
                return None
            if self._tracker is not None:
                self._tracker.finalize()
            ended = _utc_now()
            elapsed = 0.0
            if self._started_mono is not None:
                elapsed = max(0.0, time.monotonic() - self._started_mono)
            activity_seconds = self.store.activity_seconds_for_session(sid)
            events_now = len(self.store.list_recent(self.patient_id, limit=500))
            new_events = max(0, events_now - self._events_at_start)
            status = "INTERRUPTED" if interrupted else "COMPLETED"
            row = self.store.finalize_session(
                sid,
                ended_at=ended,
                duration_s=elapsed,
                session_status=status,
                final_status=self._status,
                final_activity=self._activity,
                reason_text=self._reason,
                event_count=new_events,
                activity_seconds=activity_seconds,
            )
            self._reset_runtime()
            return row

    def monitor(self) -> MonitoringSession:
        with self._lock:
            if self._monitor is None:
                self.ensure_session()
            assert self._monitor is not None
            return self._monitor

    def _reset_runtime(self) -> None:
        """Invalidate session + preview cache (do not show old frames as LIVE)."""
        self._session_id = None
        self._started_mono = None
        self._tracker = None
        self._monitor = None
        self._connected = False
        self._frame = None
        self._full_frame = None
        self._frame_ts = None
        self._frame_seq += 1  # bump so clients drop cached img URLs
        self._jpeg = None
        self._jpeg_b64 = None
        self._jpeg_encode_ms = 0.0
        self._activity = ""
        self._confidence = None
        self._last_infer_ms = None
        self._camera_fps = None
        self._ai_fps = None
        self._publish_count = 0
        self._publish_t0 = None
        self._capture_ts = None
        self._publish_lag_ms = None
