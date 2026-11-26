"""Session video recorder for CareView Camera Station (P1 / P1.1).

Records raw BGR camera frames on a dedicated writer thread so capture, AI, and
live preview never wait on disk I/O.

P1.1 browser-compatible output:
  H.264 (`avc1`) inside MP4 via imageio-ffmpeg + libx264 (already used elsewhere
  in this project for browser-playable video). OpenH264 is intentionally NOT
  bundled: Cisco's binary patent license requires a separate end-user download
  and UI attribution, which this demo does not implement.

Timing strategy (target 10 FPS):
  Each accepted frame carries a monotonic timestamp. The writer maps elapsed
  session time to output frame indices: index = floor(elapsed * fps).
  Missing indices are filled by repeating the latest frame so container
  duration ≈ wall-clock session duration. Approximation: ±1/fps at the end,
  and large capture gaps become duplicated stills rather than a compressed
  timeline.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np

from . import config

RECORDING_STATUSES = (
    "NONE",
    "RECORDING",
    "FINALIZING",
    "READY",
    "FAILED",
    "INCOMPLETE",
)

# Browser-compatible target only. No silent mp4v / AVI fallback for READY.
BROWSER_CODEC = "avc1"
BROWSER_EXT = ".mp4"
BROWSER_ENCODER = "libx264"
_PROBE_FRAMES = 8
_VALIDATE_MIN_FRAMES = 1

_SENTINEL = object()
WARN_AFTER_S = 30 * 60  # 30 minutes — warn only, do not auto-stop


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def recording_dir(patient_id: str) -> Path:
    path = config.OUTPUTS_DIR / "recordings" / patient_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.ROOT.resolve()))
    except ValueError:
        return str(path)


def mp4_has_avc1_markers(path: Path) -> bool:
    """True when the MP4 appears to contain an H.264 (avc1) sample entry."""
    try:
        blob = path.read_bytes()
    except OSError:
        return False
    # Require both the sample-entry FourCC and the AVC decoder-config box.
    return b"avc1" in blob and b"avcC" in blob


@dataclass
class RecordingMeta:
    status: str = "NONE"
    path: str | None = None
    abs_path: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    bytes: int = 0
    frames_written: int = 0
    frames_submitted: int = 0
    frames_dropped: int = 0
    error: str = ""
    warn_30m: bool = False
    session_id: int | None = None
    patient_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "recording_status": self.status,
            "recording_path": self.path,
            "recording_abs_path": self.abs_path,
            "recording_started_at": self.started_at,
            "recording_ended_at": self.ended_at,
            "recording_duration_s": self.duration_s,
            "recording_fps": self.fps,
            "recording_width": self.width,
            "recording_height": self.height,
            "recording_codec": self.codec,
            "recording_bytes": self.bytes,
            "recording_frames_written": self.frames_written,
            "recording_frames_submitted": self.frames_submitted,
            "recording_frames_dropped": self.frames_dropped,
            "recording_error": self.error,
            "recording_warn_30m": self.warn_30m,
            "recording_session_id": self.session_id,
            "recording_patient_id": self.patient_id,
        }


class _H264MP4Writer:
    """Thin wrapper around imageio-ffmpeg libx264 for BGR frames."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._gen = None
        self._opened = False
        if self.width < 2 or self.height < 2:
            raise RuntimeError(f"Invalid recording dimensions: {self.width}x{self.height}")
        if self.width % 2 or self.height % 2:
            raise RuntimeError(
                f"H.264/yuv420p requires even dimensions, got {self.width}x{self.height}"
            )
        gen = imageio_ffmpeg.write_frames(
            str(self.path),
            (self.width, self.height),
            fps=self.fps,
            codec=BROWSER_ENCODER,
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-movflags", "+faststart", "-crf", "23"],
            ffmpeg_log_level="error",
        )
        gen.send(None)  # prime generator / spawn ffmpeg
        self._gen = gen
        self._opened = True

    def isOpened(self) -> bool:
        return bool(self._opened and self._gen is not None)

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError("H.264 writer is not open")
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise RuntimeError("H.264 writer expected a BGR ndarray")
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame_bgr = cv2.resize(
                frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA,
            )
        self._gen.send(np.ascontiguousarray(frame_bgr))

    def release(self) -> None:
        gen = self._gen
        self._gen = None
        self._opened = False
        if gen is None:
            return
        try:
            gen.close()
        except Exception:
            pass


def validate_h264_mp4(
    path: Path,
    expected_width: int,
    expected_height: int,
    *,
    min_decoded_frames: int = _VALIDATE_MIN_FRAMES,
    timeout_s: float = 3.0,
) -> tuple[bool, str]:
    """Validate a browser-compatible H.264 MP4. Returns (ok, error_message)."""
    result: dict[str, Any] = {"ok": False, "error": "Validation did not complete"}

    def _check() -> None:
        try:
            if path is None or not Path(path).is_file():
                result["error"] = "Recording file is missing"
                return
            size = int(Path(path).stat().st_size)
            if size <= 0:
                result["error"] = "Recording file is empty"
                return
            if not mp4_has_avc1_markers(Path(path)):
                result["error"] = (
                    "Recording is not H.264/avc1 (missing avc1/avcC markers); "
                    "not marked browser-compatible"
                )
                return
            cap = None
            try:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    result["error"] = "Recording could not be reopened"
                    return
                report_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                report_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                decoded = 0
                shape = None
                while decoded < max(1, int(min_decoded_frames)):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    decoded += 1
                    if shape is None:
                        shape = frame.shape
                if decoded < int(min_decoded_frames):
                    result["error"] = (
                        f"Could not decode enough frames "
                        f"(decoded={decoded}, required={min_decoded_frames})"
                    )
                    return
                frame_h, frame_w = int(shape[0]), int(shape[1])
                if frame_w != int(expected_width) or frame_h != int(expected_height):
                    result["error"] = (
                        f"Decoded dimensions {frame_w}x{frame_h} "
                        f"do not match expected {expected_width}x{expected_height}"
                    )
                    return
                if report_w and report_h:
                    if report_w != int(expected_width) or report_h != int(expected_height):
                        result["error"] = (
                            f"Container dimensions {report_w}x{report_h} "
                            f"do not match expected {expected_width}x{expected_height}"
                        )
                        return
                result["ok"] = True
                result["error"] = ""
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        except Exception as exc:
            result["ok"] = False
            result["error"] = f"Validation error: {exc}"

    t = threading.Thread(target=_check, name="careview-rec-validate", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return False, f"Recording validation timed out after {timeout_s:.1f}s"
    return bool(result.get("ok")), str(result.get("error") or "")


def probe_h264_encoder(width: int, height: int, fps: float, output_dir: Path) -> tuple[str, str]:
    """Encode+decode a short probe; return (codec_name, ext) or raise RuntimeError."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / f"_codec_probe_{BROWSER_CODEC}{BROWSER_EXT}"
    try:
        if probe_path.exists():
            probe_path.unlink()
    except OSError:
        pass

    writer = None
    try:
        writer = _H264MP4Writer(probe_path, fps=fps, width=width, height=height)
        if not writer.isOpened():
            raise RuntimeError("H.264 writer failed to open")
        for i in range(_PROBE_FRAMES):
            frame = np.full((height, width, 3), (20 + i * 17) % 255, dtype=np.uint8)
            # Visible motion so encoders cannot collapse to a degenerate stream.
            x = (i * 5) % max(1, width - 4)
            cv2.rectangle(frame, (x, 2), (x + 3, height - 2), (0, 255, 0), -1)
            writer.write(frame)
        writer.release()
        writer = None
        time.sleep(0.05)
        ok, err = validate_h264_mp4(
            probe_path,
            width,
            height,
            min_decoded_frames=min(3, _PROBE_FRAMES),
            timeout_s=3.0,
        )
        if not ok:
            raise RuntimeError(err or "H.264 probe validation failed")
        return BROWSER_CODEC, BROWSER_EXT
    except Exception as exc:
        raise RuntimeError(
            f"Browser-compatible H.264 (avc1/libx264) unavailable: {exc}. "
            "Recording will be marked FAILED; monitoring can continue. "
            "Install imageio-ffmpeg with libx264 support "
            "(already listed in requirements.txt)."
        ) from exc
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        try:
            probe_path.unlink(missing_ok=True)
        except OSError:
            pass


class SessionRecorder:
    """Bounded-queue raw-frame recorder with timestamp-aware 10 FPS H.264 output."""

    def __init__(
        self,
        *,
        patient_id: str,
        session_id: int,
        width: int,
        height: int,
        fps: float = 10.0,
        queue_size: int = 30,
        output_dir: Path | None = None,
    ):
        self.patient_id = patient_id
        self.session_id = int(session_id)
        # yuv420p / libx264 require even frame dimensions.
        self.width = max(2, int(width) - (int(width) % 2))
        self.height = max(2, int(height) - (int(height) % 2))
        self.fps = float(fps)
        self.queue_size = max(2, int(queue_size))
        self.output_dir = Path(output_dir) if output_dir else recording_dir(patient_id)

        self._q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._meta = RecordingMeta(
            status="NONE",
            fps=self.fps,
            width=self.width,
            height=self.height,
            session_id=self.session_id,
            patient_id=self.patient_id,
        )
        self._writer: _H264MP4Writer | None = None
        self._out_path: Path | None = None
        self._fourcc_name: str = BROWSER_CODEC
        self._writer_ready = threading.Event()
        self._writer_failed = False
        self._t0: float | None = None
        self._last_frame: np.ndarray | None = None
        self._frames_written = 0
        self._frames_submitted = 0
        self._frames_dropped = 0
        self._accepting = False

    @property
    def status(self) -> str:
        with self._lock:
            return self._meta.status

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = self._meta.as_dict()
            if self._meta.status == "RECORDING" and self._t0 is not None:
                elapsed = max(0.0, time.perf_counter() - self._t0)
                data["recording_duration_s"] = elapsed
                data["recording_warn_30m"] = elapsed >= WARN_AFTER_S
            data["recording_frames_written"] = self._frames_written
            data["recording_frames_submitted"] = self._frames_submitted
            data["recording_frames_dropped"] = self._frames_dropped
            return data

    def start(self) -> dict[str, Any]:
        """Probe H.264 and start writer thread (encoder opens on that thread)."""
        with self._lock:
            if self._meta.status in ("RECORDING", "FINALIZING"):
                return self._meta.as_dict()
        try:
            fourcc_name, ext = probe_h264_encoder(
                self.width, self.height, self.fps, self.output_dir,
            )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"session_{self.session_id}_{stamp}{ext}"
        except Exception as exc:
            with self._lock:
                self._meta.status = "FAILED"
                self._meta.error = str(exc)
                self._meta.ended_at = _utc_now()
                self._meta.codec = ""
            return self.snapshot()

        with self._lock:
            self._writer = None
            self._out_path = out_path
            self._fourcc_name = fourcc_name
            self._meta.status = "RECORDING"
            self._meta.codec = fourcc_name
            self._meta.path = rel_to_root(out_path)
            self._meta.abs_path = str(out_path.resolve())
            self._meta.started_at = _utc_now()
            self._meta.error = ""
            self._accepting = True
            self._frames_written = 0
            self._frames_submitted = 0
            self._frames_dropped = 0
            self._t0 = None
            self._last_frame = None
            self._writer_ready = threading.Event()
            self._writer_failed = False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._writer_loop, name=f"careview-recorder-{self.session_id}", daemon=True,
        )
        self._thread.start()
        # Wait briefly for the writer thread to open the encoder.
        self._writer_ready.wait(timeout=8.0)
        with self._lock:
            if self._writer_failed or self._meta.status == "FAILED":
                self._accepting = False
                return self.snapshot()
        return self.snapshot()

    def submit(self, frame_bgr: np.ndarray, mono_ts: float | None = None) -> bool:
        """Non-blocking submit. Returns False if not accepting or dropped as newest overflow."""
        if not self._accepting:
            return False
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return False
        ts = float(mono_ts if mono_ts is not None else time.perf_counter())
        # Copy so capture thread can overwrite its buffer safely.
        h, w = frame_bgr.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            frame = np.ascontiguousarray(frame_bgr)
        item = (frame, ts)
        try:
            self._q.put_nowait(item)
            with self._lock:
                self._frames_submitted += 1
            return True
        except queue.Full:
            # Drop-oldest: remove one, then enqueue newest.
            dropped = 0
            try:
                self._q.get_nowait()
                dropped = 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
                with self._lock:
                    self._frames_submitted += 1
                    self._frames_dropped += dropped
                return True
            except queue.Full:
                with self._lock:
                    self._frames_dropped += 1 + dropped
                return False

    def stop_accepting(self) -> None:
        self._accepting = False

    def finalize(self, *, interrupted: bool = False) -> dict[str, Any]:
        """Stop accepting frames, drain queue, release writer, validate file."""
        self._accepting = False
        with self._lock:
            status = self._meta.status
            if status in ("READY", "FAILED", "INCOMPLETE", "NONE") and self._thread is None:
                if interrupted and status == "RECORDING":
                    self._meta.status = "INCOMPLETE"
                return self.snapshot()
            if status == "FAILED" and self._thread is None:
                return self.snapshot()
            self._meta.status = "FINALIZING"
        try:
            self._q.put(_SENTINEL, timeout=2.0)
        except queue.Full:
            # Force sentinel by dropping one.
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(_SENTINEL)
            except queue.Full:
                self._stop.set()
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=20.0)
        self._thread = None
        return self._finish_locked(interrupted=interrupted)

    def mark_incomplete(self, error: str = "Recording interrupted") -> dict[str, Any]:
        self._accepting = False
        self._stop.set()
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
        self._release_writer()
        with self._lock:
            if self._meta.status not in ("READY", "FAILED"):
                self._meta.status = "INCOMPLETE"
                self._meta.error = error or self._meta.error
                self._meta.ended_at = _utc_now()
            return self.snapshot()

    def _writer_loop(self) -> None:
        """Open/write/release H.264 encoder entirely on this thread."""
        writer = None
        try:
            out_path = self._out_path
            if out_path is None:
                raise RuntimeError("Recording output path missing")
            writer = _H264MP4Writer(out_path, self.fps, self.width, self.height)
            self._writer = writer
            self._writer_ready.set()
            while True:
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    if self._stop.is_set() and self._q.empty():
                        break
                    continue
                if item is _SENTINEL:
                    while True:
                        try:
                            nxt = self._q.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is _SENTINEL:
                            continue
                        self._consume_frame(nxt[0], nxt[1])
                    break
                self._consume_frame(item[0], item[1])
            # Pad timeline to elapsed session time before release.
            if self._t0 is not None:
                self._pad_to_elapsed(max(0.0, time.perf_counter() - self._t0))
        except Exception as exc:
            with self._lock:
                self._meta.status = "FAILED"
                self._meta.error = f"Writer thread error: {exc}"
                self._writer_failed = True
            self._accepting = False
            self._writer_ready.set()
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            self._writer = None

    def _consume_frame(self, frame: np.ndarray, mono_ts: float) -> None:
        writer = self._writer
        if writer is None:
            return
        if self._t0 is None:
            self._t0 = mono_ts
        self._last_frame = frame
        elapsed = max(0.0, mono_ts - self._t0)
        # Number of output frames that should exist covering [0, elapsed].
        should_have = int(elapsed * self.fps) + 1
        # Cap per-submit catch-up (~1s) so a bad timestamp cannot freeze the writer.
        catchup_limit = max(1, int(self.fps) + 2)
        target = min(should_have, self._frames_written + catchup_limit)
        try:
            while self._frames_written < target:
                writer.write(frame)
                self._frames_written += 1
        except Exception as exc:
            with self._lock:
                self._meta.status = "FAILED"
                self._meta.error = f"Disk write failed: {exc}"
            self._accepting = False

    def _pad_to_elapsed(self, elapsed_s: float) -> None:
        if self._last_frame is None or self._writer is None:
            return
        # Hard cap: 2 hours of timeline at target fps.
        max_frames = int(self.fps * 60 * 120)
        should_have = min(max_frames, max(self._frames_written, int(round(elapsed_s * self.fps))))
        try:
            while self._frames_written < should_have:
                self._writer.write(self._last_frame)
                self._frames_written += 1
        except Exception as exc:
            with self._lock:
                self._meta.status = "FAILED"
                self._meta.error = f"Finalize pad write failed: {exc}"

    def _release_writer(self) -> None:
        # Preferred path: writer thread already released. Best-effort only.
        writer = self._writer
        self._writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

    def _finish_locked(self, *, interrupted: bool) -> dict[str, Any]:
        # Writer thread owns pad+release; here we only validate the file.
        elapsed = 0.0
        if self._t0 is not None:
            elapsed = max(0.0, time.perf_counter() - self._t0)
        self._release_writer()

        path = self._out_path
        size = 0
        if path is not None and path.exists():
            try:
                size = int(path.stat().st_size)
            except OSError:
                size = 0

        with self._lock:
            already_failed = self._meta.status == "FAILED" or bool(self._meta.error)
            self._meta.ended_at = _utc_now()
            self._meta.frames_written = self._frames_written
            self._meta.frames_submitted = self._frames_submitted
            self._meta.frames_dropped = self._frames_dropped
            self._meta.bytes = size
            if self._frames_written > 0 and self.fps > 0:
                self._meta.duration_s = self._frames_written / self.fps
            else:
                self._meta.duration_s = elapsed

            if interrupted and not already_failed:
                self._meta.status = "INCOMPLETE"
                if not self._meta.error:
                    self._meta.error = "Recording interrupted before clean finalize"
            elif already_failed:
                self._meta.status = "FAILED"
            elif path is None or size <= 0 or self._frames_written <= 0:
                self._meta.status = "FAILED"
                self._meta.error = self._meta.error or (
                    f"Invalid recording output (bytes={size}, frames={self._frames_written})"
                )
            else:
                # Strict READY: reopen + decode + dimension + avc1 markers.
                # Size alone is never sufficient. Preserve file on failure for diagnosis.
                time.sleep(0.05)
                min_frames = 1 if self._frames_written == 1 else min(3, self._frames_written)
                ok, err = validate_h264_mp4(
                    path,
                    self.width,
                    self.height,
                    min_decoded_frames=min_frames,
                    timeout_s=3.0,
                )
                if ok:
                    self._meta.status = "READY"
                    self._meta.error = ""
                    self._meta.codec = BROWSER_CODEC
                else:
                    self._meta.status = "FAILED"
                    self._meta.error = err or "Recording failed H.264 validation after finalize"
            return self.snapshot()
