"""P1 / P1.1 session recording engine tests (H.264 browser-compatible)."""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_station import ServerCameraStation  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.live_hub import LiveMonitorHub  # noqa: E402
from src.session_recorder import (  # noqa: E402
    BROWSER_CODEC,
    SessionRecorder,
    mp4_has_avc1_markers,
    probe_h264_encoder,
    validate_h264_mp4,
)


def _bgr(h=48, w=64, val=40):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _decode_all(path: Path, limit: int = 200) -> tuple[int, tuple[int, int] | None]:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"could not open {path}"
    n = 0
    dims = None
    while n < limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n += 1
        if dims is None:
            dims = (frame.shape[1], frame.shape[0])
    cap.release()
    return n, dims


class TestH264ProbeAndValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_h264_probe_encodes_and_decodes(self):
        codec, ext = probe_h264_encoder(64, 48, 10.0, self.out)
        self.assertEqual(codec, BROWSER_CODEC)
        self.assertEqual(ext, ".mp4")

    def test_validate_rejects_large_undecodable_file(self):
        junk = self.out / "junk.mp4"
        # Large enough to previously pass the old size>=100 gate.
        junk.write_bytes(b"NOT_A_VALID_MP4_FILE" + b"\x00" * 5000)
        ok, err = validate_h264_mp4(junk, 64, 48, min_decoded_frames=1)
        self.assertFalse(ok)
        self.assertTrue(err)

    def test_validate_rejects_empty_file(self):
        empty = self.out / "empty.mp4"
        empty.write_bytes(b"")
        ok, err = validate_h264_mp4(empty, 64, 48)
        self.assertFalse(ok)
        self.assertIn("empty", err.lower())

    def test_validate_rejects_mp4v_as_not_browser_compatible(self):
        path = self.out / "mp4v.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for i in range(10):
            writer.write(_bgr(val=i * 20))
        writer.release()
        time.sleep(0.05)
        self.assertGreater(path.stat().st_size, 100)
        # OpenCV can often decode mp4v — still must fail browser-compat check.
        ok, err = validate_h264_mp4(path, 64, 48, min_decoded_frames=1)
        self.assertFalse(ok)
        self.assertIn("avc1", err.lower())


class TestSessionRecorder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writer_init_and_finalize_ready_h264(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=101, width=64, height=48, fps=10.0,
            queue_size=8, output_dir=self.out,
        )
        meta = rec.start()
        self.assertEqual(meta["recording_status"], "RECORDING")
        self.assertEqual(meta["recording_codec"], BROWSER_CODEC)
        self.assertTrue(str(meta["recording_path"]).endswith(".mp4"))
        t0 = time.perf_counter()
        for i in range(15):
            rec.submit(_bgr(val=i * 10), t0 + i * 0.05)
            time.sleep(0.01)
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "READY")
        self.assertEqual(final["recording_codec"], BROWSER_CODEC)
        self.assertGreater(final["recording_bytes"], 0)
        self.assertGreater(final["recording_frames_written"], 0)
        path = Path(final["recording_abs_path"])
        self.assertTrue(path.exists())
        self.assertTrue(mp4_has_avc1_markers(path))
        decoded, dims = _decode_all(path)
        self.assertGreaterEqual(decoded, 3)
        self.assertEqual(dims, (64, 48))

    def test_correct_codec_container_selected(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=111, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        meta = rec.start()
        self.assertEqual(meta["recording_status"], "RECORDING")
        self.assertEqual(meta["recording_codec"], "avc1")
        for i in range(10):
            rec.submit(_bgr(val=50 + i), time.perf_counter())
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "READY")
        self.assertEqual(final["recording_codec"], "avc1")
        blob = Path(final["recording_abs_path"]).read_bytes()
        self.assertIn(b"avc1", blob)
        self.assertIn(b"avcC", blob)
        self.assertNotIn(b"mp4v", blob)

    def test_dimensions_match_expected(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=112, width=128, height=96, fps=10.0,
            output_dir=self.out,
        )
        rec.start()
        t0 = time.perf_counter()
        for i in range(12):
            # Submit mismatched input — recorder must resize to configured dims.
            rec.submit(_bgr(h=48, w=64, val=30 + i), t0 + i * 0.1)
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "READY")
        self.assertEqual(final["recording_width"], 128)
        self.assertEqual(final["recording_height"], 96)
        _, dims = _decode_all(Path(final["recording_abs_path"]))
        self.assertEqual(dims, (128, 96))

    def test_multiple_frames_decode(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=113, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        rec.start()
        t0 = time.perf_counter()
        for i in range(25):
            rec.submit(_bgr(val=(i * 9) % 255), t0 + i * 0.1)
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "READY")
        decoded, _ = _decode_all(Path(final["recording_abs_path"]))
        self.assertGreaterEqual(decoded, 10)
        self.assertGreaterEqual(final["recording_frames_written"], 10)

    def test_large_undecodable_file_not_ready(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=114, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        meta = rec.start()
        self.assertEqual(meta["recording_status"], "RECORDING")
        t0 = time.perf_counter()
        for i in range(8):
            rec.submit(_bgr(val=40), t0 + i * 0.1)

        real_validate = validate_h264_mp4

        def _corrupt_then_validate(path, width, height, **kwargs):
            Path(path).write_bytes(b"CORRUPT" + b"\x00" * 8000)
            return real_validate(path, width, height, **kwargs)

        with mock.patch("src.session_recorder.validate_h264_mp4", side_effect=_corrupt_then_validate):
            final = rec.finalize()
        self.assertEqual(final["recording_status"], "FAILED")
        self.assertTrue(final["recording_error"])
        # Diagnostic file preserved.
        self.assertTrue(Path(final["recording_abs_path"]).exists())
        self.assertGreater(Path(final["recording_abs_path"]).stat().st_size, 100)

    def test_bounded_queue_drop_oldest(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=102, width=32, height=24, fps=10.0,
            queue_size=3, output_dir=self.out,
        )
        meta = rec.start()
        self.assertEqual(meta["recording_status"], "RECORDING")
        # Slow the writer so submit can overflow the bounded queue.
        deadline = time.time() + 2.0
        while rec._writer is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(rec._writer)
        orig_write = rec._writer.write

        def _slow_write(frame):
            time.sleep(0.05)
            return orig_write(frame)

        rec._writer.write = _slow_write  # type: ignore[method-assign]
        for i in range(40):
            rec.submit(_bgr(32, 24, val=i), time.perf_counter())
        snap = rec.snapshot()
        self.assertGreaterEqual(snap["recording_frames_dropped"], 1)
        final = rec.finalize()
        self.assertIn(final["recording_status"], ("READY", "FAILED"))

    def test_failed_writer_initialization(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=103, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        with mock.patch(
            "src.session_recorder.probe_h264_encoder",
            side_effect=RuntimeError("Browser-compatible H.264 unavailable: forced"),
        ):
            meta = rec.start()
        self.assertEqual(meta["recording_status"], "FAILED")
        self.assertIn("H.264", meta["recording_error"])
        self.assertFalse(rec.submit(_bgr()))

    def test_codec_unavailable_explicit_failure(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=115, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        with mock.patch(
            "src.session_recorder._H264MP4Writer",
            side_effect=RuntimeError("ffmpeg missing libx264"),
        ):
            meta = rec.start()
        self.assertEqual(meta["recording_status"], "FAILED")
        self.assertNotEqual(meta["recording_status"], "READY")
        self.assertTrue(meta["recording_error"])

    def test_disk_write_failure_sets_failed(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=104, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        meta = rec.start()
        self.assertEqual(meta["recording_status"], "RECORDING")

        class BoomWriter:
            def write(self, *_a, **_k):
                raise OSError("disk full")

            def release(self):
                return None

            def isOpened(self):
                return True

        rec._writer = BoomWriter()
        rec.submit(_bgr(), time.perf_counter())
        time.sleep(0.2)
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "FAILED")
        self.assertTrue(final["recording_error"])

    def test_interrupted_is_incomplete_not_ready(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=105, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        rec.start()
        rec.submit(_bgr(), time.perf_counter())
        time.sleep(0.05)
        final = rec.finalize(interrupted=True)
        self.assertEqual(final["recording_status"], "INCOMPLETE")
        self.assertNotEqual(final["recording_status"], "READY")

    def test_zero_byte_rejected(self):
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=106, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        rec.start()
        # Finalize with no frames written → invalid.
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "FAILED")
        self.assertEqual(final["recording_frames_written"], 0)

    def test_timestamp_aware_duration(self):
        """~2.0s span at 10 FPS should produce about 20 output frames (±1)."""
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=107, width=64, height=48, fps=10.0,
            output_dir=self.out,
        )
        rec.start()
        t0 = time.perf_counter()
        # Sparse submits: 5 frames across 2 seconds — writer should pad indices.
        for i, offset in enumerate((0.0, 0.5, 1.0, 1.5, 2.0)):
            rec.submit(_bgr(val=20 + i), t0 + offset)
            time.sleep(0.02)
        final = rec.finalize()
        self.assertEqual(final["recording_status"], "READY")
        written = final["recording_frames_written"]
        self.assertGreaterEqual(written, 19)
        self.assertLessEqual(written, 22)
        # Container duration ≈ frames/fps
        self.assertAlmostEqual(final["recording_duration_s"], written / 10.0, places=3)

    def test_metadata_persisted_on_session(self):
        store = EventStore(self.out / "t.db")
        sess = store.create_session(
            patient_id="p1-ana",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="",
            duration_s=0,
            source_type="live",
            session_status="ACTIVE",
            room_id="Room 01",
        )
        rec = SessionRecorder(
            patient_id="p1-ana", session_id=int(sess["id"]), width=64, height=48,
            fps=10.0, output_dir=self.out / "rec",
        )
        meta = rec.start()
        store.update_session_recording(
            int(sess["id"]),
            recording_path=meta["recording_path"],
            recording_status=meta["recording_status"],
            recording_started_at=meta["recording_started_at"],
            recording_fps=meta["recording_fps"],
            recording_width=meta["recording_width"],
            recording_height=meta["recording_height"],
            recording_codec=meta["recording_codec"],
        )
        t0 = time.perf_counter()
        for i in range(8):
            rec.submit(_bgr(), t0 + i * 0.1)
        final = rec.finalize()
        store.update_session_recording(
            int(sess["id"]),
            recording_path=final["recording_path"],
            recording_status=final["recording_status"],
            recording_ended_at=final["recording_ended_at"],
            recording_duration_s=final["recording_duration_s"],
            recording_bytes=final["recording_bytes"],
            recording_error=final["recording_error"] or "",
            recording_codec=final["recording_codec"],
        )
        row = store.get_session(int(sess["id"]))
        self.assertEqual(row["recording_status"], "READY")
        self.assertEqual(row["recording_codec"], "avc1")
        self.assertTrue(row["recording_path"])
        self.assertGreater(row["recording_bytes"], 0)

    def test_stale_active_marks_recording_incomplete(self):
        store = EventStore(self.out / "stale.db")
        sess = store.create_session(
            patient_id="p1-ana",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="",
            duration_s=0,
            source_type="live",
            session_status="ACTIVE",
        )
        store.update_session_recording(
            int(sess["id"]),
            recording_status="RECORDING",
            recording_path="outputs/recordings/p1-ana/session_x.mp4",
        )
        n = store.interrupt_stale_sessions()
        self.assertEqual(n, 1)
        row = store.get_session(int(sess["id"]))
        self.assertEqual(row["session_status"], "INTERRUPTED")
        self.assertEqual(row["recording_status"], "INCOMPLETE")


class _FakePipe:
    def process(self, frame, t):
        return {
            "activity": "standing",
            "confidence": 0.9,
            "annotated": frame,
            "fall_alert": False,
            "inactive_alert": False,
            "zone_alert": False,
        }

    def close(self):
        return None


class _FakeCap:
    def __init__(self, n=60):
        self.n = n
        self.i = 0
        self._opened = True
        self._last = None

    def isOpened(self):
        return self._opened

    def read(self):
        if self.i >= self.n:
            # Keep returning frames so capture doesn't go unavailable mid-test.
            self.i += 1
            self._last = np.full((48, 64, 3), self.i % 200, dtype=np.uint8)
            return True, self._last
        self.i += 1
        self._last = np.zeros((48, 64, 3), dtype=np.uint8)
        return True, self._last

    def grab(self):
        return False

    def retrieve(self):
        return True, self._last

    def set(self, *_a, **_k):
        return True

    def release(self):
        self._opened = False


class TestCameraStationRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "t.db")
        self.hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        self.lock = threading.Lock()
        self.rec_dir = Path(self.tmp.name) / "recordings" / "p1-ana"
        self.rec_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recording_starts_with_monitoring_and_hub_continues(self):
        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
            target_fps=12.0,
            record_fps=10.0,
        )
        submitted = {"n": 0}

        class _OkRec:
            def __init__(self, *a, **k):
                self.session_id = int(k.get("session_id") or a[1] if len(a) > 1 else 1)
                self.patient_id = k.get("patient_id") or "p1-ana"

            def start(self):
                return {
                    "recording_status": "RECORDING",
                    "recording_path": "outputs/recordings/p1-ana/session_test.mp4",
                    "recording_abs_path": str(self_rec_dir / "session_test.mp4"),
                    "recording_started_at": "2026-01-01T00:00:00+00:00",
                    "recording_ended_at": None,
                    "recording_duration_s": 0.0,
                    "recording_fps": 10.0,
                    "recording_width": 64,
                    "recording_height": 48,
                    "recording_codec": "avc1",
                    "recording_bytes": 0,
                    "recording_error": "",
                    "recording_frames_written": 0,
                    "recording_frames_submitted": 0,
                    "recording_frames_dropped": 0,
                    "recording_warn_30m": False,
                    "recording_session_id": self.session_id,
                    "recording_patient_id": self.patient_id,
                }

            def snapshot(self):
                return self.start()

            def submit(self, frame, mono_ts=None):
                submitted["n"] += 1
                return True

            def stop_accepting(self):
                return None

            def finalize(self, interrupted=False):
                return {
                    **self.start(),
                    "recording_status": "READY",
                    "recording_bytes": 1234,
                    "recording_ended_at": "2026-01-01T00:01:00+00:00",
                    "recording_duration_s": 2.0,
                    "recording_frames_written": 20,
                    "recording_frames_submitted": submitted["n"],
                }

            def mark_incomplete(self, error=""):
                return {**self.start(), "recording_status": "INCOMPLETE", "recording_error": error}

        self_rec_dir = self.rec_dir
        with mock.patch.object(station, "_open_capture", return_value=_FakeCap(40)):
            with mock.patch("src.camera_station.SessionRecorder", _OkRec):
                station.start(0)
                deadline = time.time() + 3.0
                while time.time() < deadline and self.hub.get_preview("p1-ana") is None:
                    time.sleep(0.05)
                self.assertIsNotNone(self.hub.get_preview("p1-ana"))
                time.sleep(0.25)
                self.assertGreater(submitted["n"], 0)
                finished = station.stop()
                self.assertEqual(finished["session_status"], "COMPLETED")
                row = self.store.get_session(int(finished["id"]))
                self.assertEqual(row["recording_status"], "READY")
                self.assertGreater(int(row["recording_bytes"] or 0), 0)

    def test_session_completes_when_recording_fails(self):
        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
            target_fps=15.0,
        )
        with mock.patch.object(station, "_open_capture", return_value=_FakeCap(40)):
            with mock.patch(
                "src.session_recorder.SessionRecorder.start",
                return_value={
                    "recording_status": "FAILED",
                    "recording_error": "forced",
                    "recording_session_id": 1,
                    "recording_path": None,
                    "recording_fps": 10,
                    "recording_width": 64,
                    "recording_height": 48,
                    "recording_codec": "",
                    "recording_bytes": 0,
                    "recording_frames_dropped": 0,
                    "recording_duration_s": 0,
                    "recording_started_at": None,
                    "recording_ended_at": None,
                    "recording_warn_30m": False,
                    "recording_patient_id": "p1-ana",
                    "recording_abs_path": None,
                    "recording_frames_written": 0,
                    "recording_frames_submitted": 0,
                },
            ):
                station.start(0)
                deadline = time.time() + 3.0
                while time.time() < deadline and self.hub.get_preview("p1-ana") is None:
                    time.sleep(0.05)
                self.assertIsNotNone(self.hub.get_preview("p1-ana"))
                finished = station.stop()
                self.assertEqual(finished["session_status"], "COMPLETED")

    def test_recording_failure_does_not_block_capture(self):
        """Capture path must not raise when recorder.submit fails."""
        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
            target_fps=20.0,
        )

        class BoomRec:
            def snapshot(self):
                return {"recording_status": "RECORDING", "recording_frames_dropped": 0,
                        "recording_duration_s": 0, "recording_fps": 10, "recording_error": ""}

            def submit(self, *_a, **_k):
                raise RuntimeError("submit boom")

            def stop_accepting(self):
                return None

            def finalize(self, interrupted=False):
                return {"recording_status": "FAILED", "recording_error": "submit boom",
                        "recording_session_id": 1}

            def mark_incomplete(self, error=""):
                return {"recording_status": "INCOMPLETE", "recording_error": error,
                        "recording_session_id": 1}

        with mock.patch.object(station, "_open_capture", return_value=_FakeCap(30)):
            station.start(0)
            station._recorder = BoomRec()
            time.sleep(0.3)
            # Still producing hub frames despite submit errors.
            deadline = time.time() + 2.0
            ok = False
            while time.time() < deadline:
                if self.hub.get_preview("p1-ana") is not None:
                    ok = True
                    break
                time.sleep(0.05)
            finished = station.stop()
            self.assertTrue(ok)
            self.assertEqual(finished["session_status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
