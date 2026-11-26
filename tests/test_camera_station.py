"""Server-side Camera Station tests (no physical browser webcam)."""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_station import CameraStationError, ServerCameraStation  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.live_hub import LiveMonitorHub  # noqa: E402


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
    def __init__(self, n=30):
        self.n = n
        self.i = 0
        self._opened = True
        self._last = None

    def isOpened(self):
        return self._opened

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        self._last = np.zeros((48, 64, 3), dtype=np.uint8)
        return True, self._last

    def grab(self):
        return False

    def retrieve(self):
        if self._last is None:
            return False, None
        return True, self._last

    def set(self, *_a, **_k):
        return True

    def release(self):
        self._opened = False


class TestServerCameraStation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "t.db")
        self.hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        self.lock = threading.Lock()

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_stop_publishes_frames(self):
        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
            target_fps=20.0,
        )
        with mock.patch.object(station, "_open_capture", return_value=_FakeCap(40)):
            msg = station.start(0)
            self.assertIn("Monitoring started", msg)
            # Wait until hub has a frame.
            deadline = time.time() + 3.0
            while time.time() < deadline and self.hub.get_preview("p1-ana") is None:
                time.sleep(0.05)
            self.assertIsNotNone(self.hub.get_preview("p1-ana"))
            self.assertTrue(station.running)
            self.assertIsNotNone(self.store.get_active_session("p1-ana"))
            finished = station.stop()
            self.assertIsNotNone(finished)
            self.assertEqual(finished["session_status"], "COMPLETED")
            self.assertFalse(station.running)

    def test_open_failure_readable(self):
        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
        )

        class Boom:
            def isOpened(self):
                return False

            def release(self):
                return None

        with mock.patch.object(station, "_open_capture", side_effect=CameraStationError("Could not open camera index 0")):
            with self.assertRaises(CameraStationError) as ctx:
                station.start(0)
            self.assertIn("Could not open", str(ctx.exception))

    def test_caregiver_preview_never_needs_browser_webcam(self):
        # Hub preview API is frame-buffer only.
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=np.zeros((20, 20, 3), dtype=np.uint8),
            activity="sitting",
            confidence=0.5,
            decision={"status": "SAFE", "reason": "ok"},
        )
        frame = self.hub.get_preview("p1-ana")
        self.assertIsNotNone(frame)
        self.assertIsNone(self.hub.get_preview("p3-cara"))

    def test_hub_seq_keeps_advancing(self):
        """Capture+AI threads must keep publishing new seq numbers (not freeze)."""

        class _EndlessCap(_FakeCap):
            def read(self):
                self.i += 1
                self._last = np.full((48, 64, 3), self.i % 200, dtype=np.uint8)
                return True, self._last

        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
            target_fps=15.0,
        )
        with mock.patch.object(station, "_open_capture", return_value=_EndlessCap()):
            station.start(0)
            deadline = time.time() + 3.0
            first = None
            while time.time() < deadline:
                seq = int(self.hub.snapshot().get("frame_seq") or 0)
                if seq > 0:
                    first = seq
                    break
                time.sleep(0.05)
            self.assertIsNotNone(first)
            deadline = time.time() + 2.0
            later = first
            while time.time() < deadline:
                later = int(self.hub.snapshot().get("frame_seq") or 0)
                if later > first:
                    break
                time.sleep(0.05)
            station.stop()
            self.assertGreater(later, first)

    def test_ui_has_no_browser_webcam_source(self):
        from ui_camera_station import build_camera_station_ui

        text = Path(ROOT / "ui_camera_station.py").read_text(encoding="utf-8")
        self.assertNotIn('sources=["webcam"]', text)
        self.assertNotIn("WebcamOptions", text)
        self.assertIn("station_start", text)


if __name__ == "__main__":
    unittest.main()
