"""Caregiver live-feed polling from the shared LiveMonitorHub."""
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

from src.camera_station import ServerCameraStation  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.live_hub import LiveMonitorHub, set_shared_hub  # noqa: E402


class _FakePipe:
    def process(self, frame, t):
        # Paint a unique color so we can tell frames apart.
        out = frame.copy()
        out[:] = (0, 180, 0)
        return {
            "activity": "standing",
            "confidence": 0.91,
            "annotated": out,
            "fall_alert": False,
            "inactive_alert": False,
            "zone_alert": False,
        }

    def close(self):
        return None


class _FakeCap:
    def __init__(self, n=40):
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
        # No buffered extras in the fake camera.
        return False

    def retrieve(self):
        if self._last is None:
            return False, None
        return True, self._last

    def set(self, *_a, **_k):
        return True

    def release(self):
        self._opened = False


class TestCaregiverLiveFeed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "t.db")
        self.hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        set_shared_hub(self.hub)
        self.lock = threading.Lock()

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_frame_before_monitoring(self):
        self.assertIsNone(self.hub.get_preview("p1-ana"))
        snap = self.hub.snapshot()
        self.assertFalse(snap.get("has_fresh_frame"))

    def test_station_publishes_and_caregiver_reads_same_hub(self):
        station = ServerCameraStation(
            self.hub,
            new_pipeline=lambda live=False: _FakePipe(),
            lock=self.lock,
            target_fps=20.0,
        )
        with mock.patch.object(station, "_open_capture", return_value=_FakeCap(50)):
            station.start(0)
            deadline = time.time() + 3.0
            frame = None
            while time.time() < deadline:
                frame = self.hub.get_preview("p1-ana")
                if frame is not None:
                    break
                time.sleep(0.05)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape[2], 3)
            self.assertEqual(frame.dtype, np.dtype("uint8"))
            # Other patient must not receive Ana's feed.
            self.assertIsNone(self.hub.get_preview("p3-cara"))
            snap = self.hub.snapshot()
            self.assertTrue(snap.get("has_fresh_frame"))
            self.assertTrue(snap.get("active"))
            finished = station.stop()
            self.assertEqual(finished["session_status"], "COMPLETED")
            # After stop, frames go stale / cleared from "live" presentation.
            self.hub._frame_ts = time.time() - 30  # force stale
            self.assertIsNone(self.hub.get_preview("p1-ana"))
            self.assertFalse(self.hub.snapshot().get("has_fresh_frame"))

    def test_care_live_tick_logic(self):
        # Import after hub is shared so app-level helpers see the same process hub.
        # Avoid loading YOLO: call hub APIs the tick uses.
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=np.full((30, 40, 3), 120, dtype=np.uint8),
            activity="sitting",
            confidence=0.7,
            decision={"status": "SAFE", "reason": "ok"},
        )
        frame = self.hub.get_preview("p1-ana")
        self.assertIsNotNone(frame)
        self.assertIsNone(self.hub.get_preview("p2-ben"))


if __name__ == "__main__":
    unittest.main()
