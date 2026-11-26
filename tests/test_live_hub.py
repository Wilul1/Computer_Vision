"""Live camera hub + persistent live session tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_store import EventStore  # noqa: E402
from src.live_hub import ActivityTracker, LiveMonitorHub  # noqa: E402
from src.monitoring import MonitoringSession  # noqa: E402


def _rgb(h=48, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestLiveSessionPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.store = EventStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ensure_session_not_duplicated(self):
        hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        a = hub.ensure_session()
        b = hub.ensure_session()
        self.assertEqual(a, b)
        actives = [s for s in self.store.list_sessions("p1-ana") if s.get("session_status") == "ACTIVE"]
        self.assertEqual(len(actives), 1)

    def test_stop_finalizes_completed(self):
        hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        sid = hub.ensure_session()
        hub.ingest(
            annotated_rgb=_rgb(),
            activity="standing",
            confidence=0.8,
            decision={"status": "SAFE", "reason": "ok"},
        )
        row = hub.stop(interrupted=False)
        self.assertIsNotNone(row)
        self.assertEqual(row["session_status"], "COMPLETED")
        self.assertEqual(row["id"], sid)
        self.assertIsNone(self.store.get_active_session("p1-ana"))

    def test_interrupt_stale_on_restart(self):
        hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        hub.ensure_session()
        # Simulate process restart: new hub marks leftover ACTIVE as INTERRUPTED.
        hub2 = LiveMonitorHub(self.store, patient_id="p1-ana")
        self.assertIsNone(self.store.get_active_session("p1-ana"))
        sessions = self.store.list_sessions("p1-ana")
        self.assertTrue(any(s.get("session_status") == "INTERRUPTED" for s in sessions))
        self.assertIsNotNone(hub2)

    def test_preview_only_for_live_patient(self):
        hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        hub.ensure_session()
        hub.ingest(
            annotated_rgb=_rgb(32, 40),
            activity="sitting",
            confidence=0.7,
            decision={"status": "SAFE", "reason": "ok"},
        )
        self.assertIsNotNone(hub.get_preview("p1-ana"))
        self.assertIsNone(hub.get_preview("p3-cara"))

    def test_event_linked_to_session(self):
        hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        sid = hub.ensure_session()
        mon = MonitoringSession(
            patient_id="p1-ana", source="live", store=self.store, session_id=sid,
        )
        out = mon.update(
            {"activity": "lying", "fall_alert": True, "fall_reason": "drop"},
            np.zeros((48, 64, 3), dtype=np.uint8),
        )
        self.assertTrue(out["event_created"])
        self.assertEqual(out["event"]["session_id"], sid)

    def test_activity_segments_aggregate(self):
        hub = LiveMonitorHub(self.store, patient_id="p1-ana")
        sid = hub.ensure_session()
        tracker = ActivityTracker(self.store, "p1-ana", sid)
        tracker.observe("standing", 0.9)
        import time
        time.sleep(0.45)
        tracker.observe("sitting", 0.8)
        time.sleep(0.45)
        tracker.finalize()
        secs = self.store.activity_seconds_for_session(sid)
        self.assertIn("standing", secs)
        self.assertGreaterEqual(secs["standing"], 0.4)


if __name__ == "__main__":
    unittest.main()
