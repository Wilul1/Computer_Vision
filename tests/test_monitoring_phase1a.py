"""Phase 1A unit tests: status decisions, event store, snapshots, dedupe."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alert_actions import decide  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.monitoring import MonitoringSession  # noqa: E402


def _frame(h=48, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDecide(unittest.TestCase):
    def test_safe_when_no_concern(self):
        d = decide({"activity": "sitting", "fall_alert": False, "inactive_alert": False, "zone_alert": False})
        self.assertEqual(d["status"], "SAFE")
        self.assertEqual(d["event_type"], "ok")

    def test_inactive_attention(self):
        d = decide({"activity": "standing", "inactive_alert": True, "inactive_reason": "still 8s"})
        self.assertEqual(d["status"], "ATTENTION")
        self.assertEqual(d["event_type"], "inactive")

    def test_possible_fall_alert(self):
        d = decide({"activity": "lying", "fall_alert": True, "fall_reason": "standing -> lying"})
        self.assertEqual(d["status"], "ALERT")
        self.assertEqual(d["event_type"], "possible_fall")
        self.assertIn("Possible fall", d["reason"])

    def test_lying_alone_is_not_fall(self):
        d = decide({"activity": "lying", "fall_alert": False, "inactive_alert": False, "zone_alert": False})
        self.assertEqual(d["status"], "SAFE")

    def test_camera_unavailable(self):
        d = decide(camera_unavailable=True)
        self.assertEqual(d["status"], "ATTENTION")
        self.assertEqual(d["event_type"], "camera_unavailable")

    def test_priority_fall_over_inactive(self):
        d = decide({
            "activity": "lying",
            "fall_alert": True,
            "inactive_alert": True,
            "zone_alert": True,
            "zone_kind": "partial_body",
        })
        self.assertEqual(d["status"], "ALERT")
        self.assertEqual(d["event_type"], "possible_fall")

    def test_framing_not_restricted_zone_wording(self):
        d = decide({"activity": "unknown", "zone_alert": True, "zone_kind": "partial_body"})
        self.assertEqual(d["status"], "ATTENTION")
        self.assertEqual(d["event_type"], "partial_body")
        self.assertNotIn("restricted", d["reason"].lower())
        self.assertNotIn("restricted", d["action"].lower())


class TestEventStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.store = EventStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_event_is_open(self):
        e = self.store.create_event(
            patient_id="p1", event_type="possible_fall",
            action_text="Notify", reason_text="test",
        )
        self.assertIsNotNone(e)
        self.assertEqual(e["event_status"], "OPEN")
        self.assertIsNone(e["acknowledged_at"])
        self.assertIsNone(e["resolved_at"])

    def test_survives_reconnect(self):
        e = self.store.create_event(
            patient_id="p1", event_type="inactive", action_text="Check in",
        )
        store2 = EventStore(self.db)
        got = store2.get(e["id"])
        self.assertEqual(got["event_type"], "inactive")
        self.assertEqual(got["event_status"], "OPEN")

    def test_ack_does_not_imply_patient_safe(self):
        """ACK only changes event row; decide() status stays signal-based."""
        e = self.store.create_event(
            patient_id="p1", event_type="possible_fall", action_text="Notify",
        )
        self.store.acknowledge(e["id"])
        # Patient status still ALERT if fall signal present
        status = decide({"activity": "lying", "fall_alert": True})["status"]
        self.assertEqual(status, "ALERT")
        row = self.store.get(e["id"])
        self.assertEqual(row["event_status"], "ACKNOWLEDGED")
        self.assertIsNotNone(row["acknowledged_at"])

    def test_resolve_separate(self):
        e = self.store.create_event(
            patient_id="p1", event_type="inactive", action_text="Check",
        )
        self.store.acknowledge(e["id"])
        self.store.resolve(e["id"])
        row = self.store.get(e["id"])
        self.assertEqual(row["event_status"], "RESOLVED")
        self.assertIsNotNone(row["resolved_at"])

    def test_dedupe_active_same_type(self):
        e1 = self.store.create_event(
            patient_id="p1", event_type="possible_fall", action_text="Notify",
        )
        e2 = self.store.create_event(
            patient_id="p1", event_type="possible_fall", action_text="Notify",
        )
        self.assertIsNotNone(e1)
        self.assertIsNone(e2)

    def test_different_types_can_coexist(self):
        e1 = self.store.create_event(
            patient_id="p1", event_type="possible_fall", action_text="Notify",
        )
        e2 = self.store.create_event(
            patient_id="p1", event_type="inactive", action_text="Check",
        )
        self.assertIsNotNone(e1)
        self.assertIsNotNone(e2)

    def test_new_after_resolve_allowed(self):
        e1 = self.store.create_event(
            patient_id="p1", event_type="inactive", action_text="Check",
        )
        self.store.resolve(e1["id"])
        e2 = self.store.create_event(
            patient_id="p1", event_type="inactive", action_text="Check",
        )
        self.assertIsNotNone(e2)
        self.assertNotEqual(e1["id"], e2["id"])


class TestMonitoringSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.store = EventStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_path_persisted(self):
        mon = MonitoringSession(patient_id="p1", source="test", store=self.store)
        out = mon.update(
            {"activity": "lying", "fall_alert": True, "fall_reason": "drop"},
            _frame(),
        )
        self.assertTrue(out["event_created"])
        self.assertTrue(out["event"]["snapshot_path"])
        snap = ROOT / out["event"]["snapshot_path"]
        self.assertTrue(snap.exists())

    def test_failed_snapshot_does_not_crash(self):
        mon = MonitoringSession(patient_id="p1", source="test", store=self.store)
        # Invalid image material → imwrite fails softly
        out = mon.update(
            {"activity": "standing", "inactive_alert": True},
            "not-an-image",
        )
        self.assertTrue(out["event_created"])
        self.assertFalse(out["event"]["snapshot_path"])

    def test_no_repeat_snapshot_while_active(self):
        mon = MonitoringSession(patient_id="p1", source="test", store=self.store)
        r = {"activity": "lying", "fall_alert": True, "fall_reason": "drop"}
        a = mon.update(r, _frame())
        b = mon.update(r, _frame())
        self.assertTrue(a["event_created"])
        self.assertFalse(b["event_created"])
        self.assertEqual(len(self.store.list_recent("p1")), 1)

    def test_framing_does_not_open_alert(self):
        mon = MonitoringSession(patient_id="p1", source="test", store=self.store)
        out = mon.update(
            {
                "activity": "unknown",
                "zone_alert": True,
                "zone_kind": "partial_body",
            },
            _frame(),
        )
        self.assertEqual(out["event_type"], "partial_body")
        self.assertEqual(out["status"], "ATTENTION")
        self.assertFalse(out["event_created"])
        self.assertEqual(len(self.store.list_recent("p1")), 0)

        out2 = mon.update(
            {
                "activity": "no person",
                "zone_alert": True,
                "zone_kind": "out_of_view",
            },
            _frame(),
        )
        self.assertEqual(out2["event_type"], "out_of_view")
        self.assertFalse(out2["event_created"])
        self.assertEqual(len(self.store.list_recent("p1")), 0)


class TestEffectiveBoardStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.store = EventStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_open_fall_keeps_alert_even_if_last_frame_safe(self):
        from src.care_home import effective_status, reconcile_patient_status

        self.store.upsert_patient_state("p3", status="SAFE", activity="lying")
        self.store.create_event(
            patient_id="p3", event_type="possible_fall",
            action_text="Notify", reason_text="fall",
        )
        status, _, active = effective_status(self.store, "p3")
        self.assertEqual(status, "ALERT")
        self.assertEqual(len(active), 1)
        self.assertEqual(reconcile_patient_status(self.store, "p3"), "ALERT")
        self.assertEqual(self.store.get_patient_state("p3")["status"], "ALERT")


class TestRegressionImports(unittest.TestCase):
    def test_pipeline_still_imports(self):
        from src.pipeline import MonitorPipeline
        from src.classifier import classify_pose
        self.assertTrue(callable(classify_pose))
        self.assertTrue(MonitorPipeline)


if __name__ == "__main__":
    unittest.main()
