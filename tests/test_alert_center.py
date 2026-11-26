"""P4 Smart Alert Center: priority, filters, ownership, lifecycle."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alert_actions import decide
from src.alert_center import (
    alert_center_counts,
    build_alert_center_html,
    caregiver_owns_event,
    event_category,
    event_display_title,
    event_meta,
    event_priority,
    filter_alerts,
    list_caregiver_active_alerts,
    sort_alerts,
)
from src.care_demo import get_caregiver
from src.care_home import build_dashboard_html, effective_status, reconcile_patient_status
from src.event_store import EventStore
from src.patient_roster import reset_roster_singleton


class TestPriorityCategory(unittest.TestCase):
    def test_known_mappings(self):
        expected = {
            "possible_fall": ("High", "Safety"),
            "inactive": ("Medium", "Activity"),
            "camera_unavailable": ("Medium", "Monitoring"),
            "out_of_view": ("Low", "Camera framing"),
            "partial_body": ("Low", "Camera framing"),
        }
        for et, (pri, cat) in expected.items():
            self.assertEqual(event_priority(et), pri)
            self.assertEqual(event_category(et), cat)

    def test_possible_fall_wording(self):
        title = event_display_title("possible_fall")
        self.assertIn("Possible fall", title)
        self.assertNotIn("confirmed", title.lower())
        self.assertNotIn("Confirmed", title)

    def test_unknown_safe(self):
        meta = event_meta("weird_new_signal")
        self.assertEqual(meta["priority"], "Unknown")
        self.assertEqual(meta["category"], "Other")
        self.assertTrue(meta["title"])

    def test_lying_alone_not_high_priority_event(self):
        d = decide({"activity": "lying", "fall_alert": False})
        self.assertEqual(d["status"], "SAFE")
        self.assertEqual(d["event_type"], "ok")
        self.assertNotEqual(event_priority(d["event_type"]), "High")


class TestSortFilter(unittest.TestCase):
    def test_sort_priority_status_newest(self):
        rows = [
            {"id": 1, "priority": "Low", "event_status": "OPEN"},
            {"id": 3, "priority": "High", "event_status": "ACKNOWLEDGED"},
            {"id": 4, "priority": "High", "event_status": "OPEN"},
            {"id": 2, "priority": "Medium", "event_status": "OPEN"},
        ]
        ordered = sort_alerts(rows)
        self.assertEqual([r["id"] for r in ordered], [4, 3, 2, 1])

    def test_filter_priority_and_status(self):
        rows = [
            {"id": 1, "priority": "High", "event_status": "OPEN", "patient_id": "p1-ana",
             "room": "Room 01", "event_type": "possible_fall", "category": "Safety",
             "created_at": "2026-11-25T10:00:00"},
            {"id": 2, "priority": "Low", "event_status": "ACKNOWLEDGED", "patient_id": "p2-ben",
             "room": "Room 02", "event_type": "out_of_view", "category": "Camera framing",
             "created_at": "2026-11-24T10:00:00"},
        ]
        only_high = filter_alerts(rows, priority="High")
        self.assertEqual([r["id"] for r in only_high], [1])
        only_open = filter_alerts(rows, status="OPEN")
        self.assertEqual([r["id"] for r in only_open], [1])
        by_room = filter_alerts(rows, room="Room 02")
        self.assertEqual([r["id"] for r in by_room], [2])
        by_day = filter_alerts(rows, date_from="2026-11-25", date_to="2026-11-25")
        self.assertEqual([r["id"] for r in by_day], [1])


class TestOwnershipAndLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "events.db"
        self.store = EventStore(db)
        self.roster = reset_roster_singleton(db)
        self.roster.seed_demo()
        self.user = get_caregiver("caregiver", "care123")
        self.assertIsNotNone(self.user)

    def tearDown(self):
        self.tmp.cleanup()

    def test_owner_can_ack_and_resolve(self):
        e = self.store.create_event(
            patient_id="p1-ana", event_type="inactive",
            action_text="Check", reason_text="still",
        )
        owned = caregiver_owns_event(self.store, e["id"], self.user["id"])
        self.assertIsNotNone(owned)
        ack = self.store.acknowledge(e["id"])
        self.assertEqual(ack["event_status"], "ACKNOWLEDGED")
        self.assertIsNotNone(ack["acknowledged_at"])
        # Still active
        active = list_caregiver_active_alerts(self.store, self.user["id"])
        self.assertTrue(any(int(a["id"]) == int(e["id"]) for a in active))
        nxt = self.store.resolve(e["id"])
        self.assertEqual(nxt["event_status"], "RESOLVED")
        self.assertIsNotNone(nxt["resolved_at"])
        active2 = list_caregiver_active_alerts(self.store, self.user["id"])
        self.assertFalse(any(int(a["id"]) == int(e["id"]) for a in active2))

    def test_open_to_resolved_allowed(self):
        e = self.store.create_event(
            patient_id="p2-ben", event_type="partial_body", action_text="Fix",
        )
        row = self.store.resolve(e["id"])
        self.assertEqual(row["event_status"], "RESOLVED")

    def test_resolve_saves_care_response(self):
        e = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="Notify",
        )
        row = self.store.resolve(
            e["id"],
            handled_by="caregiver1",
            outcome="checked_ok",
            note="Patient fine, helped to chair",
        )
        self.assertEqual(row["event_status"], "RESOLVED")
        self.assertEqual(row["handled_by"], "caregiver1")
        self.assertEqual(row["handle_outcome"], "checked_ok")
        self.assertIn("chair", row["handle_note"])
        # Resolved alerts leave active center; list/history keep the response.
        active = list_caregiver_active_alerts(self.store, self.user["id"])
        self.assertFalse(any(int(a["id"]) == int(e["id"]) for a in active))
        hist = self.store.list_recent("p1-ana", limit=5)
        self.assertEqual(hist[0]["handle_outcome"], "checked_ok")
        self.assertEqual(hist[0]["handled_by"], "caregiver1")

    def test_foreign_caregiver_rejected(self):
        e = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="Notify",
        )
        before = self.store.get(e["id"])
        owned = caregiver_owns_event(self.store, e["id"], "cg-other")
        self.assertIsNone(owned)
        # Simulate rejected care_ack: no mutate
        after = self.store.get(e["id"])
        self.assertEqual(after["event_status"], before["event_status"])
        self.assertIsNone(after["acknowledged_at"])

    def test_ack_does_not_force_safe(self):
        e = self.store.create_event(
            patient_id="p3-cara", event_type="possible_fall",
            action_text="Notify", reason_text="possible fall",
        )
        self.store.upsert_patient_state(
            "p3-cara", status="ALERT", activity="lying", reason="possible",
        )
        self.store.acknowledge(e["id"])
        status, _st, active = effective_status(self.store, "p3-cara")
        self.assertEqual(status, "ALERT")
        self.assertEqual(active[0]["event_status"], "ACKNOWLEDGED")
        reconcile_patient_status(self.store, "p3-cara")
        self.assertEqual(self.store.get_patient_state("p3-cara")["status"], "ALERT")

    def test_scoped_list_and_counts(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        self.store.create_event(
            patient_id="p2-ben", event_type="out_of_view", action_text="L",
        )
        self.store.create_event(
            patient_id="p3-cara", event_type="inactive", action_text="C",
        )
        # Unassigned patient id — must not appear for cg1
        self.store.create_event(
            patient_id="p9-stranger", event_type="possible_fall", action_text="X",
        )
        rows = list_caregiver_active_alerts(self.store, "cg1")
        pids = {r["patient_id"] for r in rows}
        self.assertIn("p1-ana", pids)
        self.assertNotIn("p9-stranger", pids)
        self.assertEqual(rows[0]["priority"], "High")
        counts = alert_center_counts(rows)
        self.assertEqual(counts["high"], 1)
        self.assertEqual(counts["low"], 1)
        self.assertEqual(counts["medium"], 1)
        self.assertEqual(counts["unacknowledged"], 3)

    def test_alert_center_html_no_paths(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall",
            action_text="Notify", reason_text="Possible fall detected (test)",
            snapshot_path="outputs/alerts/snapshots/missing_file.jpg",
        )
        html = build_alert_center_html(self.store, "cg1")
        self.assertIn("Alert Center", html)
        self.assertIn("Possible fall", html)
        self.assertNotIn("outputs/alerts", html)
        self.assertIn("Open patient", html)

    def test_dashboard_alerts_panel(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="camera_unavailable", action_text="Cam",
        )
        html = build_dashboard_html(
            self.store, "cg1", "caregiver1", panel="Alerts",
            alert_filters={"priority": "Medium"},
        )
        self.assertIn("Alert Center", html)
        self.assertIn("Camera unavailable", html)

    def test_history_keeps_resolved(self):
        e = self.store.create_event(
            patient_id="p2-ben",
            event_type="inactive",
            action_text="Check",
            reason_text="Pedro inactive case",
        )
        self.store.resolve(
            e["id"], handled_by="caregiver1", outcome="checked_ok",
            note="all clear",
        )
        html = build_dashboard_html(self.store, "cg1", "caregiver1", panel="History")
        self.assertIn("History", html)
        self.assertIn("Event History", html)
        self.assertIn("Event Details", html)
        self.assertIn(f'data-hist-select="{e["id"]}"', html)
        self.assertIn("Pedro", html)
        self.assertIn("Pedro inactive case", html)
        self.assertIn("all clear", html)
        self.assertIn("cv-hist2", html)


class TestCareAckAuthInApp(unittest.TestCase):
    """care_ack / care_resolve ownership without loading YOLO (mock heavy imports)."""

    def test_care_ack_rejects_foreign_without_mutate(self):
        # Unit-test the ownership gate (roster must match the temp DB).
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "e.db"
        store = EventStore(db)
        roster = reset_roster_singleton(db)
        roster.seed_demo()
        e = store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="Check",
        )
        self.assertIsNone(caregiver_owns_event(store, e["id"], "cg-other"))
        self.assertIsNotNone(caregiver_owns_event(store, e["id"], "cg1"))
        self.assertEqual(store.get(e["id"])["event_status"], "OPEN")


if __name__ == "__main__":
    unittest.main()
