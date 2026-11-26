"""Patient roster: add/list/clear (no Gradio)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.care_home import build_dashboard_html
from src.event_store import EventStore
from src.patient_roster import PatientRoster, reset_roster_singleton


class TestPatientRoster(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "e.db"
        self.store = EventStore(self.db)
        self.roster = reset_roster_singleton(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_starts_empty(self):
        self.assertEqual(self.roster.patients_for("cg1"), [])

    def test_add_and_list(self):
        row = self.roster.add_patient(name="Sam", room="Room 07", caregiver_id="cg1")
        self.assertEqual(row["name"], "Sam")
        self.assertEqual(row["room"], "Room 07")
        self.assertEqual(row["source_type"], "none")
        got = self.roster.patients_for("cg1")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["id"], row["id"])

    def test_live_patient_unique(self):
        self.roster.add_patient(
            name="Ana", room="Room 01", caregiver_id="cg1", source_type="live",
        )
        with self.assertRaises(ValueError):
            self.roster.add_patient(
                name="Other", room="Room 02", caregiver_id="cg1", source_type="live",
            )

    def test_live_camera_label_dynamic(self):
        self.assertEqual(self.roster.live_camera_label(), "No live patient assigned")
        self.roster.add_patient(
            name="Sam", room="Room 07", caregiver_id="cg1", source_type="live",
        )
        self.assertEqual(self.roster.live_camera_label(), "Sam · Room 07")
        live = self.roster.get_live_patient()
        self.assertIsNotNone(live)
        self.assertEqual(live["name"], "Sam")

    def test_careview_pack_alex_casey(self):
        rows = self.roster.seed_careview_pack()
        self.assertEqual([r["name"] for r in rows], ["Alex", "Casey"])
        live = self.roster.get_live_patient()
        self.assertIsNotNone(live)
        self.assertEqual(live["name"], "Alex")
        self.assertEqual(live["source_type"], "live")
        casey = self.roster.get("p-casey")
        self.assertEqual(casey["source_type"], "video")
        self.assertIn("fall-07", casey["video_path"].replace("\\", "/"))

    def test_home_empty_message(self):
        html = build_dashboard_html(self.store, "cg1", "caregiver1", panel="Home")
        self.assertIn("No patients yet", html)

    def test_wipe_events(self):
        self.store.create_event(
            patient_id="x", event_type="inactive", action_text="c",
        )
        self.store.wipe_all_monitoring_data()
        self.assertEqual(self.store.list_recent(limit=10), [])


if __name__ == "__main__":
    unittest.main()
