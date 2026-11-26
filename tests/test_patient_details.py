"""Patient details page: filtering, sessions, ack/resolve status rules."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.care_demo import get_caregiver, get_patient  # noqa: E402
from src.care_home import effective_status, reconcile_patient_status  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.patient_details import (  # noqa: E402
    build_patient_details_html,
    open_patient_details,
)
from src.patient_roster import reset_roster_singleton  # noqa: E402


class TestPatientDetailsFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.store = EventStore(self.db)
        self.roster = reset_roster_singleton(self.db)
        self.roster.seed_demo()
        self.user = get_caregiver("caregiver", "care123")
        self.assertIsNotNone(self.user)
        # Seed distinct events per patient.
        self.store.create_event(
            patient_id="p1-ana", event_type="partial_body",
            action_text="Check framing", reason_text="Ana partial",
        )
        self.store.create_event(
            patient_id="p2-ben", event_type="inactive",
            action_text="Check Pedro", reason_text="Pedro inactive",
        )
        e_maria = self.store.create_event(
            patient_id="p3-cara", event_type="possible_fall",
            action_text="Notify nurse", reason_text="Maria fall",
        )
        self.maria_eid = e_maria["id"]
        self.store.upsert_patient_state(
            "p1-ana", status="ATTENTION", activity="unknown", reason="partial",
        )
        self.store.upsert_patient_state(
            "p2-ben", status="SAFE", activity="sitting", reason="ok",
        )
        self.store.upsert_patient_state(
            "p3-cara", status="ALERT", activity="lying", reason="fall",
        )
        today = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.store.create_session(
            patient_id="p3-cara",
            started_at=today,
            ended_at=today,
            duration_s=12.0,
            source_type="video",
            source="room:Room 03",
            final_status="ALERT",
            final_activity="lying",
            event_count=1,
            activity_seconds={"lying": 8.0, "standing": 4.0},
            label_counts={"lying": 8, "standing": 4},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_ana_page_only_ana(self):
        html = build_patient_details_html(self.store, self.user, "p1-ana")
        self.assertIn("Ana", html)
        self.assertIn("Room 01", html)
        self.assertIn("LIVE CAMERA", html)
        self.assertIn("Ana partial", html)
        self.assertNotIn("Pedro inactive", html)
        self.assertNotIn("Maria fall", html)
        self.assertIn("p1-ana", html)

    def test_pedro_page_only_pedro(self):
        html = build_patient_details_html(self.store, self.user, "p2-ben")
        self.assertIn("Pedro", html)
        self.assertIn("Room 02", html)
        self.assertIn("PRERECORDED VIDEO", html)
        self.assertIn("Pedro inactive", html)
        self.assertNotIn("Ana partial", html)
        self.assertNotIn("Maria fall", html)

    def test_maria_page_only_maria(self):
        html = build_patient_details_html(
            self.store, self.user, "p3-cara", focus_event_id=self.maria_eid,
        )
        self.assertIn("Maria", html)
        self.assertIn("Room 03", html)
        self.assertIn("PRERECORDED VIDEO", html)
        self.assertIn("Possible fall detected", html)
        self.assertIn("Maria fall", html)
        self.assertNotIn("Ana partial", html)
        self.assertNotIn("Pedro inactive", html)
        self.assertIn("Session #", html)
        self.assertIn("lying", html.lower())

    def test_alert_opens_correct_patient(self):
        out = open_patient_details(
            self.user, "p3-cara", self.store, focus_event_id=self.maria_eid,
        )
        html, note, _prev, _mon, pid, _alerts, eid, _vis, _live = out
        self.assertEqual(pid, "p3-cara")
        self.assertEqual(eid, self.maria_eid)
        self.assertIn("Maria", html)
        self.assertIn("Maria", note)

    def test_sessions_filtered_by_patient(self):
        self.assertEqual(len(self.store.list_sessions("p3-cara")), 1)
        self.assertEqual(len(self.store.list_sessions("p1-ana")), 0)
        self.assertEqual(len(self.store.list_sessions("p2-ben")), 0)

    def test_session_history_shows_no_recording_label(self):
        html = build_patient_details_html(self.store, self.user, "p3-cara")
        self.assertIn("Session #", html)
        self.assertIn("No recording", html)
        self.assertNotIn("/api/sessions/", html)

    def test_ack_does_not_force_safe(self):
        self.store.acknowledge(self.maria_eid)
        status = reconcile_patient_status(self.store, "p3-cara")
        self.assertEqual(status, "ALERT")
        eff, _, active = effective_status(self.store, "p3-cara")
        self.assertEqual(eff, "ALERT")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["event_status"], "ACKNOWLEDGED")

    def test_resolve_updates_lifecycle(self):
        self.store.acknowledge(self.maria_eid)
        self.store.resolve(self.maria_eid)
        row = self.store.get(self.maria_eid)
        self.assertEqual(row["event_status"], "RESOLVED")

    def test_resolved_leave_patient_current_alerts(self):
        self.store.resolve(self.maria_eid)
        html = build_patient_details_html(self.store, self.user, "p3-cara")
        self.assertIn("Current alerts", html)
        self.assertIn("No open alerts", html)
        self.assertNotIn("Maria fall", html)
        self.assertIn("History", html)

    def test_access_denied_for_unknown_patient(self):
        import gradio as gr

        with self.assertRaises(gr.Error):
            build_patient_details_html(self.store, self.user, "not-a-patient")

    def test_patient_ids_match_demo_roster(self):
        self.assertEqual(get_patient("p1-ana")["name"], "Ana")
        self.assertEqual(get_patient("p2-ben")["name"], "Pedro")
        self.assertEqual(get_patient("p3-cara")["name"], "Maria")


if __name__ == "__main__":
    unittest.main()
