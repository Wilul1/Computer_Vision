"""P4.1 — one caregiver alert per patient + reviewed-bundle ack/resolve."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alert_center import (
    build_alert_center_html,
    encode_alert_bundle,
    group_alerts_by_patient,
    list_caregiver_patient_alerts,
    parse_alert_bundle,
    patient_alert_counts,
)
from src.care_demo import get_caregiver
from src.care_home import build_dashboard_html, effective_status, gr_open_event_choices, reconcile_patient_status
from src.event_store import EventStore
from src.patient_roster import reset_roster_singleton


class TestBundleToken(unittest.TestCase):
    def test_roundtrip(self):
        token = encode_alert_bundle({
            "primary_event_id": 12,
            "child_event_ids": [12, 15, 18],
        })
        self.assertEqual(token, "12|12,15,18")
        primary, ids = parse_alert_bundle(token)
        self.assertEqual(primary, 12)
        self.assertEqual(ids, [12, 15, 18])

    def test_plain_int_compat(self):
        primary, ids = parse_alert_bundle(7)
        self.assertEqual(primary, 7)
        self.assertEqual(ids, [7])


class TestGrouping(unittest.TestCase):
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

    def test_fall_plus_inactive_one_card(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall",
            action_text="Notify", reason_text="standing -> lying",
            snapshot_path="outputs/alerts/snapshots/fall.jpg",
        )
        self.store.create_event(
            patient_id="p1-ana", event_type="inactive",
            action_text="Check", reason_text="still 8s",
        )
        groups = list_caregiver_patient_alerts(self.store, self.user["id"])
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["patient_id"], "p1-ana")
        self.assertEqual(g["finding_count"], 2)
        self.assertEqual(g["priority"], "High")
        self.assertEqual(g["overall_status"], "OPEN")
        self.assertEqual(len(g["why_lines"]), 2)
        self.assertIn("standing -> lying", g["why_lines"][0])
        self.assertEqual(g["snapshot_path"], "outputs/alerts/snapshots/fall.jpg")
        html = build_alert_center_html(self.store, self.user["id"])
        self.assertEqual(html.count('class="cv-ac-card"'), 1)
        self.assertIn("2 findings", html)
        self.assertIn("Possible fall", html)
        self.assertIn("Prolonged inactivity", html)
        self.assertNotIn("confirmed fall", html.lower())
        self.assertIn("not a confirmed diagnosis", html.lower())

    def test_three_findings_still_one_card(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        self.store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="I",
        )
        self.store.create_event(
            patient_id="p1-ana", event_type="camera_unavailable", action_text="C",
        )
        groups = list_caregiver_patient_alerts(self.store, self.user["id"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["finding_count"], 3)

    def test_two_patients_two_cards(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        self.store.create_event(
            patient_id="p2-ben", event_type="inactive", action_text="I",
        )
        groups = list_caregiver_patient_alerts(self.store, self.user["id"])
        self.assertEqual(len(groups), 2)
        pids = {g["patient_id"] for g in groups}
        self.assertEqual(pids, {"p1-ana", "p2-ben"})
        self.assertEqual(groups[0]["priority"], "High")

    def test_unknown_type_safe(self):
        kids = group_alerts_by_patient([
            {
                "id": 1, "patient_id": "p1-ana", "event_type": "weird_signal",
                "event_status": "OPEN", "created_at": "2026-11-27T10:00:00",
                "reason_text": "x", "action_text": "y",
            },
            {
                "id": 2, "patient_id": "p1-ana", "event_type": "inactive",
                "event_status": "OPEN", "created_at": "2026-11-27T10:01:00",
                "reason_text": "still", "action_text": "check",
            },
        ])
        self.assertEqual(len(kids), 1)
        self.assertEqual(kids[0]["finding_count"], 2)
        # Medium (inactive) beats Unknown
        self.assertEqual(kids[0]["priority"], "Medium")

    def test_overall_ack_state_and_counts(self):
        a = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        self.store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="I",
        )
        groups = list_caregiver_patient_alerts(self.store, self.user["id"])
        self.assertEqual(groups[0]["overall_status"], "OPEN")
        self.store.acknowledge(a["id"])
        groups2 = list_caregiver_patient_alerts(self.store, self.user["id"])
        self.assertEqual(groups2[0]["overall_status"], "OPEN")  # inactive still OPEN
        for c in groups2[0]["children"]:
            if int(c["id"]) == int(a["id"]):
                self.assertEqual(c["event_status"], "ACKNOWLEDGED")
        counts = patient_alert_counts(groups2)
        self.assertEqual(counts["active"], 1)
        self.assertEqual(counts["findings"], 2)
        self.assertEqual(counts["unacknowledged"], 1)

    def test_dropdown_uses_bundle_token(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        self.store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="I",
        )
        choices = gr_open_event_choices(self.store, self.user["id"])
        self.assertEqual(len(choices), 1)
        label, token = choices[0]
        self.assertIn("+1 more", label)
        primary, ids = parse_alert_bundle(token)
        self.assertIsNotNone(primary)
        self.assertEqual(len(ids), 2)


class TestBulkAckResolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "events.db"
        self.store = EventStore(db)
        self.roster = reset_roster_singleton(db)
        self.roster.seed_demo()
        self.user = get_caregiver("caregiver", "care123")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ack_reviewed_open_only(self):
        fall = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        idle = self.store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="I",
        )
        self.store.acknowledge(idle["id"])
        result = self.store.acknowledge_events_for_patient(
            "p1-ana", [fall["id"], idle["id"]], handled_by="cg1",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["updated"]), 1)
        self.assertEqual(int(result["updated"][0]["id"]), int(fall["id"]))
        self.assertEqual(self.store.get(fall["id"])["event_status"], "ACKNOWLEDGED")
        self.assertEqual(self.store.get(idle["id"])["event_status"], "ACKNOWLEDGED")
        # Ack must not force SAFE while findings remain active.
        self.store.upsert_patient_state(
            "p1-ana", status="ALERT", activity="lying", reason="possible",
        )
        reconcile_patient_status(self.store, "p1-ana")
        status, _st, active = effective_status(self.store, "p1-ana")
        self.assertEqual(status, "ALERT")
        self.assertTrue(active)

    def test_resolve_only_reviewed_ids_protects_new(self):
        fall = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall",
            action_text="N", reason_text="drop",
        )
        idle = self.store.create_event(
            patient_id="p1-ana", event_type="inactive",
            action_text="I", reason_text="still",
        )
        reviewed = [int(fall["id"]), int(idle["id"])]
        # New finding arrives after review capture.
        cam = self.store.create_event(
            patient_id="p1-ana", event_type="camera_unavailable", action_text="Cam",
        )
        result = self.store.resolve_events_for_patient(
            "p1-ana", reviewed, handled_by="cg1", outcome="checked_ok", note="ok",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["updated"]), 2)
        self.assertEqual(self.store.get(fall["id"])["event_status"], "RESOLVED")
        self.assertEqual(self.store.get(idle["id"])["event_status"], "RESOLVED")
        self.assertEqual(self.store.get(cam["id"])["event_status"], "OPEN")
        groups = list_caregiver_patient_alerts(self.store, self.user["id"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["finding_count"], 1)
        self.assertEqual(groups[0]["children"][0]["event_type"], "camera_unavailable")
        # History keeps individual resolved rows.
        hist = self.store.list_recent("p1-ana", limit=10)
        resolved = [h for h in hist if h["event_status"] == "RESOLVED"]
        self.assertEqual(len(resolved), 2)
        types = {h["event_type"] for h in resolved}
        self.assertEqual(types, {"possible_fall", "inactive"})
        html = build_dashboard_html(
            self.store, self.user["id"], "caregiver1", panel="History",
        )
        self.assertIn("Possible fall", html)
        self.assertIn("Prolonged inactivity", html)
        self.assertIn("still", html)
        self.assertIn('data-hist-select="1"', html)
        self.assertIn('data-hist-select="2"', html)

    def test_foreign_patient_ids_reject_without_mutate(self):
        fall = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        other = self.store.create_event(
            patient_id="p2-ben", event_type="inactive", action_text="I",
        )
        before_a = self.store.get(fall["id"])["event_status"]
        before_b = self.store.get(other["id"])["event_status"]
        result = self.store.resolve_events_for_patient(
            "p1-ana", [fall["id"], other["id"]], handled_by="cg1",
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "foreign_patient")
        self.assertEqual(result["updated"], [])
        self.assertEqual(self.store.get(fall["id"])["event_status"], before_a)
        self.assertEqual(self.store.get(other["id"])["event_status"], before_b)

    def test_ack_new_event_after_capture_stays_open(self):
        fall = self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        idle = self.store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="I",
        )
        reviewed = [int(fall["id"]), int(idle["id"])]
        late = self.store.create_event(
            patient_id="p1-ana", event_type="camera_unavailable", action_text="Cam",
        )
        result = self.store.acknowledge_events_for_patient(
            "p1-ana", reviewed, handled_by="cg1",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["updated"]), 2)
        self.assertEqual(self.store.get(late["id"])["event_status"], "OPEN")

    def test_home_recent_alerts_one_row_per_patient(self):
        self.store.create_event(
            patient_id="p1-ana", event_type="possible_fall", action_text="N",
        )
        self.store.create_event(
            patient_id="p1-ana", event_type="inactive", action_text="I",
        )
        html = build_dashboard_html(
            self.store, self.user["id"], "caregiver1", panel="Home",
        )
        self.assertEqual(html.count("data-bundle="), 1)
        self.assertIn("2 findings", html)
        self.assertIn("patient alert", html.lower())


if __name__ == "__main__":
    unittest.main()
