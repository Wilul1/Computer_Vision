"""Interface separation + caregiver mobile CSS smoke checks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestCaregiverMobileCss(unittest.TestCase):
    def test_mobile_nav_rules_present(self):
        from ui import CAREGIVER_APP_CSS

        self.assertIn("#cv-mobile-top", CAREGIVER_APP_CSS)
        self.assertIn("max-width:768px", CAREGIVER_APP_CSS)
        self.assertIn("position:fixed", CAREGIVER_APP_CSS)
        self.assertIn("#cv-side-nav fieldset", CAREGIVER_APP_CSS)
        self.assertIn("cv-patients{grid-template-columns:1fr", CAREGIVER_APP_CSS)
        compact = CAREGIVER_APP_CSS.replace(" ", "")
        # PC + phone: document/body scrolls (Gradio fixed wrappers block nested overflow).
        self.assertIn("overflow-y:scroll!important", compact)
        self.assertIn("position:relative!important", compact)
        self.assertIn("position:sticky", CAREGIVER_APP_CSS)
        self.assertIn("min-width:769px", compact)
        self.assertIn("#careview-shell{", CAREGIVER_APP_CSS)
        self.assertIn("cv-details-tools", CAREGIVER_APP_CSS)
        self.assertIn("cv-handle-tools", CAREGIVER_APP_CSS)
        self.assertIn("cv-ac-filter-wrap", CAREGIVER_APP_CSS)

    def test_caregiver_tools_are_simplified(self):
        text = (ROOT / "ui.py").read_text(encoding="utf-8")
        # Add patient is Settings-only; patient page uses light tool cards.
        self.assertIn("cv-settings-tools", text)
        self.assertIn("cv-details-tools", text)
        self.assertIn("cv-events-log", text)
        self.assertIn("cv-ac-filter-wrap", text)
        self.assertIn("as calert_filters", text)
        self.assertIn("as chandle_group", text)
        self.assertIn("as csettings_tools", text)
        self.assertIn("Watch recording", text)
        self.assertIn("Care Response", text)
        self.assertIn("cv-cr-resolve", text)

    def test_camera_station_css_distinct(self):
        from ui_camera_station import CAMERA_STATION_CSS

        self.assertIn("cs-header", CAMERA_STATION_CSS)
        self.assertNotIn("cv-patients", CAMERA_STATION_CSS)


class TestEntrySeparation(unittest.TestCase):
    def test_landing_mentions_both_interfaces(self):
        # Import landing without loading YOLO if possible — app import loads models.
        # Keep this light: only check HTML builder via source.
        text = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('/camera-station', text)
        self.assertIn('/caregiver', text)
        self.assertIn("create_server_app", text)
        self.assertIn("def _landing_html", text)
        self.assertIn("Camera Station — {label}", text)
        self.assertIn("_live_patient_label", text)
        # Root opens caregiver Home, not the old lab UI.
        self.assertIn('RedirectResponse(url="/caregiver"', text)
        self.assertIn('def root_to_caregiver', text)
        self.assertIn('"/start"', text)
        # Header title comes from roster, not a hardcoded Ana string.
        cam_ui = (ROOT / "ui_camera_station.py").read_text(encoding="utf-8")
        self.assertIn("camera_station_header_html", cam_ui)
        self.assertNotIn("Ana · Room 01", cam_ui)


if __name__ == "__main__":
    unittest.main()
