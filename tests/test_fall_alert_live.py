"""Fall detector should catch short drops even with sparse live-like samples."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fall_alert import FallAlertMonitor  # noqa: E402


def _feats(hip_y: float, torso: float, body_h: float = 100.0) -> dict:
    return {"mid_hip": (50.0, hip_y), "torso_angle": torso, "body_h": body_h}


class TestFallAlertSparse(unittest.TestCase):
    def test_peak_drop_catches_short_fall(self):
        # Live-like ~4–5 samples over ~0.8s: upright then sharp hip drop into lying.
        mon = FallAlertMonitor(drop_speed_min=0.85, torso_rise_min=28.0, window_sec=1.4)
        times = [0.0, 0.2, 0.4, 0.55, 0.75]
        hips = [40.0, 42.0, 45.0, 95.0, 110.0]  # sharp drop between 0.4→0.55
        torsos = [10.0, 12.0, 15.0, 55.0, 70.0]
        acts = ["standing", "standing", "standing", "lying", "lying"]
        hit = False
        for t, y, tor, act in zip(times, hips, torsos, acts):
            out = mon.update(t, act, _feats(y, tor))
            if out["fall_alert"]:
                hit = True
                self.assertIn("hip drop", out["fall_reason"])
                break
        self.assertTrue(hit, "expected possible-fall on a short hip drop")

    def test_no_alert_while_still_standing(self):
        mon = FallAlertMonitor(drop_speed_min=0.85, torso_rise_min=28.0)
        for i in range(6):
            out = mon.update(i * 0.2, "standing", _feats(40.0 + i, 12.0))
        self.assertFalse(out["fall_alert"])


if __name__ == "__main__":
    unittest.main()
