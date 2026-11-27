"""Extension: sudden fall / abnormal drop alert.

Not a sixth activity class (evaluate still uses the five posture labels).
This watches pose history for a fast downward hip move or a sharp torso tilt
into lying, then raises a short-lived FALL ALERT for the UI / demo.

Fits proposal Objective 6 (sudden abnormal movement) that was previously
listed as future work.
"""
from collections import deque


class FallAlertMonitor:
    """Heuristic fall detector on top of existing pose features."""

    def __init__(
        self,
        window_sec: float = 1.2,
        drop_speed_min: float = 1.2,      # body-heights / s downward in the image
        torso_rise_min: float = 35.0,     # degrees of tilt increase inside the window
        cooldown_sec: float = 3.0,        # how long the on-screen alert stays up
        hold_sec: float = 2.5,
    ):
        self.window_sec = window_sec
        self.drop_speed_min = drop_speed_min
        self.torso_rise_min = torso_rise_min
        self.cooldown_sec = cooldown_sec
        self.hold_sec = hold_sec
        self.history = deque()
        self.events = 0
        self.alert_until = -1.0
        self.last_event_t = -1e9

    def reset(self):
        self.history.clear()
        self.events = 0
        self.alert_until = -1.0
        self.last_event_t = -1e9

    def update(self, t: float, activity: str, feats: dict | None) -> dict:
        """Return {fall_alert, fall_events, fall_reason} for this frame."""
        if feats is None or "mid_hip" not in feats or "body_h" not in feats:
            active = t < self.alert_until
            return {
                "fall_alert": active,
                "fall_events": self.events,
                "fall_reason": "alert hold" if active else "",
            }

        hip_y = float(feats["mid_hip"][1])
        torso = float(feats.get("torso_angle") or 0.0)
        body_h = float(max(feats["body_h"], 1e-3))
        self.history.append((t, hip_y, torso, activity))
        while self.history and t - self.history[0][0] > self.window_sec:
            self.history.popleft()

        reason = ""
        if len(self.history) >= 3 and t - self.last_event_t >= self.cooldown_sec:
            t0, y0, torso0, act0 = self.history[0]
            t1, y1, torso1, act1 = self.history[-1]
            dt = max(t1 - t0, 1e-3)
            # Image y grows downward → positive speed = hips moving down the frame.
            window_drop = (y1 - y0) / body_h / dt
            window_tilt = torso1 - torso0
            # Peak step-to-step change catches short falls between sparse live frames
            # (window average alone can dilute a 0.4s drop across 1.2s of history).
            peak_drop = 0.0
            peak_tilt = 0.0
            for i in range(1, len(self.history)):
                ta, ya, toa, _ = self.history[i - 1]
                tb, yb, tob, _ = self.history[i]
                dti = max(tb - ta, 1e-3)
                peak_drop = max(peak_drop, (yb - ya) / body_h / dti)
                peak_tilt = max(peak_tilt, tob - toa)
            drop_speed = max(window_drop, peak_drop)
            torso_rise = max(window_tilt, peak_tilt)
            was_upright = act0 in ("standing", "sitting", "walking", "running")
            # Also accept mid-window upright → end-down (activity may lag one frame).
            mid_upright = any(
                a in ("standing", "sitting", "walking", "running")
                for _, _, _, a in self.history
            )
            now_down = act1 == "lying" or torso1 >= 48.0
            fast_drop = drop_speed >= self.drop_speed_min
            fast_tilt = torso_rise >= self.torso_rise_min

            if (was_upright or mid_upright) and now_down and (fast_drop or fast_tilt):
                self.events += 1
                self.last_event_t = t
                self.alert_until = t + self.hold_sec
                bits = []
                if fast_drop:
                    bits.append(f"hip drop {drop_speed:.1f} bh/s")
                if fast_tilt:
                    bits.append(f"torso +{torso_rise:.0f}°")
                reason = f"{act0} -> lying/tilt ({', '.join(bits)})"

        active = t < self.alert_until
        if active and not reason:
            reason = "alert hold"
        return {
            "fall_alert": active,
            "fall_events": self.events,
            "fall_reason": reason if active else "",
        }
