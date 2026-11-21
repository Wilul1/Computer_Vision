"""Extension: inactivity / no-motion alert while a person is still in view.

Person present + very little body motion for several seconds → INACTIVE banner.
"""


class InactivityAlertMonitor:
    """Heuristic stillness detector on pose speed and/or MOG2 motion ratio."""

    def __init__(
        self,
        still_sec: float = 8.0,
        hip_speed_max: float = 0.04,   # body-heights / s
        motion_ratio_max: float = 0.008,
        hold_sec: float = 3.0,
        cooldown_sec: float = 6.0,
    ):
        self.still_sec = still_sec
        self.hip_speed_max = hip_speed_max
        self.motion_ratio_max = motion_ratio_max
        self.hold_sec = hold_sec
        self.cooldown_sec = cooldown_sec
        self.still_since = None
        self.armed = True  # after an alert, need motion before the next count
        self.events = 0
        self.alert_until = -1.0
        self.last_event_t = -1e9
        self.total_inactive_sec = 0.0
        self._last_t = None

    def reset(self):
        self.still_since = None
        self.armed = True
        self.events = 0
        self.alert_until = -1.0
        self.last_event_t = -1e9
        self.total_inactive_sec = 0.0
        self._last_t = None

    def update(
        self,
        t: float,
        activity: str,
        *,
        hip_speed: float | None = None,
        motion_ratio: float = 0.0,
        person_present: bool = False,
    ) -> dict:
        if self._last_t is not None and person_present and activity in (
            "standing", "sitting", "lying",
        ):
            if t < self.alert_until:
                self.total_inactive_sec += max(0.0, t - self._last_t)
        self._last_t = t

        if not person_present or activity in ("no person", "unknown", "walking", "running"):
            self.still_since = None
            if activity in ("walking", "running"):
                self.armed = True
            active = t < self.alert_until
            return self._pack(active, "")

        speed = float(hip_speed) if hip_speed is not None else None
        if speed is not None:
            is_still = speed <= self.hip_speed_max
        else:
            is_still = motion_ratio <= self.motion_ratio_max

        reason = ""
        if is_still:
            if self.still_since is None and self.armed:
                self.still_since = t
            quiet = 0.0 if self.still_since is None else (t - self.still_since)
            if (
                self.armed
                and self.still_since is not None
                and quiet >= self.still_sec
                and t - self.last_event_t >= self.cooldown_sec
            ):
                self.events += 1
                self.last_event_t = t
                self.alert_until = t + self.hold_sec
                self.armed = False
                self.still_since = None
                how = f"hip {speed:.3f} bh/s" if speed is not None else f"motion {motion_ratio:.2%}"
                reason = f"still {quiet:.0f}s ({how})"
        else:
            self.still_since = None
            self.armed = True

        active = t < self.alert_until
        if active and not reason:
            reason = "alert hold"
        return self._pack(active, reason if active else "")

    def _pack(self, active: bool, reason: str) -> dict:
        return {
            "inactive_alert": active,
            "inactive_events": self.events,
            "inactive_reason": reason,
            "inactive_seconds": round(self.total_inactive_sec, 1),
        }
