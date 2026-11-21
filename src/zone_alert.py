"""Extension: framing / zone warning (out of view or partial body).

Not an activity class. Uses YOLO + pose status already produced by the pipeline.
"""


class ZoneAlertMonitor:
    """OUT OF VIEW / PARTIAL BODY with a short hold so the banner does not flicker."""

    def __init__(self, hold_sec: float = 0.8):
        self.hold_sec = hold_sec
        self.zone = ""
        self.zone_until = -1.0
        self.out_frames = 0
        self.partial_frames = 0

    def reset(self):
        self.zone = ""
        self.zone_until = -1.0
        self.out_frames = 0
        self.partial_frames = 0

    def update(self, t: float, activity: str, n_people: int) -> dict:
        """n_people: YOLO count, or -1 if detector off."""
        instant = ""
        if n_people == 0 or (n_people < 0 and activity == "no person"):
            instant = "out_of_view"
        elif activity == "unknown":
            instant = "partial_body"
        elif activity == "no person" and n_people > 0:
            instant = "partial_body"
        elif activity == "no person":
            instant = "out_of_view"

        if instant:
            self.zone = instant
            self.zone_until = t + self.hold_sec
            if instant == "out_of_view":
                self.out_frames += 1
            else:
                self.partial_frames += 1
        elif t >= self.zone_until:
            self.zone = ""

        active = bool(self.zone) and t < self.zone_until
        label = ""
        if active:
            label = "OUT OF VIEW" if self.zone == "out_of_view" else "PARTIAL BODY"
        return {
            "zone_alert": active,
            "zone_kind": self.zone if active else "",
            "zone_label": label,
            "zone_out_frames": self.out_frames,
            "zone_partial_frames": self.partial_frames,
        }
