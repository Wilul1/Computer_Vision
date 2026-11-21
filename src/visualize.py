"""Overlays: person boxes, motion boxes, skeleton, activity label, FPS."""
import cv2

from . import config
from .pose import CONNECTIONS

COLORS = {
    "standing": (80, 200, 80),
    "sitting": (230, 160, 40),
    "walking": (40, 200, 230),
    "running": (40, 90, 240),
    "lying": (200, 60, 200),
    "no person": (150, 150, 150),
    "unknown": (150, 150, 150),
}
DISPLAY = {"unknown": "UNKNOWN - full body needed (move back ~2-5 m)"}


def draw_result(frame, r: dict):
    out = frame.copy()
    color = COLORS.get(r["activity"], (255, 255, 255))

    for (x, y, w, h) in r["motion_boxes"]:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), 1)

    for i, p in enumerate(r["people"]):
        x1, y1, x2, y2 = p["box"]
        c = color if i == 0 else (180, 180, 180)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tag = f"{r['activity']} {p['conf']:.2f}" if i == 0 else f"person {p['conf']:.2f}"
        cv2.putText(out, tag, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    lm = r["landmarks"]
    if lm is not None:
        vis = config.LANDMARK_MIN_VISIBILITY
        for a, b in CONNECTIONS:
            if lm[a, 2] >= vis and lm[b, 2] >= vis:
                cv2.line(out, tuple(map(int, lm[a, :2])), tuple(map(int, lm[b, :2])), (255, 255, 255), 2)
        for x, y, v in lm[11:]:
            if v >= vis:
                cv2.circle(out, (int(x), int(y)), 3, color, -1)

    lines = [
        f"Activity: {DISPLAY.get(r['activity'], r['activity'].upper())}  ({r['confidence']:.0%})",
        f"Motion: {'YES' if r['moving'] else 'no'}  {r['motion_ratio']:.1%}   events: {r['motion_events']}",
        f"FPS: {r['fps']:.1f}",
    ]
    if "activity_baseline" in r:
        other = "activity_baseline" if r["activity"] == r.get("activity_main") else "activity_main"
        name = "Baseline (MOG2 blob)" if other == "activity_baseline" else "Main (pose)"
        lines.insert(2, f"{name}: {r[other]}")
    if r.get("fall_events"):
        lines.append(f"Fall alerts: {r['fall_events']}")
    if r.get("inactive_events"):
        lines.append(f"Inactive alerts: {r['inactive_events']}")
    if r.get("zone_label"):
        lines.append(f"Zone: {r['zone_label']}")
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] for t in lines) + 16
    cv2.rectangle(out, (0, 0), (width, 22 * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color if i == 0 else (255, 255, 255), 2 if i == 0 else 1)

    # Extension banners (one at a time): fall > inactive > zone.
    h, w = out.shape[:2]
    if r.get("fall_alert"):
        cv2.rectangle(out, (0, h - 54), (w, h), (0, 0, 220), -1)
        msg = "FALL ALERT — sudden drop detected"
        if r.get("fall_reason") and r["fall_reason"] != "alert hold":
            msg = f"FALL ALERT — {r['fall_reason']}"
        cv2.putText(out, msg[:70], (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif r.get("inactive_alert"):
        cv2.rectangle(out, (0, h - 54), (w, h), (0, 140, 220), -1)
        msg = "INACTIVE — little motion"
        if r.get("inactive_reason") and r["inactive_reason"] != "alert hold":
            msg = f"INACTIVE — {r['inactive_reason']}"
        cv2.putText(out, msg[:70], (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif r.get("zone_alert"):
        cv2.rectangle(out, (0, h - 54), (w, h), (0, 180, 220), -1)
        kind = r.get("zone_label") or "ZONE"
        tip = "leave frame or come closer" if r.get("zone_kind") == "out_of_view" else "full body needed (move back ~2-5 m)"
        msg = f"{kind} — {tip}"
        cv2.putText(out, msg[:70], (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    return out
