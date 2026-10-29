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
}


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
        f"Activity: {r['activity'].upper()}  ({r['confidence']:.0%})",
        f"Motion: {'YES' if r['moving'] else 'no'}  {r['motion_ratio']:.1%}   events: {r['motion_events']}",
        f"FPS: {r['fps']:.1f}",
    ]
    cv2.rectangle(out, (0, 0), (330, 22 * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color if i == 0 else (255, 255, 255), 2 if i == 0 else 1)
    return out
