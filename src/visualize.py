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


def draw_result(frame, r: dict, *, show_status: bool = True):
    out = frame.copy()
    h, w = out.shape[:2]
    color = COLORS.get(r["activity"], (255, 255, 255))
    live = bool(r.get("live"))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Live: skip skeleton + alpha blends (big CPU win); keep box + solid banners.
    if live:
        if r.get("people"):
            x1, y1, x2, y2 = r["people"][0]["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        act = r["activity"].upper() if r["activity"] != "unknown" else "UNKNOWN"
        hud = f"{act}  {r['confidence']:.0%}  {r['fps']:.0f} FPS"
        cv2.rectangle(out, (0, 0), (min(w, 280), 28), (0, 0, 0), -1)
        cv2.putText(out, hud[:48], (8, 20), font, 0.5, color, 1)
        msg = bg = fg = None
        if r.get("fall_alert"):
            bg, fg, msg = (0, 0, 220), (255, 255, 255), "FALL ALERT"
            if r.get("fall_reason") and r["fall_reason"] != "alert hold":
                msg = f"FALL ALERT · {r['fall_reason']}"
        elif r.get("inactive_alert"):
            bg, fg, msg = (0, 140, 220), (255, 255, 255), "INACTIVE"
        elif r.get("zone_alert"):
            bg, fg = (0, 180, 220), (20, 20, 20)
            msg = str(r.get("zone_label") or "ZONE")
        if msg is not None:
            cv2.rectangle(out, (0, h - 30), (w, h), bg, -1)
            cv2.putText(out, msg[: max(18, w // 10)], (8, h - 10), font, 0.5, fg, 1)
        return out

    # Middle-ground overlay size: readable on recorded analysis frames.
    scale = max(0.55, min(h / 640.0, 1.0))
    hud_scale = 0.55 * scale
    hud_thick = 1
    line_h = max(18, int(20 * scale))
    pad = max(6, int(7 * scale))
    banner_h = max(30, int(38 * scale))
    banner_scale = 0.58 * scale

    for (x, y, bw, bh) in r["motion_boxes"]:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 220, 255), 1)

    for i, p in enumerate(r["people"]):
        x1, y1, x2, y2 = p["box"]
        c = color if i == 0 else (180, 180, 180)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tag = f"{r['activity']} {p['conf']:.2f}" if i == 0 else f"person {p['conf']:.2f}"
        cv2.putText(out, tag, (x1, max(y1 - 6, 14)), font, 0.45 * scale, c, 1)

    lm = r["landmarks"]
    if lm is not None:
        vis = config.LANDMARK_MIN_VISIBILITY
        for a, b in CONNECTIONS:
            if lm[a, 2] >= vis and lm[b, 2] >= vis:
                cv2.line(out, tuple(map(int, lm[a, :2])), tuple(map(int, lm[b, :2])), (255, 255, 255), 2)
        for x, y, v in lm[11:]:
            if v >= vis:
                cv2.circle(out, (int(x), int(y)), 3, color, -1)

    if show_status:
        act = r["activity"].upper() if r["activity"] != "unknown" else "UNKNOWN"
        lines = [f"{act}  {r['confidence']:.0%}   {r['fps']:.0f} FPS"]
        if not live:
            lines.append(
                f"Motion: {'YES' if r['moving'] else 'no'}  {r['motion_ratio']:.1%}  "
                f"events: {r['motion_events']}"
            )
            if "activity_baseline" in r:
                other = "activity_baseline" if r["activity"] == r.get("activity_main") else "activity_main"
                name = "Baseline" if other == "activity_baseline" else "Main"
                lines.append(f"{name}: {r[other]}")
        extras = []
        if r.get("fall_events"):
            extras.append(f"fall×{r['fall_events']}")
        if r.get("inactive_events"):
            extras.append(f"idle×{r['inactive_events']}")
        if r.get("zone_label"):
            extras.append(r["zone_label"])
        if extras:
            lines.append(" · ".join(extras))

        # Compact top-left HUD (semi-transparent feel via dark strip, limited height).
        text_w = max(cv2.getTextSize(t, font, hud_scale, hud_thick)[0][0] for t in lines) + pad * 2
        box_h = pad * 2 + line_h * len(lines)
        # Cap HUD so it stays a corner chip (~28% max), not a full-frame mask.
        max_h = max(line_h + pad * 2, int(h * 0.28))
        if box_h > max_h:
            lines = lines[: max(1, (max_h - pad * 2) // line_h)]
            box_h = pad * 2 + line_h * len(lines)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (min(text_w, w), box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
        for i, text in enumerate(lines):
            cv2.putText(
                out, text[:72], (pad, pad + line_h * (i + 1) - 4),
                font, hud_scale, color if i == 0 else (240, 240, 240), hud_thick,
            )

    # Thin bottom banner — fall > inactive > zone.
    if r.get("fall_alert"):
        bg, fg, msg = (0, 0, 220), (255, 255, 255), "FALL ALERT"
        if r.get("fall_reason") and r["fall_reason"] != "alert hold":
            msg = f"FALL ALERT · {r['fall_reason']}"
    elif r.get("inactive_alert"):
        bg, fg, msg = (0, 140, 220), (255, 255, 255), "INACTIVE"
        if r.get("inactive_reason") and r["inactive_reason"] != "alert hold":
            msg = f"INACTIVE · {r['inactive_reason']}"
    elif r.get("zone_alert"):
        kind = r.get("zone_label") or "ZONE"
        tip = "leave frame" if r.get("zone_kind") == "out_of_view" else "move back ~2-5 m"
        bg, fg, msg = (0, 180, 220), (20, 20, 20), f"{kind} · {tip}"
    else:
        bg = fg = msg = None

    if msg is not None:
        y0 = h - banner_h
        overlay = out.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), bg, -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        # Keep banner text short so it fits without wrapping into the face.
        max_chars = max(18, int(w / (11 * banner_scale)))
        cv2.putText(
            out, msg[:max_chars], (pad, h - max(6, banner_h // 4)),
            font, banner_scale, fg, 1,
        )
    return out
