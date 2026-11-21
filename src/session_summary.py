"""Build a short session summary for video Analyze and live Stop."""
from collections import Counter

from .classifier import IGNORED


def session_blurb(
    labels: list[str],
    *,
    duration_s: float,
    fall_events: int = 0,
    inactive_events: int = 0,
    inactive_seconds: float = 0.0,
    zone_out_frames: int = 0,
    zone_partial_frames: int = 0,
    fps: float = 30.0,
) -> str:
    """One-line markdown-friendly summary of a monitored session."""
    if not labels:
        return "No frames in this session."
    voting = [l for l in labels if l not in IGNORED]
    if voting:
        main, n = Counter(voting).most_common(1)[0]
        share = n / len(voting)
        main_bit = f"Mostly **{main}** ({share:.0%})"
    else:
        main_bit = "No clear activity label"

    bits = [main_bit, f"{duration_s:.0f}s monitored"]
    if fall_events:
        bits.append(f"{fall_events} fall alert{'s' if fall_events != 1 else ''}")
    if inactive_events:
        bits.append(f"{inactive_events} inactive alert{'s' if inactive_events != 1 else ''}")
    if inactive_seconds > 0:
        bits.append(f"{inactive_seconds:.0f}s inactive")
    total = max(len(labels), 1)
    out_s = zone_out_frames / max(fps, 1e-3)
    part_s = zone_partial_frames / max(fps, 1e-3)
    if zone_out_frames:
        bits.append(f"{out_s:.0f}s out of view ({zone_out_frames / total:.0%})")
    if zone_partial_frames:
        bits.append(f"{part_s:.0f}s partial body ({zone_partial_frames / total:.0%})")
    return " · ".join(bits)


def session_details(
    labels: list[str],
    *,
    duration_s: float,
    fall_events: int = 0,
    inactive_events: int = 0,
    inactive_seconds: float = 0.0,
    zone_out_frames: int = 0,
    zone_partial_frames: int = 0,
    fps: float = 30.0,
    extra: dict | None = None,
) -> dict:
    blurb = session_blurb(
        labels,
        duration_s=duration_s,
        fall_events=fall_events,
        inactive_events=inactive_events,
        inactive_seconds=inactive_seconds,
        zone_out_frames=zone_out_frames,
        zone_partial_frames=zone_partial_frames,
        fps=fps,
    )
    out = {
        "session summary": blurb.replace("**", ""),
        "fall alerts (extension)": fall_events,
        "inactive alerts (extension)": inactive_events,
        "inactive seconds (approx)": round(inactive_seconds, 1),
        "out-of-view frames": zone_out_frames,
        "partial-body frames": zone_partial_frames,
        "session length (s)": round(duration_s, 1),
    }
    if extra:
        out.update(extra)
    return out
