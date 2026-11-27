"""Monitoring session: status + durable events + snapshots (Phase 1A).

Wraps decide() and EventStore. Creates at most one active OPEN/ACK event per
(event_type, patient) while a condition remains continuous.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from . import config
from .alert_actions import decide, format_decision
from .event_store import EventStore

SNAPSHOT_DIR = config.OUTPUTS_DIR / "alerts" / "snapshots"
# Durable Alert Center events only. Framing (partial_body / out_of_view) stays
# as live overlay feedback — it does not open caregiver alerts.
EVENTABLE = frozenset({
    "possible_fall",
    "inactive",
    "camera_unavailable",
})


class MonitoringSession:
    """Per-stream monitoring bridge used by Gradio live/video handlers."""

    def __init__(
        self,
        *,
        patient_id: str = "demo-patient",
        source: str = "live",
        store: EventStore | None = None,
        session_id: int | None = None,
    ):
        self.patient_id = patient_id
        self.source = source
        self.store = store or EventStore()
        self.session_id = session_id
        self.last_decision: dict = decide(camera_unavailable=False, r={"activity": "standing"})
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        r: dict | None,
        annotated_bgr=None,
        *,
        camera_unavailable: bool = False,
    ) -> dict:
        """Compute status; maybe create a deduped event + snapshot."""
        decision = decide(r, camera_unavailable=camera_unavailable)
        self.last_decision = decision
        created = None
        snap_path = ""
        event_type = decision.get("event_type", "ok")

        if event_type in EVENTABLE:
            if not self.store.has_active_event(self.patient_id, event_type):
                snap_path = self._save_snapshot(event_type, annotated_bgr)
                created = self.store.create_event(
                    patient_id=self.patient_id,
                    event_type=event_type,
                    action_text=decision.get("action", ""),
                    reason_text=decision.get("reason", ""),
                    snapshot_path=snap_path or None,
                    source=self.source,
                    session_id=self.session_id,
                )
            else:
                # Active finding already open (dedupe). If it has no snapshot yet,
                # backfill from this frame so caregivers still get a thumb.
                snap_path = self._backfill_snapshot_if_missing(event_type, annotated_bgr)

        return {
            **decision,
            "patient_id": self.patient_id,
            "activity": None if r is None else r.get("activity"),
            "event_created": created is not None,
            "event": created,
            "snapshot_path": (created or {}).get("snapshot_path") if created else snap_path,
            "panel_markdown": format_decision(
                decision,
                activity=None if r is None else r.get("activity"),
            ),
        }

    def recent_events(self, limit: int = 12) -> list[dict]:
        return self.store.list_recent(self.patient_id, limit=limit)

    def acknowledge(self, event_id: int) -> dict | None:
        """Mark event seen. Does NOT change patient monitoring status."""
        return self.store.acknowledge(event_id)

    def resolve(self, event_id: int) -> dict | None:
        return self.store.resolve(event_id)

    def _backfill_snapshot_if_missing(self, event_type: str, annotated_bgr) -> str:
        """Attach a snapshot to the active event of this type when path is empty."""
        try:
            active = self.store.list_active(self.patient_id)
        except Exception:
            active = []
        target = None
        for e in active:
            if e.get("event_type") == event_type:
                target = e
                break
        if target is None:
            return ""
        if target.get("snapshot_path"):
            return str(target.get("snapshot_path") or "")
        snap_path = self._save_snapshot(event_type, annotated_bgr)
        if not snap_path:
            return ""
        try:
            self.store.set_event_snapshot(int(target["id"]), snap_path)
        except Exception:
            return snap_path
        return snap_path

    def _save_snapshot(self, event_type: str, annotated_bgr) -> str:
        if annotated_bgr is None:
            return ""
        try:
            import numpy as np

            frame = annotated_bgr
            # Accept RGB arrays from callers that forgot BGR (still better than no snap).
            if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
                frame = np.ascontiguousarray(frame)
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"{self.patient_id}_{event_type}_{stamp}.jpg"
            path = SNAPSHOT_DIR / name
            ok = cv2.imwrite(str(path), frame)
            if not ok:
                return ""
            # Store path relative to project root when possible.
            try:
                return str(path.relative_to(config.ROOT)).replace("\\", "/")
            except ValueError:
                return str(path)
        except Exception:
            return ""


def events_table(rows: list[dict]) -> list[list]:
    """Rows for a Gradio Dataframe."""
    out = []
    for e in rows:
        out.append([
            e.get("id"),
            e.get("created_at", ""),
            e.get("event_type", ""),
            e.get("event_status", ""),
            e.get("action_text", ""),
            e.get("snapshot_path") or "",
        ])
    return out


EVENT_TABLE_HEADERS = ["id", "time", "type", "status", "action", "snapshot"]
