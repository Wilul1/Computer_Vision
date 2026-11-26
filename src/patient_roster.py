"""Caregiver-managed patient roster (SQLite).

Replaces the hardcoded Ana/Pedro/Maria list. Starts empty unless tests seed it.
Demo login remains in care_demo; patients live here.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from . import config
from .event_store import DEFAULT_DB

# Optional: one patient may own the laptop Camera Station feed.
LIVE_CAMERA_PATIENT_ID = "p1-ana"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return (s[:24] or "patient")


class PatientRoster:
    """Thread-safe patient list shared with EventStore DB file."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patients (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        room TEXT NOT NULL,
                        caregiver_id TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        video_path TEXT,
                        note TEXT,
                        initials TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_patients_caregiver "
                    "ON patients(caregiver_id, name)"
                )
                conn.commit()
            finally:
                conn.close()

    def patients_for(self, caregiver_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM patients
                    WHERE caregiver_id = ?
                    ORDER BY name COLLATE NOCASE, id
                    """,
                    (caregiver_id,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get(self, patient_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM patients WHERE id = ?", (patient_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_live_patient(self, caregiver_id: str = "cg1") -> dict[str, Any] | None:
        """The one patient assigned to the laptop Camera Station, if any."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM patients
                    WHERE caregiver_id = ? AND source_type = 'live'
                    ORDER BY created_at ASC, id
                    LIMIT 1
                    """,
                    (caregiver_id,),
                ).fetchone()
                if row is not None:
                    return dict(row)
                # Fallback: fixed live slot id even if source_type drifted.
                row = conn.execute(
                    "SELECT * FROM patients WHERE id = ?",
                    (LIVE_CAMERA_PATIENT_ID,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def live_camera_label(
        self,
        caregiver_id: str = "cg1",
        *,
        empty: str = "No live patient assigned",
    ) -> str:
        """Display label for Camera Station header (Name · Room)."""
        patient = self.get_live_patient(caregiver_id)
        if not patient:
            return empty
        name = (patient.get("name") or "").strip() or "Patient"
        room = (patient.get("room") or "").strip() or "—"
        return f"{name} · {room}"

    def add_patient(
        self,
        *,
        name: str,
        room: str,
        caregiver_id: str = "cg1",
        source_type: str = "none",
        video_path: str | None = None,
        note: str = "",
        patient_id: str | None = None,
    ) -> dict[str, Any]:
        name = (name or "").strip()
        room = (room or "").strip()
        if not name:
            raise ValueError("Patient name is required.")
        if not room:
            raise ValueError("Room is required.")
        source_type = (source_type or "none").strip().lower()
        if source_type not in ("none", "live", "video"):
            source_type = "none"
        initials = (name[0] if name else "?").upper()
        if patient_id:
            pid = patient_id.strip()
        elif source_type == "live":
            pid = LIVE_CAMERA_PATIENT_ID
        else:
            pid = f"p-{_slug(name)}-{uuid.uuid4().hex[:6]}"

        with self._lock:
            conn = self._connect()
            try:
                if source_type == "live":
                    # Only one live Camera Station patient per caregiver.
                    existing_live = conn.execute(
                        """
                        SELECT id FROM patients
                        WHERE caregiver_id = ? AND source_type = 'live'
                        """,
                        (caregiver_id,),
                    ).fetchone()
                    if existing_live is not None:
                        raise ValueError(
                            "A live-camera patient already exists. "
                            "Remove it before assigning another."
                        )
                else:
                    clash = conn.execute(
                        "SELECT 1 FROM patients WHERE id = ?", (pid,),
                    ).fetchone()
                    if clash is not None:
                        pid = f"p-{_slug(name)}-{uuid.uuid4().hex[:6]}"

                note_text = note or (
                    "Live laptop webcam" if source_type == "live"
                    else "Added by caregiver"
                )
                conn.execute(
                    """
                    INSERT INTO patients (
                        id, name, room, caregiver_id, source_type,
                        video_path, note, initials
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pid, name, room, caregiver_id, source_type,
                        video_path, note_text, initials,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM patients WHERE id = ?", (pid,),
                ).fetchone()
                return dict(row)
            finally:
                conn.close()

    def delete_patient(self, patient_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM patients WHERE id = ?", (patient_id,),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def clear_all(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM patients")
                conn.commit()
            finally:
                conn.close()

    def seed_demo(self) -> None:
        """Test helper: classic Ana / Pedro / Maria roster."""
        self.clear_all()
        _seg = config.ROOT / "data" / "raw" / "_segments"
        _demo = config.ROOT / "tests" / "demo_tomorrow"
        self.add_patient(
            name="Ana", room="Room 01", caregiver_id="cg1",
            source_type="live", note="Live laptop webcam",
            patient_id=LIVE_CAMERA_PATIENT_ID,
        )
        self.add_patient(
            name="Pedro", room="Room 02", caregiver_id="cg1",
            source_type="video",
            video_path=str(_demo / "d01_sit_chair_day.mp4"),
            note="Prerecorded sitting clip",
            patient_id="p2-ben",
        )
        self.add_patient(
            name="Maria", room="Room 03", caregiver_id="cg1",
            source_type="video",
            video_path=str(_seg / "gmdcsa-s3__fall-07.mp4"),
            note="Prerecorded possible-fall clip",
            patient_id="p3-cara",
        )

    def seed_careview_pack(self, caregiver_id: str = "cg1") -> list[dict[str, Any]]:
        """Fresh CareView demo roster — exactly two patients.

        - Alex · Room 01 — live laptop Camera Station (`p1-ana`)
        - Casey · Room 14 — ready fall clip → Run room check for Alerts + playback
        """
        _seg = config.ROOT / "data" / "raw" / "_segments"
        fall = _seg / "gmdcsa-s4__fall-07.mp4"
        if not fall.is_file():
            raise FileNotFoundError(f"Missing fall demo clip: {fall}")

        self.clear_all()
        rows = [
            self.add_patient(
                name="Alex",
                room="Room 01",
                caregiver_id=caregiver_id,
                source_type="live",
                note="Live laptop Camera Station",
                patient_id=LIVE_CAMERA_PATIENT_ID,
            ),
            self.add_patient(
                name="Casey",
                room="Room 14",
                caregiver_id=caregiver_id,
                source_type="video",
                video_path=str(fall.relative_to(config.ROOT)).replace("\\", "/"),
                note="Ready demo — Run room check for Alert Center + playback",
                patient_id="p-casey",
            ),
        ]
        return rows


_ROSTER: PatientRoster | None = None
_ROSTER_LOCK = threading.Lock()


def get_roster(db_path: Path | str | None = None) -> PatientRoster:
    global _ROSTER
    with _ROSTER_LOCK:
        if db_path is not None:
            return PatientRoster(db_path)
        if _ROSTER is None:
            _ROSTER = PatientRoster()
        return _ROSTER


def reset_roster_singleton(db_path: Path | str | None = None) -> PatientRoster:
    """Tests / app boot: replace the process-wide roster."""
    global _ROSTER
    with _ROSTER_LOCK:
        _ROSTER = PatientRoster(db_path)
        return _ROSTER
