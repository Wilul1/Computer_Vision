"""SQLite persistence for monitoring events (Phase 1A).

Event lifecycle: OPEN → ACKNOWLEDGED → RESOLVED
Patient monitoring status is NOT stored here — it is computed from live signals.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

DEFAULT_DB = config.OUTPUTS_DIR / "monitoring" / "events.db"
ACTIVE_STATUSES = ("OPEN", "ACKNOWLEDGED")
EVENT_STATUSES = ("OPEN", "ACKNOWLEDGED", "RESOLVED")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class EventStore:
    """Thread-safe SQLite event store. Snapshot files stay on disk; only paths here."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        event_status TEXT NOT NULL,
                        action_text TEXT,
                        reason_text TEXT,
                        snapshot_path TEXT,
                        acknowledged_at TEXT,
                        resolved_at TEXT,
                        source TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_patient_status "
                    "ON events(patient_id, event_status, event_type)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patient_state (
                        patient_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        activity TEXT,
                        reason TEXT,
                        action TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                # Minimal session log for patient-details (created when a room check finishes).
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT NOT NULL,
                        duration_s REAL NOT NULL,
                        source_type TEXT NOT NULL,
                        source TEXT,
                        final_status TEXT,
                        final_activity TEXT,
                        reason_text TEXT,
                        event_count INTEGER NOT NULL DEFAULT 0,
                        activity_seconds_json TEXT,
                        label_counts_json TEXT,
                        preview_path TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_patient "
                    "ON monitoring_sessions(patient_id, id DESC)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS activity_segments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        patient_id TEXT NOT NULL,
                        activity TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT NOT NULL,
                        duration_s REAL NOT NULL,
                        confidence REAL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_segments_session "
                    "ON activity_segments(session_id, id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_segments_patient "
                    "ON activity_segments(patient_id, started_at)"
                )
                self._ensure_column(conn, "monitoring_sessions", "session_status", "TEXT")
                self._ensure_column(conn, "monitoring_sessions", "room_id", "TEXT")
                self._ensure_column(conn, "events", "session_id", "INTEGER")
                # Caregiver response / case-management fields.
                for col, typedef in (
                    ("handled_by", "TEXT"),
                    ("handle_outcome", "TEXT"),
                    ("handle_note", "TEXT"),
                ):
                    self._ensure_column(conn, "events", col, typedef)
                # P1 session recording metadata (paths relative to project root).
                for col, typedef in (
                    ("recording_path", "TEXT"),
                    ("recording_status", "TEXT"),
                    ("recording_started_at", "TEXT"),
                    ("recording_ended_at", "TEXT"),
                    ("recording_duration_s", "REAL"),
                    ("recording_fps", "REAL"),
                    ("recording_width", "INTEGER"),
                    ("recording_height", "INTEGER"),
                    ("recording_codec", "TEXT"),
                    ("recording_bytes", "INTEGER"),
                    ("recording_error", "TEXT"),
                ):
                    self._ensure_column(conn, "monitoring_sessions", col, typedef)
                # Legacy completed video sessions had no status column value.
                conn.execute(
                    """
                    UPDATE monitoring_sessions
                    SET session_status = 'COMPLETED'
                    WHERE session_status IS NULL OR session_status = ''
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, typedef: str) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")

    def has_active_event(self, patient_id: str, event_type: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT 1 FROM events
                    WHERE patient_id = ? AND event_type = ?
                      AND event_status IN ('OPEN', 'ACKNOWLEDGED')
                    LIMIT 1
                    """,
                    (patient_id, event_type),
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    def create_event(
        self,
        *,
        patient_id: str,
        event_type: str,
        action_text: str,
        reason_text: str = "",
        snapshot_path: str | None = None,
        source: str = "demo",
        session_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Create OPEN event unless an active (OPEN/ACK) event of same type exists."""
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT id FROM events
                    WHERE patient_id = ? AND event_type = ?
                      AND event_status IN ('OPEN', 'ACKNOWLEDGED')
                    LIMIT 1
                    """,
                    (patient_id, event_type),
                ).fetchone()
                if existing is not None:
                    return None
                cur = conn.execute(
                    """
                    INSERT INTO events (
                        patient_id, event_type, created_at, event_status,
                        action_text, reason_text, snapshot_path, source, session_id
                    ) VALUES (?, ?, ?, 'OPEN', ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        event_type,
                        _utc_now(),
                        action_text,
                        reason_text,
                        snapshot_path,
                        source,
                        session_id,
                    ),
                )
                conn.commit()
                return self._get(conn, int(cur.lastrowid))
            finally:
                conn.close()

    def acknowledge(
        self,
        event_id: int,
        *,
        handled_by: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        return self._set_status(
            event_id,
            "ACKNOWLEDGED",
            "acknowledged_at",
            handled_by=handled_by,
            note=note,
        )

    def resolve(
        self,
        event_id: int,
        *,
        handled_by: str | None = None,
        outcome: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        return self._set_status(
            event_id,
            "RESOLVED",
            "resolved_at",
            handled_by=handled_by,
            outcome=outcome,
            note=note,
        )

    def _set_status(
        self,
        event_id: int,
        status: str,
        time_col: str,
        *,
        handled_by: str | None = None,
        outcome: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        if status not in EVENT_STATUSES:
            raise ValueError(status)
        with self._lock:
            conn = self._connect()
            try:
                row = self._get(conn, event_id)
                if row is None:
                    return None
                if row["event_status"] == "RESOLVED":
                    return row
                if status == "ACKNOWLEDGED" and row["event_status"] != "OPEN":
                    return row
                sets = [f"event_status = ?", f"{time_col} = ?"]
                args: list[Any] = [status, _utc_now()]
                if handled_by is not None and str(handled_by).strip():
                    sets.append("handled_by = ?")
                    args.append(str(handled_by).strip()[:80])
                if outcome is not None and str(outcome).strip():
                    sets.append("handle_outcome = ?")
                    args.append(str(outcome).strip()[:64])
                if note is not None and str(note).strip():
                    sets.append("handle_note = ?")
                    args.append(str(note).strip()[:500])
                args.append(event_id)
                conn.execute(
                    f"UPDATE events SET {', '.join(sets)} WHERE id = ?",
                    args,
                )
                conn.commit()
                return self._get(conn, event_id)
            finally:
                conn.close()

    def list_recent(self, patient_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if patient_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM events
                        WHERE patient_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (patient_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get(self, event_id: int) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                return self._get(conn, event_id)
            finally:
                conn.close()

    def count_open(self, patient_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS n FROM events
                    WHERE patient_id = ? AND event_status IN ('OPEN', 'ACKNOWLEDGED')
                    """,
                    (patient_id,),
                ).fetchone()
                return int(row["n"] if row else 0)
            finally:
                conn.close()

    def list_active(self, patient_id: str, limit: int = 40) -> list[dict[str, Any]]:
        """OPEN + ACKNOWLEDGED events for one patient (newest first)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE patient_id = ? AND event_status IN ('OPEN', 'ACKNOWLEDGED')
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (patient_id, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def list_for_patients(self, patient_ids: list[str], limit: int = 40) -> list[dict[str, Any]]:
        if not patient_ids:
            return []
        placeholders = ",".join("?" * len(patient_ids))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT * FROM events
                    WHERE patient_id IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (*patient_ids, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def upsert_patient_state(
        self,
        patient_id: str,
        *,
        status: str,
        activity: str = "",
        reason: str = "",
        action: str = "",
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO patient_state (patient_id, status, activity, reason, action, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        status = excluded.status,
                        activity = excluded.activity,
                        reason = excluded.reason,
                        action = excluded.action,
                        updated_at = excluded.updated_at
                    """,
                    (patient_id, status, activity, reason, action, _utc_now()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_patient_state(self, patient_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM patient_state WHERE patient_id = ?",
                    (patient_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def create_session(
        self,
        *,
        patient_id: str,
        started_at: str,
        ended_at: str,
        duration_s: float,
        source_type: str,
        source: str = "",
        final_status: str = "",
        final_activity: str = "",
        reason_text: str = "",
        event_count: int = 0,
        activity_seconds: dict[str, float] | None = None,
        label_counts: dict[str, int] | None = None,
        preview_path: str | None = None,
        session_status: str = "COMPLETED",
        room_id: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO monitoring_sessions (
                        patient_id, started_at, ended_at, duration_s, source_type, source,
                        final_status, final_activity, reason_text, event_count,
                        activity_seconds_json, label_counts_json, preview_path,
                        session_status, room_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        started_at,
                        ended_at,
                        float(duration_s),
                        source_type,
                        source,
                        final_status,
                        final_activity,
                        reason_text,
                        int(event_count),
                        json.dumps(activity_seconds or {}),
                        json.dumps(label_counts or {}),
                        preview_path,
                        session_status,
                        room_id,
                    ),
                )
                conn.commit()
                return self._get_session(conn, int(cur.lastrowid))
            finally:
                conn.close()

    def get_active_session(self, patient_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM monitoring_sessions
                    WHERE patient_id = ? AND session_status = 'ACTIVE'
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (patient_id,),
                ).fetchone()
                return self._session_row(row) if row else None
            finally:
                conn.close()

    def finalize_session(
        self,
        session_id: int,
        *,
        ended_at: str,
        duration_s: float,
        session_status: str = "COMPLETED",
        final_status: str = "",
        final_activity: str = "",
        reason_text: str = "",
        event_count: int = 0,
        activity_seconds: dict[str, float] | None = None,
        label_counts: dict[str, int] | None = None,
        preview_path: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE monitoring_sessions SET
                        ended_at = ?,
                        duration_s = ?,
                        session_status = ?,
                        final_status = ?,
                        final_activity = ?,
                        reason_text = ?,
                        event_count = ?,
                        activity_seconds_json = ?,
                        label_counts_json = COALESCE(?, label_counts_json),
                        preview_path = COALESCE(?, preview_path)
                    WHERE id = ?
                    """,
                    (
                        ended_at,
                        float(duration_s),
                        session_status,
                        final_status,
                        final_activity,
                        reason_text,
                        int(event_count),
                        json.dumps(activity_seconds or {}),
                        json.dumps(label_counts) if label_counts is not None else None,
                        preview_path,
                        session_id,
                    ),
                )
                conn.commit()
                return self._get_session(conn, session_id)
            finally:
                conn.close()

    def interrupt_stale_sessions(self) -> int:
        """Mark leftover ACTIVE sessions INTERRUPTED (e.g. after app restart).

        Unfinished recordings (RECORDING / FINALIZING / empty) become INCOMPLETE
        and are never treated as playable READY files.
        """
        with self._lock:
            conn = self._connect()
            try:
                now = _utc_now()
                cur = conn.execute(
                    """
                    UPDATE monitoring_sessions
                    SET session_status = 'INTERRUPTED',
                        ended_at = CASE
                            WHEN ended_at IS NULL OR ended_at = '' THEN ?
                            ELSE ended_at
                        END,
                        reason_text = CASE
                            WHEN reason_text IS NULL OR reason_text = ''
                            THEN 'Session interrupted (application stopped)'
                            ELSE reason_text
                        END,
                        recording_status = CASE
                            WHEN recording_status IN ('RECORDING', 'FINALIZING')
                              OR recording_status IS NULL
                              OR recording_status = ''
                              OR recording_status = 'NONE'
                            THEN 'INCOMPLETE'
                            ELSE recording_status
                        END,
                        recording_ended_at = CASE
                            WHEN recording_status IN ('RECORDING', 'FINALIZING')
                              OR recording_status IS NULL
                              OR recording_status = ''
                              OR recording_status = 'NONE'
                            THEN ?
                            ELSE recording_ended_at
                        END,
                        recording_error = CASE
                            WHEN recording_status IN ('RECORDING', 'FINALIZING')
                              OR recording_status IS NULL
                              OR recording_status = ''
                              OR recording_status = 'NONE'
                            THEN COALESCE(
                                NULLIF(recording_error, ''),
                                'Recording left incomplete after application restart'
                            )
                            ELSE recording_error
                        END
                    WHERE session_status = 'ACTIVE'
                    """,
                    (now, now),
                )
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                conn.close()

    def update_session_recording(
        self,
        session_id: int,
        *,
        recording_path: str | None = None,
        recording_status: str | None = None,
        recording_started_at: str | None = None,
        recording_ended_at: str | None = None,
        recording_duration_s: float | None = None,
        recording_fps: float | None = None,
        recording_width: int | None = None,
        recording_height: int | None = None,
        recording_codec: str | None = None,
        recording_bytes: int | None = None,
        recording_error: str | None = None,
    ) -> dict[str, Any] | None:
        """Patch recording metadata on a monitoring session (P1)."""
        fields: list[str] = []
        values: list[Any] = []
        mapping = {
            "recording_path": recording_path,
            "recording_status": recording_status,
            "recording_started_at": recording_started_at,
            "recording_ended_at": recording_ended_at,
            "recording_duration_s": recording_duration_s,
            "recording_fps": recording_fps,
            "recording_width": recording_width,
            "recording_height": recording_height,
            "recording_codec": recording_codec,
            "recording_bytes": recording_bytes,
            "recording_error": recording_error,
        }
        for col, val in mapping.items():
            if val is not None:
                fields.append(f"{col} = ?")
                values.append(val)
        if not fields:
            return self.get_session(session_id)
        values.append(int(session_id))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"UPDATE monitoring_sessions SET {', '.join(fields)} WHERE id = ?",
                    tuple(values),
                )
                conn.commit()
                return self._get_session(conn, int(session_id))
            finally:
                conn.close()

    def add_activity_segment(
        self,
        *,
        session_id: int,
        patient_id: str,
        activity: str,
        started_at: str,
        ended_at: str,
        duration_s: float,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO activity_segments (
                        session_id, patient_id, activity, started_at, ended_at,
                        duration_s, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        patient_id,
                        activity,
                        started_at,
                        ended_at,
                        float(duration_s),
                        confidence,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM activity_segments WHERE id = ?",
                    (int(cur.lastrowid),),
                ).fetchone()
                return dict(row) if row else {}
            finally:
                conn.close()

    def activity_seconds_for_session(self, session_id: int) -> dict[str, float]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT activity, SUM(duration_s) AS secs
                    FROM activity_segments
                    WHERE session_id = ?
                    GROUP BY activity
                    """,
                    (session_id,),
                ).fetchall()
                return {str(r["activity"]): float(r["secs"] or 0) for r in rows}
            finally:
                conn.close()

    def activity_seconds_today(self, patient_id: str, day_prefix: str | None = None) -> dict[str, float]:
        prefix = day_prefix or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT activity, SUM(duration_s) AS secs
                    FROM activity_segments
                    WHERE patient_id = ? AND started_at LIKE ?
                    GROUP BY activity
                    """,
                    (patient_id, f"{prefix}%"),
                ).fetchall()
                out = {str(r["activity"]): float(r["secs"] or 0) for r in rows}
                # Also fold legacy session JSON totals when no segments exist.
                if not out:
                    sess = conn.execute(
                        """
                        SELECT activity_seconds_json FROM monitoring_sessions
                        WHERE patient_id = ? AND started_at LIKE ?
                        """,
                        (patient_id, f"{prefix}%"),
                    ).fetchall()
                    for s in sess:
                        try:
                            data = json.loads(s["activity_seconds_json"] or "{}")
                        except json.JSONDecodeError:
                            data = {}
                        for k, v in data.items():
                            out[str(k)] = out.get(str(k), 0.0) + float(v or 0)
                return out
            finally:
                conn.close()

    def list_sessions(self, patient_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM monitoring_sessions
                    WHERE patient_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (patient_id, limit),
                ).fetchall()
                return [self._session_row(r) for r in rows]
            finally:
                conn.close()

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                return self._get_session(conn, session_id)
            finally:
                conn.close()

    def count_events_by_status(self, patient_id: str) -> dict[str, int]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT event_status, COUNT(*) AS n FROM events
                    WHERE patient_id = ?
                    GROUP BY event_status
                    """,
                    (patient_id,),
                ).fetchall()
                out = {"OPEN": 0, "ACKNOWLEDGED": 0, "RESOLVED": 0}
                for r in rows:
                    out[str(r["event_status"])] = int(r["n"])
                return out
            finally:
                conn.close()

    @staticmethod
    def _get(conn: sqlite3.Connection, event_id: int) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    @classmethod
    def _get_session(cls, conn: sqlite3.Connection, session_id: int) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM monitoring_sessions WHERE id = ?", (session_id,),
        ).fetchone()
        return cls._session_row(row) if row else None

    def wipe_all_monitoring_data(self) -> None:
        """Clear events, sessions, segments, and patient_state (keep schema)."""
        with self._lock:
            conn = self._connect()
            try:
                for table in (
                    "activity_segments",
                    "monitoring_sessions",
                    "events",
                    "patient_state",
                ):
                    conn.execute(f"DELETE FROM {table}")
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _session_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("activity_seconds_json", "label_counts_json"):
            raw = d.get(key) or "{}"
            try:
                d[key.replace("_json", "")] = json.loads(raw)
            except json.JSONDecodeError:
                d[key.replace("_json", "")] = {}
        return d
