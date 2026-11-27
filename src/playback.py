"""Safe session recording playback helpers (CareView P2).

Resolves recordings only from EventStore metadata. Paths must stay under
outputs/recordings/. This module does NOT authenticate callers — the patient_id
query check is a consistency gate only (same LAN-demo trust model as /api/live).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from . import config
from .care_demo import get_patient
from .event_store import EventStore

RECORDINGS_ROOT = (config.OUTPUTS_DIR / "recordings").resolve()
CHUNK_SIZE = 128 * 1024  # 128 KiB — within 64–256 KB guidance
PLAYABLE_STATUS = "READY"


class PlaybackError(Exception):
    """Structured playback failure (maps to HTTP in app.py)."""

    def __init__(self, code: str, message: str, *, http_status: int = 404):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = int(http_status)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.code, "message": self.message}


@dataclass(frozen=True)
class PlayableRecording:
    session_id: int
    patient_id: str
    path: Path
    size: int
    session: dict[str, Any]


def recordings_root() -> Path:
    return RECORDINGS_ROOT


def _resolve_under_recordings(rel_or_abs: str) -> Path:
    """Resolve a DB path and require it to stay under outputs/recordings/."""
    raw = (rel_or_abs or "").strip()
    if not raw:
        raise PlaybackError(
            "recording_file_missing",
            "Recording path is empty",
            http_status=404,
        )
    # Disallow obvious traversal tokens before join.
    if ".." in Path(raw).parts or raw.startswith(("~", "\\\\", "//")):
        raise PlaybackError(
            "recording_unavailable",
            "Recording path is not available",
            http_status=404,
        )
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = config.ROOT / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise PlaybackError(
            "recording_unavailable",
            f"Recording path could not be resolved ({exc})",
            http_status=404,
        ) from exc
    root = recordings_root()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PlaybackError(
            "recording_unavailable",
            "Recording path is outside the recordings directory",
            http_status=404,
        ) from exc
    return resolved


def resolve_playable(
    store: EventStore,
    session_id: int,
    *,
    patient_id: str,
    require_ready: bool = True,
) -> PlayableRecording:
    """Load session + validate patient_id + READY file under recordings root."""
    pid = (patient_id or "").strip()
    if not pid:
        raise PlaybackError(
            "patient_id_required",
            "Query parameter patient_id is required",
            http_status=400,
        )
    if get_patient(pid) is None:
        raise PlaybackError(
            "patient_not_found",
            "Unknown patient_id",
            http_status=404,
        )
    try:
        sid = int(session_id)
    except (TypeError, ValueError) as exc:
        raise PlaybackError(
            "session_not_found",
            "Invalid session_id",
            http_status=404,
        ) from exc

    row = store.get_session(sid)
    if row is None:
        raise PlaybackError(
            "session_not_found",
            f"Session {sid} not found",
            http_status=404,
        )
    if str(row.get("patient_id") or "") != pid:
        raise PlaybackError(
            "patient_mismatch",
            "patient_id does not match this session",
            http_status=404,
        )

    status = (row.get("recording_status") or "NONE").upper()
    if require_ready and status != PLAYABLE_STATUS:
        raise PlaybackError(
            "recording_not_ready",
            f"Recording status is {status}, not READY",
            http_status=409,
        )

    path = _resolve_under_recordings(str(row.get("recording_path") or ""))
    if not path.is_file():
        raise PlaybackError(
            "recording_file_missing",
            "Recording file is missing",
            http_status=404,
        )
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise PlaybackError(
            "recording_read_error",
            f"Could not stat recording ({exc})",
            http_status=500,
        ) from exc
    if size <= 0:
        raise PlaybackError(
            "recording_file_missing",
            "Recording file is empty",
            http_status=404,
        )
    return PlayableRecording(
        session_id=sid,
        patient_id=pid,
        path=path,
        size=size,
        session=row,
    )


def session_meta_payload(
    store: EventStore,
    session_id: int,
    *,
    patient_id: str,
) -> dict[str, Any]:
    """JSON meta for a session. playback_available only when READY + file OK."""
    pid = (patient_id or "").strip()
    if not pid:
        raise PlaybackError(
            "patient_id_required",
            "Query parameter patient_id is required",
            http_status=400,
        )
    patient = get_patient(pid)
    if patient is None:
        raise PlaybackError(
            "patient_not_found",
            "Unknown patient_id",
            http_status=404,
        )
    try:
        sid = int(session_id)
    except (TypeError, ValueError) as exc:
        raise PlaybackError(
            "session_not_found",
            "Invalid session_id",
            http_status=404,
        ) from exc
    row = store.get_session(sid)
    if row is None:
        raise PlaybackError(
            "session_not_found",
            f"Session {sid} not found",
            http_status=404,
        )
    if str(row.get("patient_id") or "") != pid:
        raise PlaybackError(
            "patient_mismatch",
            "patient_id does not match this session",
            http_status=404,
        )

    status = (row.get("recording_status") or "NONE").upper()
    playback_available = False
    video_url = None
    if status == PLAYABLE_STATUS:
        try:
            playable = resolve_playable(store, sid, patient_id=pid, require_ready=True)
            playback_available = True
            video_url = f"/api/sessions/{sid}/video?patient_id={pid}"
            _ = playable  # path validated; never returned
        except PlaybackError:
            playback_available = False
            video_url = None

    payload: dict[str, Any] = {
        "ok": True,
        "session_id": sid,
        "patient_id": pid,
        "patient_name": patient.get("name") or "",
        "room_id": row.get("room_id") or patient.get("room") or "",
        "session_status": row.get("session_status") or "",
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "duration_s": float(row.get("duration_s") or 0),
        "recording_status": status,
        "playback_available": playback_available,
        "video_url": video_url,
    }
    if status == PLAYABLE_STATUS or row.get("recording_duration_s") is not None:
        payload["recording_duration_s"] = row.get("recording_duration_s")
        payload["recording_fps"] = row.get("recording_fps")
        payload["recording_width"] = row.get("recording_width")
        payload["recording_height"] = row.get("recording_height")
        payload["recording_codec"] = row.get("recording_codec")
        payload["recording_bytes"] = row.get("recording_bytes")
    if status not in (PLAYABLE_STATUS, "NONE", "") and row.get("recording_error"):
        payload["recording_error"] = row.get("recording_error")
    if playback_available:
        payload["content_type"] = "video/mp4"
    # Intentionally omit recording_path / absolute paths.
    return payload


def parse_byte_range(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    """Parse a single bytes Range. Returns (start, end) inclusive, or None if absent.

    Raises PlaybackError(invalid_range, http_status=416) for unsatisfiable ranges.
    """
    if not range_header:
        return None
    header = range_header.strip()
    if not header.lower().startswith("bytes="):
        raise PlaybackError(
            "invalid_range",
            "Only bytes ranges are supported",
            http_status=416,
        )
    spec = header.split("=", 1)[1].strip()
    # Support a single range only (browsers typically send one).
    if "," in spec:
        spec = spec.split(",", 1)[0].strip()
    if "-" not in spec:
        raise PlaybackError("invalid_range", "Malformed Range header", http_status=416)
    start_s, end_s = spec.split("-", 1)
    size = int(file_size)
    if size <= 0:
        raise PlaybackError("invalid_range", "Empty file", http_status=416)
    try:
        if start_s == "" and end_s != "":
            # suffix: last N bytes
            suffix = int(end_s)
            if suffix <= 0:
                raise ValueError("suffix")
            start = max(0, size - suffix)
            end = size - 1
        elif start_s != "" and end_s == "":
            start = int(start_s)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s)
    except ValueError as exc:
        raise PlaybackError("invalid_range", "Malformed Range header", http_status=416) from exc

    if start < 0 or end < start or start >= size:
        raise PlaybackError(
            "invalid_range",
            f"Requested range not satisfiable for size {size}",
            http_status=416,
        )
    end = min(end, size - 1)
    return start, end


def iter_file_range(
    path: Path,
    start: int,
    end: int,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[bytes]:
    """Yield file bytes [start, end] inclusive; always closes the handle."""
    length = end - start + 1
    remaining = length
    fh: BinaryIO | None = None
    try:
        fh = open(path, "rb")
        fh.seek(start)
        while remaining > 0:
            n = min(chunk_size, remaining)
            data = fh.read(n)
            if not data:
                break
            remaining -= len(data)
            yield data
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def resolve_event_snapshot(
    store: EventStore,
    event_id: int,
    *,
    patient_id: str,
) -> Path:
    """Resolve an event snapshot file under the project tree (patient_id gate)."""
    pid = (patient_id or "").strip()
    if not pid:
        raise PlaybackError(
            "patient_id_required",
            "patient_id query parameter is required.",
            http_status=400,
        )
    row = store.get(int(event_id))
    if row is None:
        raise PlaybackError("not_found", "Event not found.", http_status=404)
    if str(row.get("patient_id") or "") != pid:
        raise PlaybackError(
            "patient_mismatch",
            "Event does not belong to that patient.",
            http_status=404,
        )
    rel = str(row.get("snapshot_path") or "").strip()
    if not rel:
        raise PlaybackError("no_snapshot", "No snapshot on file for this event.", http_status=404)
    path = Path(rel)
    if not path.is_absolute():
        path = (config.ROOT / path).resolve()
    else:
        path = path.resolve()
    root = config.ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PlaybackError(
            "path_escape",
            "Snapshot path is outside the project.",
            http_status=404,
        ) from exc
    if not path.is_file():
        raise PlaybackError("missing_file", "Snapshot file is missing.", http_status=404)
    return path


def event_snapshot_url(event_id: int, patient_id: str) -> str:
    """Public URL for an event snapshot (same trust model as session video)."""
    return f"/api/events/{int(event_id)}/snapshot?patient_id={patient_id}"


def register_playback_routes(api: Any, store: EventStore) -> None:
    """Attach /api/sessions/{id}/meta|/video and /api/events/{id}/snapshot."""
    from fastapi import Header, Query
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    def _error_response(exc: PlaybackError) -> JSONResponse:
        headers = {}
        if exc.http_status == 416:
            headers["Content-Range"] = "bytes */*"
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_dict(),
            headers=headers,
        )

    @api.get("/api/sessions/{session_id}/meta")
    def session_recording_meta(
        session_id: int,
        patient_id: str = Query(..., description="Must match the session patient (not auth)"),
    ):
        try:
            return session_meta_payload(store, session_id, patient_id=patient_id)
        except PlaybackError as exc:
            return _error_response(exc)

    @api.get("/api/sessions/{session_id}/video")
    def session_recording_video(
        session_id: int,
        patient_id: str = Query(..., description="Must match the session patient (not auth)"),
        range_header: str | None = Header(default=None, alias="Range"),
    ):
        try:
            playable = resolve_playable(
                store, session_id, patient_id=patient_id, require_ready=True,
            )
        except PlaybackError as exc:
            return _error_response(exc)

        size = playable.size
        try:
            byte_range = parse_byte_range(range_header, size)
        except PlaybackError as exc:
            resp = _error_response(exc)
            if exc.http_status == 416:
                resp.headers["Content-Range"] = f"bytes */{size}"
            return resp

        common = {
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
            "Cache-Control": "private, max-age=0, no-transform",
        }
        if byte_range is None:
            start, end = 0, size - 1
            status = 200
            headers = {**common, "Content-Length": str(size)}
        else:
            start, end = byte_range
            status = 206
            headers = {
                **common,
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{size}",
            }

        def _stream():
            try:
                yield from iter_file_range(playable.path, start, end)
            except Exception:
                return

        return StreamingResponse(
            _stream(),
            status_code=status,
            media_type="video/mp4",
            headers=headers,
        )

    @api.get("/api/events/{event_id}/snapshot")
    def event_snapshot_image(
        event_id: int,
        patient_id: str = Query(..., description="Must match the event patient (not auth)"),
    ):
        try:
            path = resolve_event_snapshot(store, event_id, patient_id=patient_id)
        except PlaybackError as exc:
            return _error_response(exc)
        suffix = path.suffix.lower()
        media = "image/jpeg" if suffix in {".jpg", ".jpeg"} else (
            "image/png" if suffix == ".png" else "application/octet-stream"
        )
        return FileResponse(
            path,
            media_type=media,
            headers={"Cache-Control": "private, max-age=60"},
        )
