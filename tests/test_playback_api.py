"""P2 session playback API tests (synthetic videos in temp dirs only)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_store import EventStore  # noqa: E402
from src.patient_roster import reset_roster_singleton  # noqa: E402
from src.playback import (  # noqa: E402
    PlaybackError,
    iter_file_range,
    parse_byte_range,
    register_playback_routes,
    resolve_playable,
    session_meta_payload,
)


def _write_synthetic_mp4(path: Path, *, frames: int = 12, w: int = 64, h: int = 48) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h),
    )
    assert writer.isOpened(), "synthetic VideoWriter failed"
    for i in range(frames):
        frame = np.full((h, w, 3), (i * 17) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    size = path.stat().st_size
    assert size > 0
    return size


class PlaybackApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.rec_root = (self.base / "outputs" / "recordings").resolve()
        self.rec_root.mkdir(parents=True)
        self.store = EventStore(self.base / "events.db")
        self.roster = reset_roster_singleton(self.base / "events.db")
        self.roster.seed_demo()
        self._root_patch = mock.patch("src.playback.RECORDINGS_ROOT", self.rec_root)
        self._root_patch.start()
        self.addCleanup(self._root_patch.stop)
        self.addCleanup(self.tmp.cleanup)

        api = FastAPI()
        register_playback_routes(api, self.store)
        self.client = TestClient(api)

    def _ready_session(self, *, patient_id: str = "p1-ana", status: str = "READY"):
        video = self.rec_root / patient_id / "session_test.mp4"
        size = _write_synthetic_mp4(video)
        row = self.store.create_session(
            patient_id=patient_id,
            started_at="2026-11-25T00:00:00+00:00",
            ended_at="2026-11-25T00:00:10+00:00",
            duration_s=10.0,
            source_type="live",
            source="laptop_webcam",
            session_status="COMPLETED",
            room_id="Room 01",
        )
        sid = int(row["id"])
        # Store absolute path under patched recordings root (production uses rel paths).
        self.store.update_session_recording(
            sid,
            recording_path=str(video),
            recording_status=status,
            recording_started_at="2026-11-25T00:00:00+00:00",
            recording_ended_at="2026-11-25T00:00:10+00:00",
            recording_duration_s=10.0,
            recording_fps=10.0,
            recording_width=64,
            recording_height=48,
            recording_codec="mp4v",
            recording_bytes=size,
            recording_error="",
        )
        return sid, video, size

    def test_ready_meta_and_no_filesystem_paths(self):
        sid, video, size = self._ready_session()
        r = self.client.get(f"/api/sessions/{sid}/meta", params={"patient_id": "p1-ana"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["playback_available"])
        self.assertEqual(data["patient_name"], "Ana")
        self.assertEqual(data["video_url"], f"/api/sessions/{sid}/video?patient_id=p1-ana")
        self.assertEqual(data["recording_status"], "READY")
        self.assertEqual(data["recording_bytes"], size)
        blob = r.text.lower()
        self.assertNotIn(str(video).lower().replace("\\", "\\\\"), blob)
        self.assertNotIn("recording_path", data)
        self.assertNotIn("recording_abs_path", data)
        self.assertNotIn(str(self.rec_root).lower(), blob)

    def test_full_video_response(self):
        sid, video, size = self._ready_session()
        r = self.client.get(f"/api/sessions/{sid}/video", params={"patient_id": "p1-ana"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("content-type", "").split(";")[0], "video/mp4")
        self.assertEqual(r.headers.get("accept-ranges"), "bytes")
        self.assertEqual(int(r.headers.get("content-length")), size)
        self.assertEqual(len(r.content), size)
        self.assertEqual(r.content, video.read_bytes())

    def test_bounded_range_206(self):
        sid, video, size = self._ready_session()
        r = self.client.get(
            f"/api/sessions/{sid}/video",
            params={"patient_id": "p1-ana"},
            headers={"Range": "bytes=0-99"},
        )
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.headers.get("content-range"), f"bytes 0-99/{size}")
        self.assertEqual(int(r.headers.get("content-length")), 100)
        self.assertEqual(len(r.content), 100)
        self.assertEqual(r.content, video.read_bytes()[:100])

    def test_open_ended_range(self):
        sid, video, size = self._ready_session()
        start = max(0, size // 2)
        r = self.client.get(
            f"/api/sessions/{sid}/video",
            params={"patient_id": "p1-ana"},
            headers={"Range": f"bytes={start}-"},
        )
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.headers.get("content-range"), f"bytes {start}-{size - 1}/{size}")
        self.assertEqual(int(r.headers.get("content-length")), size - start)
        self.assertEqual(r.content, video.read_bytes()[start:])

    def test_invalid_range_416(self):
        sid, _video, size = self._ready_session()
        r = self.client.get(
            f"/api/sessions/{sid}/video",
            params={"patient_id": "p1-ana"},
            headers={"Range": f"bytes={size + 10}-{size + 20}"},
        )
        self.assertEqual(r.status_code, 416)
        self.assertIn(f"bytes */{size}", r.headers.get("content-range", ""))
        self.assertEqual(r.json().get("error"), "invalid_range")

    def test_missing_file_404(self):
        sid, video, _size = self._ready_session()
        video.unlink()
        r = self.client.get(f"/api/sessions/{sid}/video", params={"patient_id": "p1-ana"})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json().get("error"), "recording_file_missing")
        meta = self.client.get(f"/api/sessions/{sid}/meta", params={"patient_id": "p1-ana"})
        self.assertEqual(meta.status_code, 200)
        self.assertFalse(meta.json()["playback_available"])

    def test_failed_and_incomplete_409(self):
        for status in ("FAILED", "INCOMPLETE", "RECORDING", "FINALIZING", "NONE"):
            sid, _v, _s = self._ready_session(status=status)
            r = self.client.get(f"/api/sessions/{sid}/video", params={"patient_id": "p1-ana"})
            self.assertEqual(r.status_code, 409, status)
            self.assertEqual(r.json().get("error"), "recording_not_ready")

    def test_path_traversal_rejected(self):
        row = self.store.create_session(
            patient_id="p1-ana",
            started_at="2026-11-25T00:00:00+00:00",
            ended_at="2026-11-25T00:00:01+00:00",
            duration_s=1.0,
            source_type="live",
            session_status="COMPLETED",
        )
        sid = int(row["id"])
        self.store.update_session_recording(
            sid,
            recording_path="../secrets/x.mp4",
            recording_status="READY",
            recording_bytes=1,
        )
        r = self.client.get(f"/api/sessions/{sid}/video", params={"patient_id": "p1-ana"})
        self.assertEqual(r.status_code, 404)
        self.assertIn(r.json().get("error"), ("recording_unavailable", "recording_file_missing"))

    def test_path_outside_recordings_root(self):
        outside = self.base / "outside.mp4"
        _write_synthetic_mp4(outside)
        row = self.store.create_session(
            patient_id="p1-ana",
            started_at="2026-11-25T00:00:00+00:00",
            ended_at="2026-11-25T00:00:01+00:00",
            duration_s=1.0,
            source_type="live",
            session_status="COMPLETED",
        )
        sid = int(row["id"])
        self.store.update_session_recording(
            sid,
            recording_path=str(outside.resolve()),
            recording_status="READY",
            recording_bytes=outside.stat().st_size,
        )
        with self.assertRaises(PlaybackError) as ctx:
            resolve_playable(self.store, sid, patient_id="p1-ana")
        self.assertEqual(ctx.exception.code, "recording_unavailable")
        r = self.client.get(f"/api/sessions/{sid}/video", params={"patient_id": "p1-ana"})
        self.assertEqual(r.status_code, 404)

    def test_patient_mismatch_and_unknown_session(self):
        sid, _v, _s = self._ready_session()
        bad = self.client.get(f"/api/sessions/{sid}/video", params={"patient_id": "p2-ben"})
        self.assertEqual(bad.status_code, 404)
        self.assertEqual(bad.json().get("error"), "patient_mismatch")
        missing = self.client.get("/api/sessions/999999/meta", params={"patient_id": "p1-ana"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json().get("error"), "session_not_found")
        no_pid = self.client.get(f"/api/sessions/{sid}/meta")
        self.assertEqual(no_pid.status_code, 422)  # FastAPI missing required query

    def test_stream_closes_file_handle(self):
        sid, video, size = self._ready_session()
        opened = {"n": 0}
        closed = {"n": 0}
        real_open = open

        def tracking_open(*a, **k):
            fh = real_open(*a, **k)
            opened["n"] += 1
            orig_close = fh.close

            def _close():
                closed["n"] += 1
                return orig_close()

            fh.close = _close  # type: ignore[method-assign]
            return fh

        with mock.patch("src.playback.open", tracking_open):
            list(iter_file_range(video, 0, min(50, size - 1), chunk_size=16))
        self.assertGreaterEqual(opened["n"], 1)
        self.assertEqual(opened["n"], closed["n"])

    def test_parse_byte_range_helpers(self):
        self.assertIsNone(parse_byte_range(None, 100))
        self.assertEqual(parse_byte_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(parse_byte_range("bytes=50-", 100), (50, 99))
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))
        with self.assertRaises(PlaybackError):
            parse_byte_range("bytes=200-300", 100)

    def test_meta_payload_helper_matches_http(self):
        sid, _v, _s = self._ready_session()
        payload = session_meta_payload(self.store, sid, patient_id="p1-ana")
        self.assertTrue(payload["playback_available"])
        self.assertNotIn("recording_path", payload)


if __name__ == "__main__":
    unittest.main()
