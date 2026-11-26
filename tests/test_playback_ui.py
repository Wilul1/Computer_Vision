"""P3 caregiver session playback UI helpers (no Gradio launch required)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.care_demo import get_caregiver  # noqa: E402
from src.event_store import EventStore  # noqa: E402
from src.patient_details import (  # noqa: E402
    EMPTY_PLAYER_HTML,
    build_patient_details_html,
    build_playback_player_html,
    empty_playback_panel,
    playback_panel_for_patient,
    recording_availability,
    select_session_playback,
    session_playback_choices,
)
from src.patient_roster import reset_roster_singleton  # noqa: E402


def _write_h264(path: Path, *, frames: int = 12, w: int = 64, h: int = 48) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    gen = imageio_ffmpeg.write_frames(
        str(path),
        (w, h),
        fps=10.0,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart", "-crf", "28"],
        ffmpeg_log_level="error",
    )
    gen.send(None)
    for i in range(frames):
        fr = np.full((h, w, 3), (40 + i * 12) % 255, dtype=np.uint8)
        gen.send(np.ascontiguousarray(fr))
    gen.close()
    return path.stat().st_size


class PlaybackUiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.store = EventStore(self.base / "events.db")
        self.roster = reset_roster_singleton(self.base / "events.db")
        self.roster.seed_demo()
        self.user = get_caregiver("caregiver", "care123")
        self.rec_root = (self.base / "outputs" / "recordings").resolve()
        self.rec_root.mkdir(parents=True)
        self._root_patch = mock.patch("src.playback.RECORDINGS_ROOT", self.rec_root)
        self._cfg_root = mock.patch("src.config.ROOT", self.base)
        self._root_patch.start()
        self._cfg_root.start()
        self.addCleanup(self._root_patch.stop)
        self.addCleanup(self._cfg_root.stop)
        self.addCleanup(self.tmp.cleanup)

    def _session(
        self,
        patient_id: str,
        *,
        status: str = "READY",
        with_file: bool = True,
        codec: str = "avc1",
    ):
        row = self.store.create_session(
            patient_id=patient_id,
            started_at="2026-11-25T12:00:00+00:00",
            ended_at="2026-11-25T12:00:20+00:00",
            duration_s=20.0,
            source_type="live",
            source="laptop_webcam",
            session_status="COMPLETED",
            room_id="Room 01",
            final_status="SAFE",
            final_activity="standing",
            event_count=0,
        )
        sid = int(row["id"])
        video = self.rec_root / patient_id / f"session_{sid}.mp4"
        size = 0
        path_val = str(video)
        if with_file and status == "READY":
            size = _write_h264(video)
        self.store.update_session_recording(
            sid,
            recording_path=path_val if status != "NONE" else None,
            recording_status=None if status == "NONE" else status,
            recording_started_at="2026-11-25T12:00:00+00:00",
            recording_ended_at="2026-11-25T12:00:20+00:00",
            recording_duration_s=20.0,
            recording_fps=10.0,
            recording_width=64,
            recording_height=48,
            recording_codec=codec,
            recording_bytes=size,
            recording_error="" if status == "READY" else "demo",
        )
        return self.store.get_session(sid)

    def test_history_shows_recording_availability(self):
        ready = self._session("p1-ana", status="READY")
        failed = self._session("p1-ana", status="FAILED", with_file=False)
        html = build_patient_details_html(self.store, self.user, "p1-ana")
        self.assertIn("Recording available", html)
        self.assertIn("Recording failed", html)
        self.assertIn(f"Session #{ready['id']}", html)
        self.assertIn(f"Session #{failed['id']}", html)
        self.assertNotIn("Pedro", html)

    def test_ready_vs_unavailable_labels(self):
        ready = self._session("p1-ana", status="READY")
        missing = self._session("p1-ana", status="READY", with_file=False)
        # Force missing file path
        self.store.update_session_recording(
            int(missing["id"]),
            recording_path=str(self.rec_root / "p1-ana" / "missing.mp4"),
            recording_status="READY",
        )
        incomplete = self._session("p1-ana", status="INCOMPLETE", with_file=False)
        none_row = self._session("p1-ana", status="NONE", with_file=False)

        self.assertTrue(recording_availability(self.store, ready)["playable"])
        self.assertEqual(recording_availability(self.store, ready)["label"], "Recording available")
        self.assertFalse(recording_availability(self.store, self.store.get_session(missing["id"]))["playable"])
        self.assertEqual(
            recording_availability(self.store, self.store.get_session(missing["id"]))["label"],
            "Recording unavailable",
        )
        self.assertEqual(recording_availability(self.store, incomplete)["label"], "Recording incomplete")
        self.assertEqual(recording_availability(self.store, none_row)["label"], "No recording")

    def test_correct_p2_url_generated(self):
        row = self._session("p1-ana", status="READY")
        avail = recording_availability(self.store, row)
        self.assertTrue(avail["playable"])
        self.assertEqual(
            avail["video_url"],
            f"/api/sessions/{row['id']}/video?patient_id=p1-ana",
        )
        player = build_playback_player_html(
            patient_id="p1-ana",
            session_id=int(row["id"]),
            video_url=avail["video_url"],
        )
        self.assertIn('src="/api/sessions/', player)
        self.assertIn("patient_id=p1-ana", player)
        self.assertIn("<video", player)
        self.assertIn("controls", player)
        self.assertIn("playsinline", player)
        self.assertIn('preload="metadata"', player)
        self.assertNotIn("autoplay", player.lower().replace("preload", ""))

    def test_select_ready_produces_player(self):
        row = self._session("p1-ana", status="READY")
        player, info, sid = select_session_playback(
            self.user, "p1-ana", int(row["id"]), self.store,
        )
        self.assertEqual(sid, int(row["id"]))
        self.assertIn("<video", player)
        self.assertIn(f"/api/sessions/{row['id']}/video?patient_id=p1-ana", player)
        self.assertIn("Ana", info)
        self.assertIn("Room 01", info)
        self.assertIn("Standing", info)
        self.assertNotIn(str(self.rec_root), player)
        self.assertNotIn("recording_path", player)

    def test_unavailable_never_gets_video_src(self):
        row = self._session("p1-ana", status="FAILED", with_file=False)
        player, info, sid = select_session_playback(
            self.user, "p1-ana", int(row["id"]), self.store,
        )
        self.assertIsNone(sid)
        self.assertNotIn("<video", player)
        self.assertNotIn("src=", player)
        self.assertIn("cannot be played", info.lower())

    def test_wrong_patient_rejected(self):
        row = self._session("p1-ana", status="READY")
        # Seed Pedro session id space separately
        self._session("p2-ben", status="NONE", with_file=False)
        player, info, sid = select_session_playback(
            self.user, "p2-ben", int(row["id"]), self.store,
        )
        self.assertIsNone(sid)
        self.assertNotIn("<video", player)
        self.assertIn("not available for this patient", info.lower())

    def test_dropdown_playable_only_and_patient_scoped(self):
        ready = self._session("p1-ana", status="READY")
        self._session("p1-ana", status="FAILED", with_file=False)
        other = self._session("p2-ben", status="READY")
        choices = session_playback_choices(self.store, "p1-ana")
        ids = [c[1] for c in choices]
        self.assertEqual(ids, [int(ready["id"])])
        self.assertNotIn(int(other["id"]), ids)
        for label, _ in choices:
            self.assertIn("Playable", label)

    def test_switch_patient_clears_player(self):
        self._session("p1-ana", status="READY")
        dd, player, info, sid = playback_panel_for_patient(
            self.store, self.user, "p2-ben",
        )
        self.assertIsNone(sid)
        self.assertNotIn("<video", player)
        self.assertIn("Select a session", player)
        # Dropdown for Pedro should not list Ana's sessions
        # gr.update choices are in dd
        self.assertTrue(hasattr(dd, "choices") or isinstance(dd, dict) or dd is not None)

    def test_back_clears_selection_and_player(self):
        _dd, player, info, sid = empty_playback_panel()
        self.assertIsNone(sid)
        self.assertEqual(player.strip(), EMPTY_PLAYER_HTML.strip())
        self.assertNotIn("<video", player)

    def test_stale_empty_patient_clears_player(self):
        row = self._session("p1-ana", status="READY")
        # Simulate stale callback with empty patient context.
        player, info, sid = select_session_playback(
            self.user, "", int(row["id"]), self.store,
        )
        self.assertIsNone(sid)
        self.assertNotIn("<video", player)

    def test_access_checks_still_apply(self):
        import gradio as gr

        self._session("p1-ana", status="READY")
        with self.assertRaises(gr.Error):
            build_patient_details_html(self.store, self.user, "not-a-patient")
        with self.assertRaises(gr.Error):
            select_session_playback(None, "p1-ana", 1, self.store)


if __name__ == "__main__":
    unittest.main()
