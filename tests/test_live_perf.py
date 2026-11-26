"""Latest-frame cache, JPEG reuse, and stale/LIVE semantics."""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_store import EventStore  # noqa: E402
from src.live_hub import LiveMonitorHub  # noqa: E402


def _rgb(h=240, w=320, val=80):
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestLivePerfCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "t.db")
        self.hub = LiveMonitorHub(self.store, patient_id="p1-ana")

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_frame_cache_bounded_and_seq_advances(self):
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(val=10),
            activity="standing",
            confidence=0.9,
            decision={"status": "SAFE", "reason": "ok"},
        )
        seq1 = self.hub.snapshot()["frame_seq"]
        self.hub.ingest(
            annotated_rgb=_rgb(val=200),
            activity="sitting",
            confidence=0.8,
            decision={"status": "SAFE", "reason": "ok"},
        )
        snap = self.hub.snapshot()
        self.assertEqual(snap["frame_seq"], seq1 + 1)
        self.assertEqual(snap["publish_count"], 2)
        # Only the newest preview is kept (pixel check).
        preview = self.hub.get_preview("p1-ana")
        self.assertIsNotNone(preview)
        self.assertGreaterEqual(int(preview.mean()), 150)

    def test_jpeg_encoded_once_and_reused(self):
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(480, 640, 100),
            activity="standing",
            confidence=0.9,
            decision={"status": "SAFE", "reason": "ok"},
            inference_ms=40.0,
        )
        snap = self.hub.snapshot()
        self.assertGreater(snap["jpeg_bytes"], 500)
        self.assertLess(snap["jpeg_bytes"], 120_000)
        self.assertEqual(snap["jpeg_cache_misses"], 1)
        data1, meta1 = self.hub.get_preview_jpeg("p1-ana")
        self.assertIsNotNone(data1)
        hits_after = self.hub.snapshot()["jpeg_cache_hits"]
        data2, meta2 = self.hub.get_preview_jpeg("p1-ana", since_seq=meta1["seq"])
        self.assertIsNone(data2)
        self.assertTrue(meta2["unchanged"])
        self.assertGreater(self.hub.snapshot()["jpeg_cache_hits"], hits_after)
        # New frame → new encode, previous jpeg replaced.
        self.hub.ingest(
            annotated_rgb=_rgb(480, 640, 40),
            activity="lying",
            confidence=0.7,
            decision={"status": "ATTENTION", "reason": "fall check"},
        )
        data3, meta3 = self.hub.get_preview_jpeg("p1-ana")
        self.assertIsNotNone(data3)
        self.assertNotEqual(meta3["seq"], meta1["seq"])
        self.assertNotEqual(data3, data1)

    def test_stale_not_live_and_stop_invalidates(self):
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(),
            activity="standing",
            confidence=0.9,
            decision={"status": "SAFE", "reason": "ok"},
        )
        self.assertEqual(self.hub.snapshot()["live_state"], "LIVE")
        self.hub._frame_ts = time.time() - 30
        snap = self.hub.snapshot()
        self.assertFalse(snap["has_fresh_frame"])
        self.assertEqual(snap["live_state"], "STALE")
        self.assertIsNone(self.hub.get_preview("p1-ana"))
        data, meta = self.hub.get_preview_jpeg("p1-ana")
        self.assertIsNone(data)
        self.assertTrue(meta["stale"])
        self.hub.stop()
        after = self.hub.snapshot()
        self.assertIsNone(after["session_id"])
        self.assertFalse(after["has_frame"])
        self.assertEqual(after["live_state"], "OFFLINE")
        self.assertIsNone(self.hub.get_preview_jpeg("p1-ana")[0])

    def test_patient_isolation(self):
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(),
            activity="standing",
            confidence=0.9,
            decision={"status": "SAFE", "reason": "ok"},
        )
        self.assertIsNotNone(self.hub.get_preview_jpeg("p1-ana")[0])
        self.assertIsNone(self.hub.get_preview_jpeg("p2-ben")[0])
        self.assertIsNone(self.hub.get_preview("p3-cara"))

    def test_concurrent_readers_writers(self):
        self.hub.ensure_session()
        errors: list[str] = []

        def writer():
            try:
                for i in range(40):
                    self.hub.ingest(
                        annotated_rgb=_rgb(val=(i * 5) % 250),
                        activity="standing",
                        confidence=0.5,
                        decision={"status": "SAFE", "reason": "ok"},
                        inference_ms=12.0,
                    )
            except Exception as exc:  # pragma: no cover
                errors.append(f"writer:{exc}")

        def reader():
            try:
                for _ in range(80):
                    self.hub.get_preview("p1-ana")
                    self.hub.get_preview_jpeg("p1-ana")
                    self.hub.snapshot()
            except Exception as exc:  # pragma: no cover
                errors.append(f"reader:{exc}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(self.hub.snapshot()["frame_seq"], 40)

    def test_preview_resized(self):
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(720, 1280, 90),
            activity="walking",
            confidence=0.6,
            decision={"status": "SAFE", "reason": "ok"},
        )
        preview = self.hub.get_preview("p1-ana")
        full = self.hub.get_full_frame("p1-ana")
        self.assertIsNotNone(preview)
        self.assertIsNotNone(full)
        self.assertLessEqual(preview.shape[1], 640)
        self.assertLessEqual(preview.shape[0], 360)
        self.assertEqual(full.shape[0], 720)
        self.assertEqual(full.shape[1], 1280)

    def test_preview_html_embeds_jpeg_not_cached_url(self):
        """Messenger/WebView-safe: data-URI embed, not a cacheable /api/live URL."""
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(120, 160, 90),
            activity="standing",
            confidence=0.9,
            decision={"status": "SAFE", "reason": "ok"},
        )
        seq = self.hub.snapshot()["frame_seq"]
        html = self.hub.preview_img_html("p1-ana", seq=seq, embed_jpeg=True)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn(f'data-seq="{seq}"', html)
        self.assertNotIn("/api/live/", html)
        url_html = self.hub.preview_img_html("p1-ana", seq=seq, embed_jpeg=False)
        self.assertIn(f"/api/live/p1-ana/frame/{seq}.jpg", url_html)

    def test_html_refreshes_when_seq_advances(self):
        self.hub.ensure_session()
        self.hub.ingest(
            annotated_rgb=_rgb(val=10),
            activity="standing",
            confidence=0.9,
            decision={"status": "SAFE", "reason": "ok"},
        )
        html1 = self.hub.preview_img_html("p1-ana", embed_jpeg=True)
        self.hub.ingest(
            annotated_rgb=_rgb(val=200),
            activity="sitting",
            confidence=0.8,
            decision={"status": "SAFE", "reason": "ok"},
        )
        html2 = self.hub.preview_img_html("p1-ana", embed_jpeg=True)
        self.assertNotEqual(html1, html2)
        self.assertIn('data-seq="', html2)


if __name__ == "__main__":
    unittest.main()
