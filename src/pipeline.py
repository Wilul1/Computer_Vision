"""The whole system as four stages, each testable on its own:

    frame -> preprocess()  -> resized BGR, RGB, blurred gray           (src/preprocess.py)
          -> infer()       -> MOG2 mask + blobs, YOLO boxes, 33 pose landmarks, camera shift
          -> postprocess() -> features -> rules -> smoothing + persistence -> label
          -> render()      -> annotated frame (boxes, skeleton, label, FPS) + CSV row

Both methods run on every frame: the MAIN method (pose + rules) is the label
shown; the BASELINE (MOG2 blob + rules) is computed alongside so the two can be
compared on exactly the same frames. Each frame also yields a `trace` entry with
the features both rule sets need, which evaluate.py and tune.py replay.
"""
import time

import numpy as np

from . import config
from .camera_motion import CameraMotion
from .classifier import NO_PERSON, TemporalFilter, classify_blob, classify_pose
from .features import MotionFeatures, posture_features
from .motion import BlobFeatures, MotionDetector, MotionEventCounter
from .params import Params
from .pose import PoseEstimator, pose_on_crop
from .preprocess import preprocess
from .visualize import draw_result

VIS = config.LANDMARK_MIN_VISIBILITY


class MonitorPipeline:
    def __init__(self, params: Params | None = None, method: str = "main", use_yolo: bool = True, enhance: bool = True):
        self.params = params or Params.load()
        self.method = method
        self.enhance = enhance
        self.motion = MotionDetector()
        self.events = MotionEventCounter()
        self.blobs = BlobFeatures()
        self.pose = PoseEstimator()
        self.crop_pose = PoseEstimator(video=False)
        self.pose_source = "frame"
        self.camera = CameraMotion()
        self.detector = None
        if use_yolo:
            from .detector import PersonDetector
            self.detector = PersonDetector()
        self.speeds = MotionFeatures()
        self.filter_main = TemporalFilter(self.params)
        self.filter_base = TemporalFilter(self.params, "baseline")
        self.frame_idx = 0
        self.people = []
        self.fps = 0.0

    # ---- stage 1 -------------------------------------------------------------
    def preprocess(self, frame) -> dict:
        return preprocess(frame, self.enhance)

    # ---- stage 2 -------------------------------------------------------------
    def infer(self, x: dict, t: float) -> dict:
        m = self.motion(x["motion"])
        if self.detector is not None and self.frame_idx % config.YOLO_EVERY_N_FRAMES == 0:
            self.people = self.detector(x["bgr"])
        lm, self.pose_source = None, "frame"
        person = self.people[0]["box"] if self.people else None
        small = person is not None and person[3] - person[1] < config.POSE_CROP_BELOW_PX
        if not small:
            lm = self.pose(x["rgb"], int(t * 1000))
        if person is not None and (small or lm is None):
            # Small or missed person: pose on the upscaled YOLO crop.
            lm = pose_on_crop(self.crop_pose, x["rgb"], person, config.POSE_CROP_TARGET_H, config.POSE_CROP_MARGIN)
            self.pose_source = "crop"
        # Mask the person out of the camera-motion estimate: pose box if we have
        # one, else the largest YOLO box.
        box = None
        if lm is not None and (lm[:, 2] >= VIS).sum() >= 4:
            pts = lm[lm[:, 2] >= VIS, :2]
            box = (*pts.min(axis=0), *pts.max(axis=0))
        elif self.people:
            box = self.people[0]["box"]
        cam = self.camera.update(x["motion"], box)
        return {"motion": m, "people": self.people, "landmarks": lm, "cam_offset": cam}

    # ---- stage 3 -------------------------------------------------------------
    def postprocess(self, raw: dict, t: float) -> dict:
        p = self.params
        m, lm = raw["motion"], raw["landmarks"]
        self.events.update(m["moving"])

        # Baseline: blob shape + speed.
        blob = self.blobs.update(t, m["boxes"])
        base_label, _ = self.filter_base(classify_blob(blob, p))

        # Main: pose features.
        pose_feats = None
        if lm is None:
            self.speeds.reset()
            main_raw = NO_PERSON
        else:
            f = posture_features(lm)
            f["cam_offset"] = raw["cam_offset"]
            f.update(self.speeds.update(t, f))
            pose_feats = {k: v for k, v in f.items() if k not in ("mid_hip", "ankles_rel", "cam_offset")}
            no_box = self.detector is not None and not raw["people"]
            main_raw = NO_PERSON if (p.require_person_box and no_box) else classify_pose(pose_feats, p)
        main_label, conf = self.filter_main(main_raw)

        visible = 0 if lm is None else int((lm[:, 2] >= VIS).sum())
        trace = {
            "t": round(t, 4),
            "pose": pose_feats,
            "blob": None if blob is None else {"aspect": blob["aspect"], "speed": blob["speed"]},
            # Diagnostics for the error analysis (why did a frame fail?).
            "visible_landmarks": visible,
            "cam_offset_x": float(raw["cam_offset"][0]),
            "motion_ratio": m["ratio"],
            # -1 = YOLO not running, so "no box" says nothing about the frame.
            "n_people": len(raw["people"]) if self.detector is not None else -1,
            "pose_source": self.pose_source,
        }
        return {
            "time": t,
            "activity": main_label if self.method == "main" else base_label,
            "activity_main": main_label,
            "activity_baseline": base_label,
            "raw_activity": main_raw,
            "confidence": conf,
            "landmarks": lm,
            "people": raw["people"],
            "motion_boxes": m["boxes"],
            "motion_ratio": m["ratio"],
            "moving": m["moving"],
            "motion_events": self.events.events,
            "trace": trace,
            **(pose_feats or {}),
        }

    # ---- stage 4 -------------------------------------------------------------
    def render(self, x: dict, result: dict):
        return draw_result(x["bgr"], result)

    def process(self, frame, t: float) -> dict:
        """Run all four stages on one frame taken at time `t` (seconds)."""
        t0 = time.perf_counter()
        x = self.preprocess(frame)
        t1 = time.perf_counter()
        raw = self.infer(x, t)
        t2 = time.perf_counter()
        result = self.postprocess(raw, t)
        t3 = time.perf_counter()
        elapsed = t3 - t0
        self.fps = 1 / elapsed if self.fps == 0 else 0.9 * self.fps + 0.1 / elapsed
        result["fps"] = self.fps
        result["annotated"] = self.render(x, result)
        t4 = time.perf_counter()
        result["timing_ms"] = {k: round(v * 1000, 2) for k, v in
                               {"preprocess": t1 - t0, "infer": t2 - t1, "postprocess": t3 - t2, "render": t4 - t3}.items()}
        self.frame_idx += 1
        return result

    def reset(self):
        """Fresh state for a new input (keeps the loaded YOLO model)."""
        self.close()
        detector = self.detector
        self.__init__(self.params, self.method, use_yolo=False, enhance=self.enhance)
        self.detector = detector

    def close(self):
        self.pose.close()
        self.crop_pose.close()


def to_jsonable(trace: list[dict]) -> list[dict]:
    """numpy -> plain floats, for caching traces as JSON."""
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    return [conv(f) for f in trace]
