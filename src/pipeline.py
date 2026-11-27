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

import cv2
import numpy as np

from . import config
from .camera_motion import CameraMotion
from .classifier import NO_PERSON, UNKNOWN, TemporalFilter, classify_blob, classify_pose
from .fall_alert import FallAlertMonitor
from .features import MotionFeatures, pose_plausible, posture_features
from .inactivity_alert import InactivityAlertMonitor
from .motion import BlobFeatures, MotionDetector, MotionEventCounter
from .params import Params
from .pose import PoseEstimator, pose_on_crop
from .preprocess import preprocess, resize_max_side
from .visualize import draw_result
from .zone_alert import ZoneAlertMonitor

VIS = config.LANDMARK_MIN_VISIBILITY


class MonitorPipeline:
    def __init__(
        self,
        params: Params | None = None,
        method: str = "main",
        use_yolo: bool = True,
        enhance: bool = True,
        live: bool = False,
    ):
        self.params = params or Params.load()
        self.method = method
        self.enhance = enhance
        # live=True (webcam only): skip MOG2 + camera optical flow. Same YOLO/pose/rules.
        # Video Analyze / evaluate keep live=False so reported FPS/accuracy stay unchanged.
        self.live = live
        # Live: much smaller frames + no CLAHE → cut end-to-end lag on CPU laptops.
        self.max_side = 320 if live else config.PROCESS_MAX_SIDE
        if live:
            self.enhance = False
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
        # Live thresholds slightly softer: sparse FPS still needs to catch a short drop.
        self.fall_alert = (
            FallAlertMonitor(
                window_sec=1.4,
                drop_speed_min=0.85,
                torso_rise_min=28.0,
                hold_sec=3.0,
            )
            if live
            else FallAlertMonitor()
        )
        self.zone_alert = ZoneAlertMonitor()
        self.inactivity = InactivityAlertMonitor()
        self.filter_main = TemporalFilter(self.params)
        self.filter_base = TemporalFilter(self.params, "baseline")
        self.frame_idx = 0
        self.people = []
        self.fps = 0.0

    # ---- stage 1 -------------------------------------------------------------
    def preprocess(self, frame) -> dict:
        if self.live:
            # Skip gray/blur motion prep — live infer never uses MOG2.
            resized = resize_max_side(frame, self.max_side)
            return {
                "bgr": resized,
                "rgb": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
                "motion": None,
            }
        return preprocess(frame, self.enhance, max_side=self.max_side)

    # ---- stage 2 -------------------------------------------------------------
    def infer(self, x: dict, t: float) -> dict:
        if self.live:
            m = {"mask": None, "boxes": [], "ratio": 0.0, "moving": False}
        else:
            m = self.motion(x["motion"])
        # Live: cheaper YOLO less often; pose still runs every frame for fall timing.
        yolo_every = 3 if self.live else config.YOLO_EVERY_N_FRAMES
        if self.detector is not None and self.frame_idx % yolo_every == 0:
            if self.live:
                self.people = self.detector(x["bgr"], imgsz=320)
            else:
                self.people = self.detector(x["bgr"])
        lm, self.pose_source = None, "frame"
        person = self.people[0]["box"] if self.people else None
        small = person is not None and person[3] - person[1] < config.POSE_CROP_BELOW_PX
        # Live always uses full-frame pose (crop path is a latency spike).
        if self.live or not small:
            lm = self.pose(x["rgb"], int(t * 1000))
        if (not self.live) and person is not None and (small or lm is None):
            # Small or missed person: pose on the upscaled YOLO crop.
            lm = pose_on_crop(self.crop_pose, x["rgb"], person, config.POSE_CROP_TARGET_H, config.POSE_CROP_MARGIN)
            self.pose_source = "crop"
        if self.live:
            cam = np.zeros(2, np.float32)
        else:
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

        # Baseline: blob shape + speed (not shown in live UI — skip there).
        blob = None
        if self.live:
            base_label = "standing"
        else:
            blob = self.blobs.update(t, m["boxes"])
            base_label, _ = self.filter_base(classify_blob(blob, p))

        # Main: pose features.
        pose_feats = None
        person_box = raw["people"][0]["box"] if raw["people"] else None
        frame_hw = raw.get("frame_shape")
        fall_feats = None
        if lm is None or not pose_plausible(lm, person_box, frame_hw):
            # No pose, or a hallucinated one (hand/face close-up, fake skeleton).
            # YOLO may still see a person (desk webcam: face only) — that is
            # "unknown" (body incomplete), not an empty room.
            if lm is not None:
                lm = None
                raw = {**raw, "landmarks": None}
            self.speeds.reset()
            main_raw = UNKNOWN if raw["people"] else NO_PERSON
        else:
            f = posture_features(lm)
            f["cam_offset"] = raw["cam_offset"]
            f.update(self.speeds.update(t, f))
            fall_feats = f
            pose_feats = {k: v for k, v in f.items() if k not in ("mid_hip", "ankles_rel", "cam_offset")}
            no_box = self.detector is not None and not raw["people"]
            main_raw = NO_PERSON if (p.require_person_box and no_box) else classify_pose(pose_feats, p)
        main_label, conf = self.filter_main(main_raw)
        fall = self.fall_alert.update(t, main_label, fall_feats)
        n_people = len(raw["people"]) if self.detector is not None else -1
        zone = self.zone_alert.update(t, main_label, n_people)
        person_present = n_people > 0 or main_label not in (NO_PERSON,)
        if n_people == 0:
            person_present = False
        elif n_people < 0:
            person_present = main_label != NO_PERSON
        hip_speed = None if pose_feats is None else pose_feats.get("hip_speed_comp", pose_feats.get("hip_speed"))
        inactive = self.inactivity.update(
            t, main_label,
            hip_speed=hip_speed,
            motion_ratio=float(m["ratio"]),
            person_present=person_present and main_label != UNKNOWN,
        )

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
            "n_people": n_people,
            "pose_source": self.pose_source,
            "fall_alert": fall["fall_alert"],
            "zone_alert": zone["zone_alert"],
            "inactive_alert": inactive["inactive_alert"],
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
            "fall_alert": fall["fall_alert"],
            "fall_events": fall["fall_events"],
            "fall_reason": fall["fall_reason"],
            "zone_alert": zone["zone_alert"],
            "zone_kind": zone["zone_kind"],
            "zone_label": zone["zone_label"],
            "zone_out_frames": zone["zone_out_frames"],
            "zone_partial_frames": zone["zone_partial_frames"],
            "inactive_alert": inactive["inactive_alert"],
            "inactive_events": inactive["inactive_events"],
            "inactive_reason": inactive["inactive_reason"],
            "inactive_seconds": inactive["inactive_seconds"],
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
        raw["frame_shape"] = x["bgr"].shape[:2]
        t2 = time.perf_counter()
        result = self.postprocess(raw, t)
        t3 = time.perf_counter()
        elapsed = t3 - t0
        self.fps = 1 / elapsed if self.fps == 0 else 0.9 * self.fps + 0.1 / elapsed
        result["fps"] = self.fps
        result["live"] = self.live
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
        self.__init__(
            self.params, self.method, use_yolo=False, enhance=self.enhance, live=self.live,
        )
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
