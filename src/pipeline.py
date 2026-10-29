"""The whole system in one place:
    frame -> preprocess -> motion (MOG2) + person (YOLO) + pose (MediaPipe)
          -> features -> rule-based classifier -> overlay
"""
import time

from . import config
from .classifier import ActivityClassifier
from .features import MotionFeatures, posture_features
from .motion import MotionDetector, MotionEventCounter
from .pose import PoseEstimator
from .preprocess import preprocess
from .visualize import draw_result


class MonitorPipeline:
    def __init__(self, use_yolo: bool = True, enhance: bool = True):
        self.enhance = enhance
        self.motion = MotionDetector()
        self.events = MotionEventCounter()
        self.pose = PoseEstimator()
        self.detector = None
        if use_yolo:
            from .detector import PersonDetector
            self.detector = PersonDetector()
        self.speeds = MotionFeatures()
        self.classifier = ActivityClassifier()
        self.frame_idx = 0
        self.people = []
        self.fps = 0.0

    def process(self, frame, t: float) -> dict:
        """Analyse one frame taken at time `t` (seconds). Returns the result dict;
        the annotated image is in result['annotated']."""
        start = time.perf_counter()
        x = preprocess(frame, self.enhance)

        m = self.motion(x["motion"])
        self.events.update(m["moving"])

        if self.detector is not None and self.frame_idx % config.YOLO_EVERY_N_FRAMES == 0:
            self.people = self.detector(x["bgr"])

        lm = self.pose(x["rgb"], int(t * 1000))
        features = {}
        if lm is None:
            self.speeds.reset()
            activity, raw, conf = "no person", "no person", 1.0
        else:
            features = posture_features(lm)
            features.update(self.speeds.update(t, features))
            activity, raw, conf = self.classifier(features)

        elapsed = time.perf_counter() - start
        self.fps = 1 / elapsed if self.fps == 0 else 0.9 * self.fps + 0.1 / elapsed
        self.frame_idx += 1

        result = {
            "time": t,
            "activity": activity,
            "raw_activity": raw,
            "confidence": conf,
            "landmarks": lm,
            "people": self.people,
            "motion_boxes": m["boxes"],
            "motion_ratio": m["ratio"],
            "moving": m["moving"],
            "motion_events": self.events.events,
            "fps": self.fps,
            **{k: v for k, v in features.items() if k not in ("mid_hip", "ankles_rel")},
        }
        result["annotated"] = draw_result(x["bgr"], result)
        return result

    def close(self):
        self.pose.close()


def run(image, pipeline: MonitorPipeline | None = None, t: float = 0.0):
    """Shape required by the checkpoint guide: one input -> one annotated output."""
    pipeline = pipeline or MonitorPipeline()
    return pipeline.process(image, t)["annotated"]
