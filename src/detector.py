"""Person detection with a pretrained YOLO model (COCO class 0 = person)."""
from . import config


class PersonDetector:
    def __init__(self, weights: str = config.YOLO_WEIGHTS):
        from ultralytics import YOLO  # imported here so --no-yolo runs without loading torch

        # Passing the full path makes ultralytics download the weights into models/.
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(config.MODELS_DIR / weights))

    def __call__(self, bgr) -> list[dict]:
        result = self.model.predict(bgr, classes=[0], conf=config.YOLO_CONF, verbose=False)[0]
        people = []
        for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            people.append({"box": (x1, y1, x2, y2), "conf": float(conf)})
        # Largest person first: we track the primary (closest) subject.
        people.sort(key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]), reverse=True)
        return people
