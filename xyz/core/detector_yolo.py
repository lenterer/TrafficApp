# core/detector_yolo.py
import cv2
try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except Exception:
    YOLO = None
    ULTRALYTICS_OK = False

class YOLODetector:
    def __init__(self, model_path: str):
        if not ULTRALYTICS_OK:
            raise RuntimeError("ultralytics not installed")
        if not model_path:
            raise ValueError("model_path required")
        self.model = YOLO(model_path)
        self.current_counts = {}
        self.total_counts = {}
        self.seen_ids = set()

    def reset(self):
        self.current_counts = {}
        self.total_counts = {}
        self.seen_ids = set()

    def process_frame(self, frame, allowed_classes=None):
        """
        Input: frame BGR
        allowed_classes: list of strings (detector labels) to DRAW/COUNT only; None = all
        Returns: annotated BGR frame
        Side effects: updates self.current_counts and self.total_counts (uses tracker IDs if available)
        """
        self.current_counts = {}

        try:
            # Prefer tracking to get stable IDs
            results = self.model.track(frame, persist=True)
        except Exception:
            # fallback to single-frame detection (no ids)
            results = self.model(frame)

        if not results or len(results) == 0:
            return frame

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return frame

        # Try to extract arrays (works with torch tensors or lists)
        try:
            xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            try:
                xyxy = boxes.xyxy.numpy()
            except Exception:
                xyxy = list(boxes.xyxy)

        try:
            cls_arr = boxes.cls.cpu().numpy()
        except Exception:
            try:
                cls_arr = boxes.cls.numpy()
            except Exception:
                cls_arr = list(boxes.cls)

        ids_arr = None
        if hasattr(boxes, "id") and boxes.id is not None:
            try:
                ids_arr = boxes.id.cpu().numpy()
            except Exception:
                try:
                    ids_arr = boxes.id.numpy()
                except Exception:
                    ids_arr = list(boxes.id)

        names = self.model.names  # mapping id -> label

        annotated = frame.copy()
        n = len(xyxy)
        for i in range(n):
            box = xyxy[i]
            cls_id = int(cls_arr[i]) if len(cls_arr) > i else None
            cls_name = str(names.get(cls_id, str(cls_id)))

            # if filter provided, skip other classes
            if allowed_classes and cls_name not in allowed_classes:
                continue

            # count current
            self.current_counts[cls_name] = self.current_counts.get(cls_name, 0) + 1

            # count unique total if id available
            obj_id = None
            if ids_arr is not None and len(ids_arr) > i:
                try:
                    obj_id = int(ids_arr[i])
                except Exception:
                    obj_id = None

            if obj_id is not None:
                if obj_id not in self.seen_ids:
                    self.seen_ids.add(obj_id)
                    self.total_counts[cls_name] = self.total_counts.get(cls_name, 0) + 1

            # draw box + label
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, cls_name, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        return annotated