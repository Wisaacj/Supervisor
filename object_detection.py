import numpy as np
import supervision as sv

from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_name: str = "models/yolov8n.pt"):
        self.model = YOLO(model_name)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def detect(self, frame: np.ndarray, verbose: bool = False) -> np.ndarray:
        results = self.model(frame, verbose=verbose)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        labels = [
            f"Object #{tracker_id}; P({results.names[class_id].upper()}) = {confidence*100:.2f}%"
            for class_id, tracker_id, confidence
            in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

        return self.trace_annotator.annotate(annotated_frame, detections=detections), results