import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO

STREAM_URL = "https://192.168.0.57:8080/video"


class ObjectDetector:

    def __init__(self, model_name: str = "models/yolov8n.pt"):
        self.model = YOLO(model_name)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def detect(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)[0]
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

        return self.trace_annotator.annotate(annotated_frame, detections=detections)


def main():
    capture = cv2.VideoCapture(STREAM_URL)
    pipeline = ObjectDetector()

    if not capture.isOpened():
        raise RuntimeError("Couldn't open video stream.")

    print("Stream opened successfully.")

    while True:
        # Read a frame in from the stream.
        ret, frame = capture.read()

        if not ret:
            raise RuntimeError("Couldn't read a frame.")

        # Process frame here.
        annotated_frame = pipeline.detect(frame)
        cv2.imshow("Live Stream", annotated_frame)

        # Break the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tidying up and closing stream.")
            break

    # Clean up: release capture object and close any OpenCV windows.
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()