import cv2
import threading
import numpy as np
import supervision as sv

from ultralytics import YOLO
from flask_socketio import SocketIO
from flask import Flask, Response, render_template

app = Flask(__name__)
socketio = SocketIO(app)

# Global variable to hold the latest processed frame
current_frame = None

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


def gen_frames():
    global current_frame

    while True:
        if current_frame is not None:
            ret, buffer = cv2.imencode(".jpg", current_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # If no frame is available, yield a placeholder or simply pass
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            

@app.route('/')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask_app():
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)


def process_video_stream():
    global current_frame

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
        # current_frame = pipeline.detect(frame)

        # Break the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tidying up and closing stream.")
            break

    # Clean up: release capture object and close any OpenCV windows.
    capture.release()


if __name__ == "__main__":
    threading.Thread(target=run_flask_app, daemon=True).start()

    # Continue with video processing
    process_video_stream()