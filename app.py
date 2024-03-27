import cv2
import sys
import signal
import threading
import video_streamer

from flask_socketio import SocketIO
from flask import Flask, Response


app = Flask(__name__)
socketio = SocketIO(app)

STREAM_URL = "https://192.168.0.57:8080/video"


@app.route('/')
def video_feed():
    return Response(
        video_streamer.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def run_flask_app():
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def signal_handler(sig, frame):
    # The actual cleanup will be handled in the threads.
    print("\nShutting down gracefully...")
    video_streamer.shutdown_flag = True

def main():
    # Setup signal handling and threads.
    signal.signal(signal.SIGINT, signal_handler)

    flask_thread = threading.Thread(target=run_flask_app, daemon=True)

    video_process_thread = threading.Thread(
        target=video_streamer.process_video_stream, 
        args=(STREAM_URL,),
        daemon=False,
    )

    flask_thread.start()
    video_process_thread.start()

    # Wait for the video process thread to complete execution before exiting the main thread.
    video_process_thread.join()
    sys.exit(0)


if __name__ == "__main__":
    main()