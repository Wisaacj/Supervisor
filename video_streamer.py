import cv2

from object_detection import ObjectDetector


current_frame = None
shutdown_flag = False


def generate_frames():
    global current_frame

    while not shutdown_flag:
        if current_frame is not None:
            ret, buffer = cv2.imencode(".jpg", current_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # If no frame is available, yield a placeholder or simply pass
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            
def print_inference_stats(results):
    speed_info = " ".join(
        f"{k.capitalize()}: {v:.2f}ms;" for k,v in results.speed.items())
    # Padding the end of the message with spaces to overwrite any characters
    # remaining from the previous call to `print`.
    print_message = f"\r{speed_info}".ljust(100)

    # Print the prepared message. Using \r to return cursor to the start of the
    # line without newline.
    print(print_message, end="", flush=True)
            
def process_video_stream(stream_url: str):
    global current_frame

    capture = cv2.VideoCapture(stream_url)
    pipeline = ObjectDetector()

    if not capture.isOpened():
        print("Couldn't open video stream.")
        return

    print("Stream opened successfully.")

    while not shutdown_flag:
        # Read a frame in from the stream.
        ret, frame = capture.read()

        if not ret:
            print("Couldn't read a frame. Exiting...")
            break

        # Process frame here.
        current_frame, results = pipeline.detect(frame)
        print_inference_stats(results)

    print("\nReleasing video capture...")
    capture.release()