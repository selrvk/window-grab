# main.py might abandon this but this is just mjpeg format (still works)
#hopefully i remember to push the frontend implementation for mjpeg to the repo lmao
import mediapipe as mp
import cv2
import numpy as np
import mss
import pygetwindow as gw
import ctypes
import threading
import time
from flask import Flask, Response
from waitress import serve

from utils import draw_gestures_on_image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ctypes.windll.user32.SetProcessDPIAware()

base_options = python.BaseOptions(
    model_asset_path='gesture_recognizer.task',
    delegate=python.BaseOptions.Delegate.CPU 
)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

latest_frame = None
latest_result = None
frame_lock = threading.Lock()
result_lock = threading.Lock()

INFERENCE_SCALE = 0.5
TARGET_STREAM_FPS = 60

def get_monitor():
    windows = gw.getWindowsWithTitle("Chrome")
    if not windows:
        raise RuntimeError("Chrome window not found.")
    w = windows[0]
    w.activate()
    return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

def capture_loop(monitor):
    global latest_frame
    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            with frame_lock:
                latest_frame = frame

def inference_loop():
    global latest_result
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            continue

        # run mediapipe on half-res copy for speed
        small = cv2.resize(frame, (0, 0), fx=INFERENCE_SCALE, fy=INFERENCE_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        with result_lock:
            latest_result = result

app = Flask(__name__)

def generate_frames():
    frame_time = 1.0 / TARGET_STREAM_FPS

    while True:
        t0 = time.time()

        with frame_lock:
            frame = latest_frame
        with result_lock:
            result = latest_result

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if result and result.hand_landmarks:
            annotated = draw_gestures_on_image(rgb, result)
        else:
            annotated = rgb

        final = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        elapsed = time.time() - t0
        sleep = frame_time - elapsed
        if sleep > 0:
            time.sleep(sleep)

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    monitor = get_monitor()

    threading.Thread(target=capture_loop, args=(monitor,), daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    print("Streaming at http://0.0.0.0:5000/video")
    serve(app, host="0.0.0.0", port=5000, threads=4)