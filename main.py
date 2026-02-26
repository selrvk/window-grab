import mediapipe as mp
import cv2
import numpy as np
import mss
import pygetwindow as gw
import ctypes
from flask import Flask, Response

from utils import draw_gestures_on_image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
ctypes.windll.user32.SetProcessDPIAware()

#mediapipe
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

#flask
app = Flask(__name__)

def generate_frames():

    # capture chrome cuz where else would u watch ig live from 
    windows = gw.getWindowsWithTitle("Chrome")
    if not windows:
        print("Chrome window not found.")
        return

    window = windows[0]
    window.activate()

    monitor = {
        "left": window.left,
        "top": window.top,
        "width": window.width,
        "height": window.height
    }

    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(monitor)

            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            recognition_result = recognizer.recognize(mp_image)

            if recognition_result.hand_landmarks:
                annotated_image = draw_gestures_on_image(rgb, recognition_result)
            else:
                annotated_image = rgb

            final_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # encode frame as jpeg (essential for mjpeg)
            _, buffer = cv2.imencode('.jpg', final_bgr_image)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)