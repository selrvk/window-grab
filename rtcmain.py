# rtcmain.py
import mediapipe as mp
import cv2
import numpy as np
import mss
import pygetwindow as gw
import ctypes
import threading
import time
import asyncio
import fractions
from collections import deque

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from utils import draw_gestures_on_image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ctypes.windll.user32.SetProcessDPIAware()

# mediapipe (change models here rn i just have gesture and hand_landmark)
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
TARGET_FPS = 30

# smoothing config
HISTORY_SIZE = 5       # how many recent results to consider
MIN_CONFIRMATIONS = 3  # how many must agree before we accept the gesture

# window capture
def get_monitor():
    windows = gw.getWindowsWithTitle("Chrome")
    if not windows:
        raise RuntimeError("Chrome window not found.")
    w = windows[0]
    return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

# use threads cuz lowkey kinda slow and hella bottlenecks if one process
# thread 1: capture
def capture_loop(monitor):
    global latest_frame
    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            with frame_lock:
                latest_frame = frame

# thread 2: interface
def inference_loop():
    global latest_result
    last_frame_id = None

    # one deque per hand (index 0 and 1)
    # each deque stores (gesture_name, score) tuples
    history = [deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE)]

    while True:
        with frame_lock:
            frame = latest_frame
            frame_id = id(frame)

        # skip if frame no change
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        small = cv2.resize(frame, (0, 0), fx=INFERENCE_SCALE, fy=INFERENCE_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = recognizer.recognize(mp_image)

        if not result.gestures:
            # no hands detected lowkey means clear history and push empty result
            history[0].clear()
            history[1].clear()
            with result_lock:
                latest_result = result
            continue

        # check each detected hand
        confirmed = True
        for idx, gesture_list in enumerate(result.gestures):
            if idx >= 2:
                break
            gesture_name = gesture_list[0].category_name
            history[idx].append(gesture_name)

            # count how many of the last N frames are cool w this gesture
            confirmations = history[idx].count(gesture_name)
            if confirmations < MIN_CONFIRMATIONS:
                confirmed = False

        # only update latest_result if gesture is stable enough
        if confirmed:
            with result_lock:
                latest_result = result

# webrtc video
class GestureVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._timestamp = 0

    async def recv(self):
        await asyncio.sleep(1 / TARGET_FPS)

        with frame_lock:
            frame = latest_frame
        with result_lock:
            result = latest_result

        if frame is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if result and result.hand_landmarks:
                rgb = draw_gestures_on_image(rgb, result)

        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        self._timestamp += int(90000 / TARGET_FPS)
        video_frame.pts = self._timestamp
        video_frame.time_base = fractions.Fraction(1, 90000)

        return video_frame

# webrtc signaling
pcs = set()

async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(GestureVideoTrack())

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

# entry point
if __name__ == "__main__":
    monitor = get_monitor()

    threading.Thread(target=capture_loop, args=(monitor,), daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    app = web.Application()
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    async def add_cors(request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    async def options_handler(request):
        return web.Response(headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        })

    app.router.add_route('OPTIONS', '/offer', options_handler)
    app.middlewares.append(web.middleware(add_cors))

    print("WebRTC signaling server at http://0.0.0.0:8080/offer")
    web.run_app(app, host="0.0.0.0", port=8080)