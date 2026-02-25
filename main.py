import mediapipe as mp
import cv2
import numpy as np

from utils import draw_landmarks_on_image, display_batch_of_images_with_gestures_and_hand_landmarks, draw_gestures_on_image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
#detector = vision.HandLandmarker.create_from_options(options)

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []

# Get camera frame
cap = cv2.VideoCapture(0)

if not cap.isOpened():

  print("Can't open camera!")
  exit()

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    recognition_result = recognizer.recognize(mp_image)

    if recognition_result.hand_landmarks:
        annotated_image = draw_gestures_on_image(rgb, recognition_result)
    else:
        annotated_image = rgb

    final_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Gesture Recognition', final_bgr_image)

    #esc
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cv2.destroyAllWindows()
cap.release()
