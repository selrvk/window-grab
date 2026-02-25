import mediapipe as mp
import cv2
import numpy as np

from utils import draw_landmarks_on_image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Get camera frame
cap = cv2.VideoCapture(0)

if not cap.isOpened():

  print("Can't open camera!")
  exit()

while True:
    
    ret, frame = cap.read()

    if not ret:
       print("Can't receive frame!")
       break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # static image file for testing (works)
    #image = mp.Image.create_from_file("image.jpg")

    detection_result = detector.detect(mp_image)
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    final_bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Detection', final_bgr_image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
