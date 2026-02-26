import mediapipe as mp
import cv2
import numpy as np
import pyautogui

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

    #for data structure breakdown
    if recognition_result.gestures:
      top_gesture = recognition_result.gestures[0][0]
      
      #print(f"Gesture: {top_gesture.category_name} - Score: {top_gesture.score:.2f}")

      if top_gesture.category_name == "Pointing_Up":
        pointing_up_img = cv2.imread('./pointing_up.jpeg')
        pointing_up_img = cv2.resize(pointing_up_img, (480, 480))
        cv2.imshow('Result Image', pointing_up_img)

      elif top_gesture.category_name == "Closed_Fist":
        closed_fist_img = cv2.imread('./closed_fist.jpg')
        closed_fist_img = cv2.resize(closed_fist_img, (480, 480))
        cv2.imshow('Result Image', closed_fist_img)
        #try:
        #  np_window = pyautogui.getActiveWindow()

        #  if np_window:
        #     np_window.moveTo(150,200)

        #except Exception as e:
        #  print(f"An error occurred: {e}")
           
      
      elif top_gesture.category_name == "Victory":
        victory_img = cv2.imread('./victory.webp')
        victory_img = cv2.resize(victory_img, (480, 480))
        cv2.imshow('Result Image', victory_img)

      elif top_gesture.category_name == "ILoveYou":
        ily_img = cv2.imread('./i_love_you.jpg')
        ily_img = cv2.resize(ily_img, (480, 480))
        cv2.imshow('Result Image', ily_img)

      elif top_gesture.category_name == "Thumb_Up":
        tu_img = cv2.imread('./thumbs_up.gif')
        tu_img = cv2.resize(tu_img, (480, 480))
        cv2.imshow('Result Image', tu_img)


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
