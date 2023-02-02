# Zach Vincent
# zvincent@nd.edu
# 05/10/2022

import numpy as np
import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):

  frame = img.copy()

  face_rects = face_cascade.detectMultiScale(frame)

  for (x,y,w,h) in face_rects:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 5)

  return frame

while True:

  ret, frame = cam.read()

  result = detect_face(frame)

  cv2.imshow('Detect face from webcam', result)

  c = cv2.waitKey(1)
  if c == 27:
    break

cam.release()
cv2.destroyAllWindows()