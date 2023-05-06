# Zach Vincent
# zvincent@nd.edu
# 26/04/2023

import cv2
import mediapipe as mp
import imutils
import pyautogui

MOVING_AVERAGE_LEN  = 5
SCALE_FACTOR_X      = 1
SCALE_FACTOR_Y      = 1

cam = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

mpos = [(0,0) for i in range(MOVING_AVERAGE_LEN)]

while True:
    ret,img = cam.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for i,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                
                cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        mx = pyautogui.size()[0] - pyautogui.size()[0]*handLms.landmark[0].x
        my = pyautogui.size()[1]*handLms.landmark[0].y

        mpos.append((mx*SCALE_FACTOR_X, my*SCALE_FACTOR_Y))
        mpos.pop(0)

        mx = sum([x for (x,y) in mpos])/MOVING_AVERAGE_LEN
        my = sum([y for (x,y) in mpos])/MOVING_AVERAGE_LEN

        pyautogui.moveTo(mx, my)


    cv2.imshow("Hands", img)
    k = cv2.waitKey(1)
    if k == 27:
        break
    

cam.release()
cv2.destroyAllWindows