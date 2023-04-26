from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
kinect_c = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

while True:
  if kinect.has_new_depth_frame():
    frame = kinect.get_last_depth_frame()

    frame = np.reshape(frame, (424,512))

    cv2.circle(frame, (212,256), 10, (255,255,255), cv2.FILLED)

    cv2.imshow('frame', frame)

  k = cv2.waitKey(1)
  if k == 27:
    break

kinect.close()
cv2.destroyAllWindows()