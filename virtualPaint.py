import cv2
import numpy as np
import os
import time
import handtrackModule as hm

image = cv2.imread('header/1.jpg')
image = cv2.resize(image, (1280, 125))
overlayList = []
overlayList.append(image)

header = overlayList[0]


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = hm.handDetector(detectionCon=0.8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # 2. Find hand landmarks
    img = detector.findHands(img, draw=False)
    lmlist = detector.findPosition(img, draw=False)
    
    if len(lmlist)!=0:
        print(lmlist)
        
        # tip of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
    
    # 3. check which fingers are up
    # 4. if selection mode - 2 fingers are up
    # 5. if drawing mode - Index finger is up
    
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()