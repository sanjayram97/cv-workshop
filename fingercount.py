import cv2
import mediapipe as mp
import os
import handtrackModule as hm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folder = "fingerimages"
myList = os.listdir(folder)
print(myList)
overLayList = []
for i in myList:
    img = cv2.imread(folder+"/"+i)
    img = cv2.resize(img, (200, 200))
    overLayList.append(img)


detector = hm.handDetector()
tipIds = [8, 12, 16, 20]

while True:
    success, img = cap.read()
    
    img[0:200, 0:200] = overLayList[1]
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    
    if len(lmlist)!=0:
        fingers = []
        
        # Thumb
        if lmlist[4][1] > lmlist[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in tipIds:
            if lmlist[id][2] < lmlist[id-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers.count(1))
        cv2.putText(img, str(fingers.count(1)), (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
        
cap.release()
cv2.destroyAllWindows()