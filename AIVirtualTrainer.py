import math
import cv2
import mediapipe as mp
import numpy as np

import PoseEstimationModule as pem
count = 0
pTime = 0
cap = cv2.VideoCapture(0)
detector = pem.PoseDetector()

while True:
    success, img1 = cap.read()
    img = cv2.resize(img1, (1280,720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    img = detector.findpose(img)
    lmlist = detector.findpositions(img)
    if len(lmlist)!=0:
        angle = detector.findAngle(img,12,14,16)
        detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle,(0,100),(100,650))
        cv2.rectangle(img,(1100,100),(1175,650),(255,0,0),3)
        cv2.rectangle(img,(1100, int(per)), (1175, 650), (255, 0, 0), cv2.FILLED)
        print(per)
    cv2.imshow("image",img)
    cv2.waitKey(1)

