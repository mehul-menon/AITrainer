import math
import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self,mode=False,complexity=1,smooth_l=True,en_segm=False,smooth_segm=True,detectCon=0.5,trackCon=0.5):
        self.mode = False
        self.complexity = complexity
        self.smooth_l = smooth_l
        self.en_segm = en_segm
        self.smooth_segm = smooth_segm
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth_l,self.en_segm,self.smooth_segm,self.detectCon,self.trackCon)
    def findpose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    def findpositions(self,img,draw=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
        return self.lmlist
    def findAngle(self,img,p1,p2,p3,draw=True):
        # storing co-ordinates of points p1, p2, p3
        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        x3,y3 = self.lmlist[p3][1:]
        #finding angle
        angle = math.degrees(math.atan2((y3-y2),(x3-x2))-math.atan2((y1-y2),(x1-x2))) 
        if angle<0:
            angle+=360
        if draw:
            cv2.circle(img,(x1,y1),5,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2-50,y2+50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        return angle
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img1 = cap.read()
        img = cv2.resize(img1, (640, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = detector.findpose(img)
        lmlist = detector.findpositions(img)
        if len(lmlist)!=0:
            detector.findAngle(img,12,14,16)
            detector.findAngle(img, 11, 13, 15)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
        cv2.imshow("image",img)
        cv2.waitKey(1)
if __name__=='__main__':
    main()
