#importing libraries
from __future__ import division
import cv2
import pyautogui
import mediapipe as mp
import math
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#Input image code initialization
points = []
protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],
[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],
[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
frame = cv2.imread("1.jpeg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
aspect_ratio = frameWidth/frameHeight
threshold = 0.2
t = time.time()
inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward()

#Webcam code initialization
tipIds = [4, 8, 12, 16, 20]
mode = ''
active = 0
color = (0,215,255)
#hand range 50-200
hmin = 50
hmax = 200
wCam, hCam = 640, 480
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume range -65 - 0
volRange = volume.GetVolumeRange() 
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

#MENU
print()
print("****************************************************************")
print("                                 WELCOME")
print("                  HANDPOSE DETECTION WITH GESTURE CONTROL")
print("****************************************************************")
print("OPTIONS:-")
print("PRESS 1 TO RUN WEBCAM")
print("PRESS 2 TO INPUT AN IMAGE")
print()


op = int(input())
print()

if op == 1:

    while True:

        lmList = []
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

        fingers = []

        if len(lmList) != 0:

            if lmList[tipIds[0]][1] > lmList[tipIds[0 -1]][1]:
                if lmList[tipIds[0]][1] >= lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            elif lmList[tipIds[0]][1] < lmList[tipIds[0 -1]][1]:
                if lmList[tipIds[0]][1] <= lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
           
            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)

            if (fingers == [0,0,0,0,0]) & (active == 0 ):
                mode='None'
            elif (fingers == [0, 1, 0, 0, 0] or fingers == [0, 1, 1, 0, 0]) & (active == 0 ):
                mode = 'Scroll'
                active = 1
            elif (fingers == [1, 1, 0, 0, 0] ) & (active == 0 ):
                mode = 'Volume'
                active = 1

        if mode == 'Volume':
            active = 1
            print(mode)
            if len(lmList) != 0:
                if fingers[-1] == 1:
                    active = 0
                    mode = 'None'
                    print(mode) 

                else:
                    # print(lmList[4], lmList[8])
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    length = math.hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [hmin, hmax], [minVol, maxVol])
                    volBar = np.interp(vol, [minVol, maxVol], [400, 150])
                    volPer = np.interp(vol, [minVol, maxVol], [0, 100])
                                        
                    # print(vol)
                    volume.SetMasterVolumeLevel(vol, None)

                    if length < 50:
                        cv2.circle(img, (cx, cy), 11, (0, 0, 255), cv2.FILLED)

                    cv2.rectangle(img, (30, 150), (55, 400), (209, 206, 0), 3)
                    cv2.rectangle(img, (30, int(volBar)), (55, 400), (215, 255, 127), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)}%', (25, 430), cv2.FONT_HERSHEY_COMPLEX, 0.9, (209, 206, 0), 3)

                            
        if mode == 'Scroll':
            active = 1
            print(mode)
            if len(lmList) != 0:
                if fingers == [0,1,0,0,0]:
                    pyautogui.scroll(300)
                    print('up')
                if fingers == [0,1,1,0,0]:
                    pyautogui.scroll(-300)
                    print('down')
                elif fingers == [0, 0, 0, 0, 0]:
                    active = 0
                    mode = 'None'


        cv2.imshow("Img", img)
        cv2.waitKey(1)

            
elif op == 2:
    # Empty list to store the detected keypoints
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        if prob > threshold :

            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    cv2.imwrite('Output-Skeleton.jpg', frame)
    cv2.waitKey(0)




