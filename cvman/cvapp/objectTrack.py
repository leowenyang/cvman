# -*-coding:utf8-*-#

import cv2
import imutils
import numpy as np

import cvlib.core as core
import cvlib.canvas as canvas
import cvlib.image as image
from cvlib.video import CVCapture, CVWriter

def objTrack(videoFile):
    cap = cv2.VideoCapture(videoFile)

    # take first frame of the video
    ret, frame = cap.read()

    # setup initial location of window
    r, h, c, w = 313, 144, 660, 237
    trackWindow = (c, r, w, h)

    roi = frame[r:r+h, c:c+w]
    hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvRoi, np.array((0, 60, 32)), np.array((180, 255, 255)))
    roiHist = cv2.calcHist([hsvRoi], [0], mask, [180], [0, 180])
    cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination critera either 10 iteration or move by at least 1 pt
    termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while(1):
        ret, frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, trackWindow = cv2.meanShift(dst, trackWindow, termCrit)

            # Draw it on image
            x, y, w, h = trackWindow
            img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
            cv2.imshow('img', img2)

            k = cv2.waitKey(60) & 0xFF
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg", img2)
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

def run(videoFile):
    objTrack(videoFile)

if __name__ == '__main__':
    main()