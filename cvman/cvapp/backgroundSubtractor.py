# -*-coding:utf8-*-#

import cv2
import imutils
import numpy as np

import cvlib.core as core
import cvlib.canvas as canvas
import cvlib.image as image
from cvlib.video import CVCapture, CVWriter

def backgroundSubtractor1(videoFile):
    cap = cv2.VideoCapture(videoFile)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
     
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        cv2.imshow("", fgmask)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def backgroundSubtractor2(videoFile):
    cap = cv2.VideoCapture(videoFile)
    fgbg = cv2.createBackgroundSubtractorMOG2()
     
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)

        cv2.imshow("", fgmask)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def backgroundSubtractor3(videoFile):
    cap = cv2.VideoCapture(videoFile)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
     
    while(1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow("", fgmask)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def grabCut(imgFile):
    img = cv2.imread(imgFile)
    height, width, channal = img.shape

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # rect = (1, 1, width-1, height-1)
    rect = (49, 41, 498, 637)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 5, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    result = cv2.bitwise_and(img, img, mask=mask2)

    return result

    cv2.imshow("", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(videoFile):
    # backgroundSubtractor1(videoFile)
    # backgroundSubtractor2(videoFile)
    backgroundSubtractor3(videoFile)

if __name__ == '__main__':
    main()