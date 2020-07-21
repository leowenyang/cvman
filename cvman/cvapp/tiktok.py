# -*-coding:utf8-*-#

import cv2
import imutils
import numpy as np

#####
# - 失焦的图片和对焦准确的图片最大的区别就是正常图片轮廓明显，
#   而失焦图片几乎没有较大像素值之间的变化
# - 对图像的横向，以及纵向，分别做差分，累计差分可以用来作为判断是否失焦的参考
#####

# 简单设定阈值判断是否失焦
# True : 未失焦
# False : 失焦
def focusDetect(img):
    diff = 0
    diff_thre = 20
    diff_sum_thre = 1000

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    for i in range(int(rows / 10), rows, int(rows / 10)):
        for j in range(0, cols-1):
            if (abs(int(img[i][j+1]) - int(img[i][j])) > diff_thre):
                print('=============')
                print(int(img[i][j+1]))
                print(int(img[i][j]))
                print(abs(int(img[i][j+1]) - int(img[i][j])))
                print('=============')
                diff = diff + abs(int(img[i][j+1]) - int(img[i][j]))

    print(diff)
    if (diff < diff_sum_thre):
        return False
    return True

# 返回一个与焦距是否对焦成功的一个比例因子
# 经验值 : < 4.0  图片失焦
#         >= 4.0  图片正常
# True : 未失焦
# False : 失焦
def focus_measure_GRAT(img):
    totalsum = 0
    totalnum = 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    for i in range(0, rows-1):
        for j in range(0, cols-1):
            A = abs(int(img[i+1][j]) - int(img[i][j]))
            B = abs(int(img[i][j+1]) - int(img[i][j]))
            totalsum += max(A, B)
            totalnum += 1

    if totalsum/totalnum < 4.0:
        return False
    return True

def detectBigMan(frame):
    # faceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_upperbody.xml")
    faceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt2.xml")
    eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

    # 如果维度为3，先转化为灰度图gray. 如果不为3, 原图就是灰度图
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.equalizeHist(gray)

    cv2.namedWindow("gray", 0)
    cv2.resizeWindow("gray", 640, 360)
    # cv2.imshow("gray", gray) 

    # 1.3和5是特征的最小、最大检测窗口，它改变检测, 结果也会改变
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    cv2.namedWindow("detail", 0)
    cv2.resizeWindow("detail", 640, 360)
    result = []
    for (x, y, w, h) in faces:
        cv2.imshow("gray", gray[y:y+h, x:x+w])
        # result.append((x, y, x + w, y + h))
        roiGray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roiGray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            result.append((x, y, x + w, y + h))
        cv2.rectangle(frame,     # draw rectangle on original image
                (x, y),          # upper left corner
                (x+w, y+h),      # lower right corner
                (0, 255, 0),     # green
                2)               # thickness
        cv2.imshow("detail", frame)        # show input image with green boxes drawn
        cv2.waitKey(30)             # wait for user key press
    return result

def run(videoFile):
    cap = cv2.VideoCapture(videoFile)
      
    frameNumber = 0
    frameFPS = cap.get(cv2.CAP_PROP_FPS)

    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        # 跳帧 (1s 取 1 帧)
        frameNumber = frameNumber + 1
        if frameNumber % int(frameFPS):
            continue

        cv2.namedWindow("frame", 0)
        cv2.imshow("frame", frame)

        if len(detectBigMan(frame)) and focus_measure_GRAT(frame):
            cv2.imwrite('./tmp/tiktok/' + str(frameNumber) + '.jpg', frame)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()