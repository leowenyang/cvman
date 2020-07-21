#-*-coding:utf8-*-#
import os
import cv2
import json
import numpy as np

import cvapp.faceDetect as faceDetect
import cvapp.backgroundSubtractor as backgroundSubtractor

from cvapp.FaceChanger import *

def blanceColor(img):
    b, g, r = cv2.split(img)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    merged = cv2.merge([b,g,r])
    return merged

# 脸部闪烁
# 脸部摆动会有问题
def run(imgFile):
    fc = FaceChanger()

    # 取出人脸区域
    faces = faceDetect.detectFaces(imgFile)
    imgFace = cv2.imread(imgFile)
    for (x1, y1, x2, y2) in faces:
        # cv2.rectangle(imgFace,     # draw rectangle on original image
        #         (x1, y1),      # upper left corner
        #         (x2, y2),      # lower right corner
        #         (0, 255, 0),   # green
        #         2)             # thickness
        # imgFace = backgroundSubtractor.grabCut(imgFile)
        imgFace = imgFace[y1+10:y2-10, x1+10:x2-10]
        break
    # cv2.imshow("test", imgFace)        # show input image with green boxes drawn
    # cv2.waitKey(0)             # wait for user key press
    # return

    # 获取视频中人脸的 mask
    with open("C:/Users/leo/Desktop/face/2/jpg/vott-json-export/2-export.json", encoding='utf-8') as f:
        json_object = json.load(f)
        for (index, asset) in json_object['assets'].items():
            path = asset['asset']['path'][5:]
            print(path)
            width = int(asset['asset']['size']['width'])
            height = int(asset['asset']['size']['height'])
            boxWidth = int(asset['regions'][0]['boundingBox']['width'])
            boxHeight = int(asset['regions'][0]['boundingBox']['height'])
            boxLeft = int(asset['regions'][0]['boundingBox']['left'])
            boxTop = int(asset['regions'][0]['boundingBox']['top'])
            points = asset['regions'][0]['points']

            img = cv2.imread(path)
            # # rect
            # cv2.rectangle(img,     # draw rectangle on original image
            #         (boxLeft, boxTop),      # upper left corner
            #         (boxLeft + boxWidth, boxTop + boxHeight),      # lower right corner
            #         (0, 255, 0),   # green
            #         2)             # thickness
            
            # Poly
            imgMask = np.zeros((height, width), dtype=np.uint8)
            x_data = []
            y_data = []
            for point in points:
                x_data.append(int(point['x']))
                y_data.append(int(point['y']))
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            pts = np.vstack((x_data, y_data)).astype(np.int32).T
            cv2.fillPoly(imgMask, [pts], (255, 255, 255))
            imgMask = imgMask[boxTop:boxTop + boxHeight, boxLeft:boxLeft + boxWidth]
            # find angle, to rotation img, in order to easy to recognize
            rect = cv2.fitEllipse(pts)
            # box = np.int0(cv2.boxPoints(rect))
            # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            matShift = cv2.getRotationMatrix2D((rect[0][0]-boxLeft, rect[0][1]-boxTop), rect[2], 1)

            imgFace2 = img[boxTop:boxTop + boxHeight, boxLeft:boxLeft + boxWidth]

            # two img is face
            if fc.loadImages(imgFace2, imgFace):
                imgFace3 = fc.run()
                imgFace3 = cv2.bitwise_and(imgFace3, imgFace3, mask=imgMask)
                # imgFace3 is float64
                imgMask2 = imgFace3.astype(imgFace2.dtype)
                
                # # color balance
                # cv2.imshow("", imgMask2)
                # cv2.waitKey(0)
                # imgMask2 = blanceColor(imgMask2)
                # cv2.imshow("", imgMask2)
                # cv2.waitKey(0)
                # return

                # merge
                output = cv2.seamlessClone(imgMask2, img, imgMask, (int(boxLeft+boxWidth/2), int(boxTop+boxHeight/2)), cv2.NORMAL_CLONE)
                # cv2.imshow(index, output)
                cv2.imwrite(path, output)
            else:
                imgFace2 = cv2.warpAffine(imgFace2, matShift, (boxWidth, boxHeight))
                # cv2.imshow("", imgFace2)
                # cv2.waitKey(0)
                # retry
                if fc.loadImages(imgFace2, imgFace):
                    imgFace3 = fc.run()
                    imgFace3 = cv2.bitwise_and(imgFace3, imgFace3, mask=imgMask)
                    # imgFace3 is float64
                    imgMask2 = imgFace3.astype(imgFace2.dtype)

                    # merge
                    output = cv2.seamlessClone(imgMask2, img, imgMask, (int(boxLeft+boxWidth/2), int(boxTop+boxHeight/2)), cv2.NORMAL_CLONE)
                    cv2.imwrite(path, output)
                # else:
                #     # transform
                #     print(boxHeight, boxWidth)
                #     imgFace2 = cv2.resize(imgFace, (boxWidth, boxHeight), None, 0, 0)
                #     imgMask2 = cv2.bitwise_and(imgFace2, imgFace2, mask=imgMask)

                #     # merge
                #     output = cv2.seamlessClone(imgMask2, img, imgMask, (int(boxLeft+boxWidth/2), int(boxTop+boxHeight/2)), cv2.NORMAL_CLONE)
                #     # cv2.imshow(index, output)
                #     cv2.imwrite(path, output)

            # for row in range(boxTop, boxTop+boxHeight):
            #     for col in range(boxLeft, boxLeft+boxWidth):
            #         w1 = imgMask[row-boxTop, col-boxLeft]/255
            #         b, g, r = img[row, col]
            #         b1, g1, r1 = imgMask2[row-boxTop, col-boxLeft]
            #         b = b if (w1 == 0) else b1
            #         g = g if (w1 == 0) else g1
            #         r = r if (w1 == 0) else r1
            #         img[row, col] = (b, g, r)
            # cv2.imshow(index, img)        # show input image with green boxes drawn
    
    # cv2.waitKey(0)             # wait for user key press
        

    # 扩大 人脸区域 到 mask
    # 融合 人脸区域
    # 处理整个视频

if __name__ == '__main__':
    run()