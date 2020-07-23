# -*-coding:utf8-*-#
import os
import numpy as np

from cvman.cvio import gui as cvGui
from cvman.cvio import canvas as cvCanvas
from cvman.cvio import image as cvImage

import cv2

BOARDWIDTH  = 9
BOARDHEIGHT = 6
SQUARESIZE  = 50

# 计算标定板上模块的实际物理坐标
def calRealPoint(boardwidth, boardheight, imgNumber, squaresize):
    imgPoint = np.zeros((boardheight * boardwidth, 3), np.float32)
    for row in range(boardheight):
        for col in range(boardwidth):
            imgPoint[row * boardwidth + col][0] = col * squaresize
            imgPoint[row * boardwidth + col][1] = row * squaresize
            imgPoint[row * boardwidth + col][2] = 0

    objPoint = []
    for imgIndex in range(imgNumber):
        objPoint.append(imgPoint)

    return objPoint

# # 设置相机的初始参数 也可以不估计
# # 标定结束后进行评价
# def calibrationEvaluate(objRealPoint, corners, intrinsic, distortionCoeff, rvecs, tvecs):
#     print("每幅图像的定标误差：")
#     for i in range(len(corners)):
#         imagePoint = cv2.projectPoints(objRealPoint[i], rvecs[i], tvecs[i], intrinsic, distortionCoeff)
#         print(imagePoint)
#         return
#         tempImagePoint = corners[i]
#         print(type(imagePoint))
#         print(imagePoint.shape)
#         # Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
#         # Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
#         imagePoints2Mat = []
#         for j in range(tempImagePoint.shape[0]):
#             print(imagePoint[j])
#             return
#             point = [imagePoint[j][0][0], imagePoint[j][0][1]]
#             imagePoints2Mat.append(point)

#         err = cv2.norm(imagePoints2Mat, tempImagePoint, cv2.NORM_L2)
#         totalErr = err + totalErr
#         print("第 %s 幅图像的平均误差：%s 像素" % (i+1, err))

#     print("总体平均误差：%s 像素" % (totalErr / (corners.size() + 1)))


def guessCameraParam(width, height):
    # 分配内存
    intrinsic = np.zeros((3, 3), np.float64)
    distortionCoeff = np.zeros((1, 5), np.float64)

    # 现以NiKon D700相机为例进行求解其内参数矩阵：   
    # 焦距 f = 35mm  最高分辨率：4256×2832  传感器尺寸：36.0×23.9 mm
    # 根据以上定义可以有：
    # u0= 4256/2 = 2128   v0= 2832/2 = 1416  dx = 36.0/4256   dy = 23.9/2832
    # fx = f/dx = 4137.8   fy = f/dy = 4147.3
    # 
    # 焦距: fx fy
    # 光学中学: cx cy
    # fx 0  cx
    # 0  fy cy
    # 0  0  1
    intrinsic[0, 0] = 4137.8       # fx
    # intrinsic[0, 0] = 897.8       # fx
    intrinsic[0, 2] = width/2      # cx
    # intrinsic[1, 1] = 897.3       # fy
    intrinsic[1, 1] = 4147.3       # fy
    intrinsic[1, 2] = height/2     # cy

    intrinsic[0, 1] = 0
    intrinsic[1, 0] = 0
    intrinsic[2, 0] = 0
    intrinsic[2, 1] = 0
    intrinsic[2, 2] = 1

    # k1 k2 p1 p2 k3
    # 径向畸变系数: k1 k2 k3
    # 径向畸变跟像素点到光心的距离的平方有关: k1*r^2 + k2*r^4 + k3*r^6
    # 切向畸变系数: p1 p2
    # 透镜和成像平面不平行造成: 2*p1*x*y + p2(r^2 + 2*x^2)
    distortionCoeff[0, 0] = -0.000001  # k1 
    distortionCoeff[0, 1] = -0.000001  # k2 
    distortionCoeff[0, 2] = 0.000001   # p1 
    distortionCoeff[0, 3] = 0.000001   # p2 
    distortionCoeff[0, 4] = 0          # k3 

    return intrinsic, distortionCoeff

def outputCameraParam(intrinsic, distortionCoeff, rvecs, tvecs):
    # 输出数据
    print("fx : %s   fy : %s" % (intrinsic[0, 0], intrinsic[1, 1]))
    print("cx : %s   cy : %s" % (intrinsic[0, 2], intrinsic[1, 2]))

    print("k1 : %s" % (distortionCoeff[0, 0]))
    print("k2 : %s" % (distortionCoeff[0, 1]))
    print("p1 : %s" % (distortionCoeff[0, 2]))
    print("p2 : %s" % (distortionCoeff[0, 3]))
    print("k3 : %s" % (distortionCoeff[0, 4]))

    print("r : %s" % (rvecs))

    print("t : %s" % (tvecs))

def getWorldPoints(imgPoint, intrinsic, distortionCoeff, rvec, tvec):
    #                        |u|   |x|
    #  R^(-1) * M^(-1) * s * |v| = |y| + R^(-1) * T
    #                        |1|   |z|
    # 获取图像坐标
    # u,v,1
    imagePoint = np.ones((3, 1), dtype=np.float64)
    imagePoint[0, 0] = imgPoint[0]
    imagePoint[1, 0] = imgPoint[1]
    imagePoint = np.mat(imagePoint)

    # 计算比例参数S
    zConst = 0
    rotationMatrix = np.mat(cv2.Rodrigues(rvec)[0])
    cameraMatrix = np.mat(intrinsic)
    leftMat = rotationMatrix.I * cameraMatrix.I * imagePoint
    rightMat = rotationMatrix.I * np.mat(tvec)
    s = zConst + rightMat[2, 0]
    s /= leftMat[2, 0]

    # 计算世界坐标
    wcPoint = rotationMatrix.I * (s * cameraMatrix.I * imagePoint - np.mat(tvec))
    return (wcPoint[0, 0], wcPoint[1, 0], wcPoint[2, 0])

def calCamera(path):
    # 相机标定时需要采用的图像帧数
    frameNumber = 7
    goodFrameCount = 0
    corners = []
    while goodFrameCount < frameNumber:
        goodFrameCount = goodFrameCount + 1
        filename = os.path.join(path, "right" + str(goodFrameCount) + ".jpg")
 
        rgbImage = cvGui.imread(filename, cvGui.IMREAD_COLOR)
        grayImage = cvImage.cvtColor(rgbImage, cvImage.COLOR_BGR2GRAY)

        # 寻找棋盘图的内角点位置
        # corner shape is (width * height, 1, 2)
        isFind, corner = cv2.findChessboardCorners(rgbImage, (BOARDWIDTH, BOARDHEIGHT))
        if isFind:
            criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corner = cv2.cornerSubPix(grayImage, corner, (5, 5), (-1, -1), criteria)
            # # 棋盘格角点的绘制
            # img = cv2.drawChessboardCorners(rgbImage, (BOARDWIDTH, BOARDHEIGHT), corner, isFind)
            # cvGui.imshow(img, str(goodFrameCount))
            # cvGui.waitKey()
            corners.append(corner)
            print("The image is good")
        else:
            print("The image is bad please try again")

    # image info
    imgShape = cvImage.shape(rgbImage)
    # 设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置
    intrinsic, distortionCoeff = guessCameraParam(imgShape[0], imgShape[1])
    print("guess successful")
    # 计算实际的校正点的三维坐标
    objRealPoint = calRealPoint(BOARDWIDTH, BOARDHEIGHT, len(corners), SQUARESIZE)
    print("cal real successful")
    # print(corner)
    # 标定摄像头
    ret, intrinsic, distortionCoeff, rvecs, tvecs = cv2.calibrateCamera(objRealPoint, corners, grayImage.shape[::-1], None, None)
    # PNP 标定相机
    # SOLVEPNP_ITERATIVE 适合点在同一平面上的情况
    #ret, rvecs, tvecs = solvePnP(objRealPoint, corners, intrinsic, distortionCoeff)
    print("calibration successful")
    # 保存并输出参数
    outputCameraParam(intrinsic, distortionCoeff, rvecs[0], tvecs[0])
    # calibrationEvaluate(objRealPoint, corners, intrinsic, distortionCoeff, rvecs, tvecs)
    print("out successful")
    # 显示畸变校正效果
    cImage = cv2.undistort(rgbImage, intrinsic, distortionCoeff)
    cvGui.imshow(cImage, "undistort")

    # /********图像坐标到世界坐标*******/
    # 已知 Pc=R*Po+T 
    # 定义Pc为相机坐标系的点值，Po为世界坐标系的点值，R、T为世界坐标系和相机坐标系的相对外参。
    # solvePnP返回的raux是旋转向量，可通过罗德里格斯变换成旋转矩阵R。
    image_idx = 0
    rotation_matrix = cv2.Rodrigues(rvecs[image_idx])
    H = rotation_matrix[0].copy()
    translation_ve = tvecs[image_idx].copy()
    H[0, 2] = translation_ve[0, 0]
    H[1, 2] = translation_ve[1, 0]
    H[2, 2] = translation_ve[2, 0]
    hu = intrinsic * H
    hu2 = np.linalg.inv(hu)
    a1 = hu2[0, 0]
    a2 = hu2[0, 1]
    a3 = hu2[0, 2]
    a4 = hu2[1, 0]
    a5 = hu2[1, 1]
    a6 = hu2[1, 2]
    a7 = hu2[2, 0]
    a8 = hu2[2, 1]
    a9 = hu2[2, 2]

    # 显示一张原图，矫正之后，再寻找角点
    name = os.path.join(path, "right" + str(image_idx+1) + ".jpg")

    show_tuxiang_gray = cvGui.imread(name, cvGui.IMREAD_GRAYSCALE)
    show_tuxiang_rgb = cvGui.imread(name, cvGui.IMREAD_COLOR)
	# 根据畸变系数，进行图片的矫正
    show_gray = cv2.undistort(show_tuxiang_gray, intrinsic, distortionCoeff)
    show_rgb = cv2.undistort(show_tuxiang_rgb, intrinsic, distortionCoeff)
    # corner.clear()
    isFind, corner = cv2.findChessboardCorners(show_gray, (BOARDWIDTH, BOARDHEIGHT))
    corner = cv2.cornerSubPix(show_gray, corner, (5, 5), (-1, -1), criteria)
    
    for i in range(BOARDWIDTH * BOARDHEIGHT):
        x = round(corner[i][0][1])
        y = round(corner[i][0][1])
        cvCanvas.circle(show_rgb, (x, y), 3, (0, 0, 255), -1, 8)

    cvGui.imshow(show_rgb, "Image Cor")

    # print(objRealPoint[0][26])
    # xe = corner[26][0][0]  # 图像中点坐标x
    # ye = corner[26][0][1]  # 图像中点坐标y
    # wP = getWorldPoints((xe, ye), intrinsic, distortionCoeff, rvecs[0], tvecs[0])
    # print(wP)
    # return

    # 相机矫正后的结果
    shijie = []
    for i in range(BOARDWIDTH * BOARDHEIGHT):
        xe = corner[i][0][0]  # 图像中点坐标x
        ye = corner[i][0][1]  # 图像中点坐标y
        xw = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9) # 世界坐标中x值
        yw = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9) # 世界坐标中Y值
        shijie.append((xw, yw))
    
    show_shijie = np.zeros((480, 640), np.uint8)
    for i in range(BOARDWIDTH * BOARDHEIGHT):
        x = round(shijie[i][0]) + 220
        y = round(shijie[i][1]) + 120

        cvCanvas.circle(show_shijie, (int(x), int(y)), 3, (255, 255, 255), -1, 8)

    cvGui.imshow(show_shijie, "Real World - adjust")

    # print(objRealPoint[0][25])
    # xe = corners[image_idx][25][0][0]  # 图像中点坐标x
    # ye = corners[image_idx][25][0][1]  # 图像中点坐标y
    # wP = getWorldPoints((xe, ye), intrinsic, distortionCoeff, rvecs[0], tvecs[0])
    # print(wP)
    # return

    # /*对比没有进行相机矫正的结果*/
    shijie.clear()
    for i in range(BOARDWIDTH * BOARDHEIGHT):
        xe = corners[image_idx][i][0][0]  # 图像中点坐标x
        ye = corners[image_idx][i][0][1]  # 图像中点坐标y
        xw = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9) # 世界坐标中x值
        yw = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9) # 世界坐标中Y值
        shijie.append((xw, yw))

    show_shijie2 = np.zeros((480, 640), np.uint8)
    for i in range(BOARDWIDTH * BOARDHEIGHT):
        x = round(shijie[i][0]) + 220
        y = round(shijie[i][1]) + 120

        cvCanvas.circle(show_shijie2, (int(x), int(y)), 3, (255, 255, 255), -1, 8)

    cvGui.imshow(show_shijie2, "Real World - no adjust")

    #/*************图像坐标到世界坐标结束***********************/

    cvGui.waitKey()

def calcImage(imgFile):
    rgbImage = cvGui.imread(imgFile, cvGui.IMREAD_COLOR)
    imgShape = cvImage.shape(rgbImage)
    # 设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置
    intrinsic, distortionCoeff = guessCameraParam(imgShape[1], imgShape[0])
    print("guess successful")
    # 图像和世界坐标
    # corners = np.zeros((4, 2), np.float32)
    corners = np.array([[
        [ 664.0, 497.0  ],
        [ 231.0, 331.0  ],
        [ 63.0 , 268.0  ],
        [ 451.0, 231.0  ],
        [ 667.0, 260.0  ],
        [ 1054.0, 317.0 ],
        ]], dtype=np.float64)
    # objRealPoint = np.zeros((4, 3), np.float32)
    objRealPoint = np.array([[
        [ 0.0 , 0.0, 0.0   ],
        [ 0.0 , 52.5, 0.0  ],
        [ 0.0 , 105.0, 0.0 ],
        [ 68.0, 105.0, 0.0 ],
        [ 68.0, 52.5, 0.0  ],
        [ 68.0, 0.0, 0.0   ],
        ]], dtype=np.float64)
    print("cal real successful")
    # # 标定摄像头
    # print(objRealPoint.shape)
    # ret, intrinsic, distortionCoeff, rvecs, tvecs = cv2.calibrateCamera(objRealPoint, corners, rgbImage.shape[::-1], None, None)
    # # 保存并输出参数
    # outputCameraParam(intrinsic, distortionCoeff, rvecs, tvecs)
    # return
    # PNP 标定相机
    # SOLVEPNP_ITERATIVE 适合点在同一平面上的情况
    ret, rvec, tvec = cv2.solvePnP(objRealPoint, corners, intrinsic, distortionCoeff, flags=cv2.SOLVEPNP_ITERATIVE)
    print("calibration successful")
    # 保存并输出参数
    outputCameraParam(intrinsic, distortionCoeff, rvec, tvec)
    # calibrationEvaluate(objRealPoint, corners, intrinsic, distortionCoeff, rvecs, tvecs)
    print("out successful")

    # 根据畸变系数，进行图片的矫正
    showImage = cv2.undistort(rgbImage, intrinsic, distortionCoeff)
    cvGui.imshow(showImage, "Image Cor")
    cvGui.waitKey()

    # /********图像坐标到世界坐标*******/
    xe = 1054  # 图像中点坐标x
    ye = 317  # 图像中点坐标y
    #  (488, 289) -> (34, 52.5)
    wP = getWorldPoints((xe, ye), intrinsic, distortionCoeff, rvec, tvec)

    print(wP)

def run(path):
    # calCamera(path)
    filename = os.path.join(path, "football.jpg")
    calcImage(filename)


if __name__ == '__main__':
    main()