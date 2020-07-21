# -*-coding:utf8-*-#
import numpy as np

from cvman.cvlib import gui as cvGui
from cvman.cvlib import canvas as cvCanvas
from cvman.cvlib import image as cvImage
 
# 度数转换(弧度 -> 度)
def degreeTrans(theta):
    res = theta / np.pi * 180
    return res
 
# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(img, degree):
    # 旋转中心为图像中心
    height, width, channel = cvImage.shape(img)
    return cvImage.rotate(img, (int(width/2), int(height/2)), degree, 1)

# 通过霍夫变换计算角度
def calcDegree(img):
    midImage = cvImage.cvtColor(img, cvImage.COLOR_BGR2GRAY)
    dstImage = cvImage.canny(img, 50, 200)
    lineimage = img.copy()
 
    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cvImage.houghLines(dstImage, 1, np.pi/180, 200)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            sum += theta
            cvCanvas.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cvCanvas.LINE_AA)
 
    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)
    angle = degreeTrans(average) - 90
    return angle

def run(imgFile, outFile):
    image = cvGui.imread(imgFile)
    # cvGui.imshow(image)
    # 倾斜角度矫正
    degree = calcDegree(image)
    rotate = rotateImage(image, degree)
    cvGui.imwrite(outFile, rotate)
