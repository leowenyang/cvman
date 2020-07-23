#-*-coding:utf8-*-#
import os
import cv2
import dlib

from cvman.cvio import gui as cvGui
from cvman.cvio import canvas as cvCanvas
from cvman.cvio import image as cvImage

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def detectFacesByDlib(imgFile):
    # opencv 读取图片，并显示
    img = cvGui.imread(imgFile)

    detector = dlib.get_frontal_face_detector() #获取人脸分类器
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果

    # enumerate是一个Python的内置方法，用于遍历索引
    # index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
    # left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
    result = []
    for index, face in enumerate(dets):
        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cvCanvas.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        result.append((left, top, right, bottom))
    # cvGui.imshow('body', img)
    # # 等待按键，随后退出，销毁窗口
    # cvGui.waitKey()

    return result

def detectLandMarkByDlib(imgFile):
    # opencv 读取图片，并显示
    img = cvGui.imread(imgFile)

    # shape_predictor_68_face_landmarks.dat是进行人脸标定的模型，基于HOG特征的
    predictorPath = os.path.join(ROOT_DIR, "..", "data/dlib/shape_predictor_68_face_landmarks.dat")
    print(predictorPath)
    detector = dlib.get_frontal_face_detector()         # 获取人脸分类器
    predictor = dlib.shape_predictor(predictorPath)     # 获取人脸检测器
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果

    result = []
    # enumerate是一个Python的内置方法，用于遍历索引
    # index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
    # left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
    for index, face in enumerate(dets):
        # # 在图片中标注人脸，并显示
        # left = face.left()
        # top = face.top()
        # right = face.right()
        # bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    
        # 寻找人脸的68个标定点
        shape = predictor(img, face) 
        # 遍历所有点，打印出其坐标，并圈出来
        for index, pt in enumerate(shape.parts()):
            print('Part {}: {}'.format(index, pt))
            ptPos = (pt.x, pt.y)
            cvCanvas.circle(img, ptPos, 2, (0, 0, 255), 1)
            result.append(ptPos)
    # cvGui.imshow('body', img)
    # # 等待按键，随后退出，销毁窗口
    # cvGui.waitKey()

    return result


"""
下面的代码将眼睛、人脸、笑脸在不同的图像上框出，如果需要在同一张图像上框出，改一下代码就可以了。
总之，利用opencv里训练好的haar特征的xml文件，在图片上检测出人脸的坐标，利用这个坐标，
我们可以将人脸区域剪切保存，也可以在原图上将人脸框出。剪切保存人脸以及用矩形工具框出人脸，
本程序使用的是PIL里的Image、ImageDraw模块。
此外，opencv里面也有画矩形的模块，同样可以用来框出人脸。
"""

# 返回图像中所有人脸的矩形坐标（矩形左上、右下顶点）
# 使用haar特征的级联分类器haarcascade_frontalface_default.xml
# 注：haarcascades目录下训练好的分类器必须以灰度图作为输入。
def detectFaces(imgFile):
    img = cvGui.imread(imgFile)
    xmlPath = os.path.join(ROOT_DIR, "..", "data/haarcascades/haarcascade_frontalface_default.xml")
    print(xmlPath)
    faceCascade = cv2.CascadeClassifier(xmlPath)
    
    # 如果维度为3，先转化为灰度图gray. 如果不为3, 原图就是灰度图
    if img.ndim == 3:
        gray = cvImage.cvtColor(img, cvImage.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 1.3和5是特征的最小、最大检测窗口，它改变检测, 结果也会改变
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
    return result

# 在原图像上画矩形，框出所有人脸。
def drawFaces(imgFile):
    img = cvGui.imread(imgFile)
    faces = detectFaces(imgFile)
    if faces:
        for (x1, y1, x2, y2) in faces:
            cvCanvas.rectangle(img,     # draw rectangle on original image
                    (x1, y1),      # upper left corner
                    (x2, y2),      # lower right corner
                    (0, 255, 0),   # green
                    2)             # thickness
        # cv2.imshow("", img)        # show input image with green boxes drawn
        # cv2.waitKey(0)             # wait for user key press
    # save
    cvGui.imwrite(imgFile, img)

# 检测眼睛，返回坐标
# 由于眼睛在人脸上，往往是先检测出人脸，再细入地检测眼睛。
# 故detectEyes可在detectFaces基础上来进行，代码中需要注意“相对坐标”。
def detectEyes(imgFile):
    xmlPath = os.path.join(ROOT_DIR, "..", "data/haarcascades/haarcascade_eye.xml")
    print(xmlPath)
    eyeCascade = cv2.CascadeClassifier(xmlPath)
    faces = detectFaces(imgFile)

    img = cvGui.imread(imgFile)
    gray = cvImage.cvtColor(img, cvImage.COLOR_BGR2GRAY)
    result = []
    for (x1, y1, x2, y2) in faces:
        roiGray = gray[y1:y2, x1:x2]
        eyes = eyeCascade.detectMultiScale(roiGray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            result.append((x1 + ex, y1 + ey, x1 + ex + ew, y1 + ey + eh))
    return result

# 在原图像上框出眼睛.
def drawEyes(imgFile):
    img = cvGui.imread(imgFile)
    eyes = detectEyes(imgFile)
    if eyes:
        for (x1, y1, x2, y2) in eyes:
            cvCanvas.rectangle(img,     # draw rectangle on original image
                    (x1, y1),      # upper left corner
                    (x2, y2),      # lower right corner
                    (0, 255, 0),   # green
                    2)             # thickness
        # cv2.imshow("", img)        # show input image with green boxes drawn
        # cv2.waitKey(0)             # wait for user key press
        # cv2.destroyAllWindows()    # remove windows from memory
    # save
    cvGui.imwrite(imgFile, img)

# 检测笑脸
def detectSmiles(imgFile):
    img = cvGui.imread(imgFile)
    xmlPath = os.path.join(ROOT_DIR, "..", "data/haarcascades/haarcascade_smile.xml")
    print(xmlPath)
    smilesCascade = cv2.CascadeClassifier(xmlPath)
    # 如果维度为3，先转化为灰度图gray. 如果不为3, 原图就是灰度图
    if img.ndim == 3:
        gray = cvImage.cvtColor(img, cvImage.COLOR_BGR2GRAY)
    else:
        gray = img

    smiles = smilesCascade.detectMultiScale(gray, 4, 5)
    result = []
    for (x, y, width, height) in smiles:
        result.append((x, y, x + width, y + height))
    return result

# 在原图像上框出笑脸
def drawSmiles(imgFile):
    img = cvGui.imread(imgFile)
    smiles = detectSmiles(imgFile)
    if smiles:
        for (x1, y1, x2, y2) in smiles:
            cvCanvas.rectangle(img,     # draw rectangle on original image
                    (x1, y1),      # upper left corner
                    (x2, y2),      # lower right corner
                    (0, 255, 0),   # green
                    2)             # thickness
        # cv2.imshow("", img)        # show input image with green boxes drawn
        # cv2.waitKey(0)             # wait for user key press
        # cv2.destroyAllWindows()    # remove windows from memory
    # save
    cvGui.imwrite(imgFile, img)

def run(imgFile):
    # drawSmiles(imgFile)
    # drawEyes(imgFile)
    # drawFaces(imgFile)
    # detectFacesByDlib(imgFile)
    detectLandMarkByDlib(imgFile)

if __name__ == '__main__':
    pass


import cv2
import numpy as np
import random

#Check if a point is insied a rectangle
def rect_contains(rect,point):
    if point[0] <rect[0]:
        return False
    elif point[1]<rect[1]:
        return  False
    elif point[0]>rect[2]:
        return False
    elif point[1] >rect[3]:
        return False
    return True

# Draw a point
def draw_point(img,p,color):
    cv2.circle(img,p,2,color)

#Draw delaunay triangles
def draw_delaunay(img,subdiv,delaunay_color):
    trangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0,0,size[1],size[0])
    for t in  trangleList:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        if (rect_contains(r,pt1) and rect_contains(r,pt2) and rect_contains(r,pt3)):
            cv2.line(img,pt1,pt2,delaunay_color,1)
            cv2.line(img,pt2,pt3,delaunay_color,1)
            cv2.line(img,pt3,pt1,delaunay_color,1)

# Draw voronoi diagram
def draw_voronoi(img,subdiv):
    (facets,centers) = subdiv.getVoronoiFacetList([])

    for i in range(0,len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr,np.int)
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.fillConvexPoly(img,ifacet,color)
        ifacets = np.array([ifacet])
        cv2.polylines(img,ifacets,True,(0,0,0),1)
        cv2.circle(img,(centers[i][0],centers[i][1]),3,(0,0,0))


if __name__ == '__main__':
    #Define window names;
    win_delaunary = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    #Turn on animations while drawing triangles
    animate = True

    #Define colors for drawing
    delaunary_color = (255,255,255)
    points_color = (0,0,255)

    #Read in the image
    img_path = "E:/data_ceshi/timg.jpg"

    img = cv2.imread(img_path)

    #Keep a copy   around
    img_orig = img.copy()

    #Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0,0,size[1],size[0])

    #Create an instance of Subdiv2d
    subdiv = cv2.Subdiv2D(rect)
    #Create an array of points
    points = []
    #Read in the points from a text file
    with open("E:/data_ceshi/points.txt") as file:
        for line in file:
            x,y = line.split()
            points.append((int(x),int(y)))
    #Insert points into subdiv
    for p in points:
        subdiv.insert(p)

        #Show animate
        if animate:
            img_copy = img_orig.copy()
            #Draw delaunay triangles
            draw_delaunay(img_copy,subdiv,(255,255,255))
            cv2.imshow(win_delaunary,img_copy)
            cv2.waitKey(100)

    #Draw delaunary triangles
    draw_delaunay(img,subdiv,(255,255,255))

    #Draw points
    for p in points:
        draw_point(img,p,(0,0,255))

    #Allocate space for Voroni Diagram
    img_voronoi = np.zeros(img.shape,dtype = img.dtype)

    #Draw Voonoi diagram
    draw_voronoi(img_voronoi,subdiv)

    #Show results
    cv2.imshow(win_delaunary,img)
    cv2.imshow(win_voronoi,img_voronoi)
    cv2.waitKey(0)