#-*-coding:utf8-*-#
import os
import cv2
import dlib
import numpy as np
import random

from cvlib.video import CVCapture, CVWriter

# Check if a point is insied a rectangle
def getLanmarks(imgFile):
    predictorPath  = "./dlib/shape_predictor_68_face_landmarks.dat"
    txtPath = "./data/points.txt"

    f = open(txtPath,'w+')
    detector = dlib.get_frontal_face_detector()
    #相撞
    predicator = dlib.shape_predictor(predictorPath)
    img1 = cv2.imread(imgFile)

    dets = detector(img1, 1)
    print("Number of faces detected : {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}  left:{}  Top: {} Right {}  Bottom {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()
        ))
        lanmarks = [[p.x, p.y] for p in predicator(img1, d).parts()]
        for idx, point in enumerate(lanmarks):
            f.write(str(point[0]))
            f.write("\t")
            f.write(str(point[1]))
            f.write('\n')

# Check if a point is insied a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return  False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw a point
def drawPoint(img, p, color):
    cv2.circle(img, p, 2, color, -1)

# Draw delaunay triangles
def drawDelaunay(img, subdiv, delaunayColor):
    trangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in  trangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if (rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3)):
            cv2.line(img, pt1, pt2, delaunayColor, 2)
            cv2.line(img, pt2, pt3, delaunayColor, 2)
            cv2.line(img, pt3, pt1, delaunayColor, 2)

# Draw voronoi diagram
def drawVoronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacetArr = []
        for f in facets[i]:
            ifacetArr.append(f)

        ifacet = np.array(ifacetArr, np.int)
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.fillConvexPoly(img, ifacet, color)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0,0,0), 1)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0,0,0))

def run(imgFile):
    getLanmarks(imgFile)

    #Define window names;
    winDelaunary = "Delaunay Triangulation"
    winVoronoi = "Voronoi Diagram"

    #Turn on animations while drawing triangles
    animate = True

    #Define colors for drawing
    delaunaryColor = (255, 255, 255)
    pointsColor = (0, 0, 255)

    #Read in the image
    img = cv2.imread(imgFile)
    #Keep a copy around
    imgOrig = img.copy()

    #Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    #Create an instance of Subdiv2d
    subdiv = cv2.Subdiv2D(rect)
    #Create an array of points
    points = []

    writer = CVWriter('./data/show.mp4', 'H264', int(size[1]), int(size[0]), frameRate=10.0)

    #Read in the points from a text file
    with open("./data/points.txt") as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))
    #Insert points into subdiv
    for p in points:
        subdiv.insert(p)

        #Show animate
        if animate:
            imgCopy = imgOrig.copy()
            #Draw delaunay triangles
            drawDelaunay(imgCopy, subdiv, (255, 255, 176))
            cv2.imshow(winDelaunary, imgCopy)
            cv2.waitKey(100)

            writer.write(imgCopy)
    writer.release()

    #Draw delaunary triangles
    drawDelaunay(img, subdiv, (255, 255, 255))

    #Draw points
    for p in points:
        drawPoint(img, p, (0,0,255))
    cv2.imwrite("./data/show.jpg", img)

    #Allocate space for Voroni Diagram
    imgVoronoi = np.zeros(img.shape, dtype = img.dtype)

    #Draw Voonoi diagram
    drawVoronoi(imgVoronoi, subdiv)

    #Show results
    cv2.imshow(winDelaunary, img)
    cv2.imshow(winVoronoi, imgVoronoi)
    cv2.waitKey(0)

if __name__ == '__main__':
    run()
