#-*-coding:utf8-*-#
import os
import numpy as np
import random

from cvman.cvio import gui as cvGui
from cvman.cvio import canvas as cvCanvas
from cvman.cvio import image as cvImage
from cvman.cvio.video import CVCapture, CVWriter
from cvman.cvmod.Subdiv import Subdiv
from cvman.cvmod import faceDetect as faceDetect

def runVideo(imgFile, outFile):
        #Define window names;
    winDelaunary = "Delaunay Triangulation"
    winVoronoi = "Voronoi Diagram"

    #Turn on animations while drawing triangles
    animate = True
    #Define colors for drawing
    delaunaryColor = (255, 255, 255)
    pointsColor = (0, 0, 255)

    #Read in the image
    img = cvGui.imread(imgFile)
    #Keep a copy around
    imgOrig = img.copy()

    #Create an instance of Subdiv2d
    size = cvImage.shape(img)
    rect = (0, 0, size[1], size[0])
    subdiv = Subdiv(rect)

    writer = CVWriter(outFile, 'H264', int(size[1]), int(size[0]), frameRate=10.0)

    points = faceDetect.detectLandMarkByDlib(imgFile)
    #Insert points into subdiv
    for p in points:
        subdiv.insertPoint(p)

        #Show animate
        if animate:
            imgCopy = imgOrig.copy()
            #Draw delaunay triangles
            subdiv.drawDelaunay(imgCopy, (255, 255, 176))
            # cvGui.imshow(imgCopy, winDelaunary)
            cvGui.waitKey(100)

            writer.write(imgCopy)
    writer.release()

    # #Draw delaunary triangles
    # subdiv.drawDelaunay(img, (255, 255, 255))

    # #Draw points
    # for p in points:
    #     cvCanvas.circle(img, p, 2, color, -1)

    # #Allocate space for Voroni Diagram
    # imgVoronoi = np.zeros(img.shape, dtype = img.dtype)
    # #Draw Voonoi diagram
    # subdiv.drawVoronoi(imgVoronoi)

def runImg(imgFile, outFile):
    #Define colors for drawing
    pointsColor = (0, 0, 255)

    #Read in the image
    img = cvGui.imread(imgFile)

    #Create an instance of Subdiv2d
    size = cvImage.shape(img)
    rect = (0, 0, size[1], size[0])
    subdiv = Subdiv(rect)

    points = faceDetect.detectLandMarkByDlib(imgFile)
    #Insert points into subdiv
    for p in points:
        subdiv.insertPoint(p)

    #Draw delaunary triangles
    subdiv.drawDelaunay(img, (255, 255, 255))

    #Draw points
    for p in points:
        cvCanvas.circle(img, p, 2, pointsColor, -1)
    cvGui.imwrite(outFile, img)

    # #Allocate space for Voroni Diagram
    # imgVoronoi = np.zeros(img.shape, dtype = img.dtype)
    # #Draw Voonoi diagram
    # subdiv.drawVoronoi(imgVoronoi)

if __name__ == '__main__':
    run()
