# -*-coding:utf8-*-#
import cv2

from cvman.cvio import gui as cvGui
from cvman.cvio import canvas as cvCanvas
from cvman.cvio import image as cvImage

class Subdiv():
    def __init__(self, rect):
        self.subdiv = cv2.Subdiv2D(rect)

    def insertPoint(self, point):
        self.subdiv.insert(point)
    
    # Check if a point is insied a rectangle
    def rectContains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return  False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def getTriangleList(self):
        return self.subdiv.getTriangleList()

    def drawDelaunay(self, img, color):
        trangleList = self.getTriangleList()
        size = img.shape
        r = (0, 0, size[1], size[0])
        for t in trangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if (self.rectContains(r, pt1) 
                and self.rectContains(r, pt2) 
                and self.rectContains(r, pt3)):
                cvCanvas.line(img, pt1, pt2, color, 2)
                cvCanvas.line(img, pt2, pt3, color, 2)
                cvCanvas.line(img, pt3, pt1, color, 2)

    def getVoronoiFacetList(self):
        return self.subdiv.getVoronoiFacetList([])

    def drawVoronoi(self, img):
        (facets, centers) = self.getVoronoiFacetList()

        for i in range(0, len(facets)):
            ifacetArr = []
            for f in facets[i]:
                ifacetArr.append(f)

            ifacet = np.array(ifacetArr, np.int)
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            cvCanvas.fillConvexPoly(img, ifacet, color)
            ifacets = np.array([ifacet])
            cvCanvas.polylines(img, ifacets, True, (0,0,0), 1)
            cvCanvas.circle(img, (centers[i][0], centers[i][1]), 3, (0,0,0))
