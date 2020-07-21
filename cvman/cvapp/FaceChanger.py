#coding=utf-8  
import cv2
import dlib
import os
import numpy as np
import glob
from color_transfer import color_transfer

class TooManyFaces(Exception):
    pass

class NoFace(Exception):
    pass

class FaceChanger(object):
    def __init__(self):
        print('Starting your FaceChanger...')

        # some parameters
        self.SCALE_FACTOR = 1 
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                             self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)

        # Points from the second image to overlay on the first. The convex hull of each
        # element will be overlaid.
        self.OVERLAY_POINTS = [
              self.LEFT_EYE_POINTS
            + self.RIGHT_EYE_POINTS
            + self.LEFT_BROW_POINTS
            + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS,
        ]

        self.COLOUR_CORRECT_BLUR_FRAC = 0.6

        # load in models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./dlib/shape_predictor_68_face_landmarks.dat')

        self.image1 = None
        self.image2 = None
        self.landmarks1 = None
        self.landmarks2 = None

    def loadImages(self, image1, image2):
        self.image1 = image1
        self.image2 = image2

        self.landmarks1 = self.getLandmark(self.image1)
        if len(self.landmarks1) == 0:
            return False
        self.landmarks2 = self.getLandmark(self.image2)
        if len(self.landmarks2) == 0:
            return False

        return True

    def run(self, showProcedure=False, saveResult=True):
        if self.image1 is None or self.image2 is None:
            print('You need to load two images first.')
            return

        if showProcedure == True:
            print('Showing the procedure.Press any key to continue your process.')
            cv2.imshow("1", self.image1)
            cv2.waitKey(0)
            cv2.imshow("2", self.image2)
            cv2.waitKey(0)
        # tranfer = color_transfer(self.image1, self.image2)
        # cv2.imshow("6.1", tranfer)
        # cv2.waitKey(0)

        M = self.transformationFromPoints(self.landmarks1[self.ALIGN_POINTS], self.landmarks2[self.ALIGN_POINTS])

        mask = self.getFaceMask(self.image2, self.landmarks2)
        if showProcedure == True:
            cv2.imshow("3", mask)
            cv2.waitKey(0)

        warpedMask = self.warpImage(mask, M, self.image1.shape)
        if showProcedure == True:
            cv2.imshow("4", warpedMask)
            cv2.waitKey(0)

        combinedMask = np.max([self.getFaceMask(self.image1, self.landmarks1), warpedMask], axis=0)
        if showProcedure == True:
            cv2.imshow("5", combinedMask) 
            cv2.waitKey(0)

        warpedImg2 = self.warpImage(self.image2, M, self.image1.shape)
        if showProcedure == True:
            cv2.imshow("6", warpedImg2)
            cv2.waitKey(0)

        warpedCorrectedImg2 = self.correctColours(self.image1, warpedImg2, self.landmarks1)
        warpedCorrectedImg2Temp = np.zeros(warpedCorrectedImg2.shape, dtype=warpedCorrectedImg2.dtype)
        cv2.normalize(warpedCorrectedImg2, warpedCorrectedImg2Temp, 0, 1, cv2.NORM_MINMAX)
        if showProcedure == True:
            cv2.imshow("7", warpedCorrectedImg2Temp)
            cv2.waitKey(0)

        output = self.image1 * (1.0 - combinedMask) + warpedCorrectedImg2 * combinedMask
        outputShow = np.zeros(output.shape, dtype=output.dtype)
        cv2.normalize(output, outputShow, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)

        if showProcedure == True:
            cv2.imshow("8", outputShow.astype(outputShow.dtype))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if saveResult is True:
            cv2.imwrite("output.jpg", output)

        return output

    def getLandmark(self, image):
        face_rect = self.detector(image, 1)

        if len(face_rect) > 1:
            print('Too many faces.We only need no more than one face.')
            return []
        elif len(face_rect) == 0:
            print('No face.We need at least one face.')
            return []
        else:
            print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect[0].left(), face_rect[0].top(), face_rect[0].right(), face_rect[0].bottom()))
            # box = face_rect[0]
            # shape = predictor(image, box)
            # return np.matrix([[p.x, p.y] for p in shape.parts()])
            return np.matrix([[p.x, p.y] for p in self.predictor(image, face_rect[0]).parts()])

    def transformationFromPoints(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def warpImage(self, image, M, dshape):
        outputImage = np.zeros(dshape, dtype=image.dtype)
        cv2.warpAffine(image, M[:2], (dshape[1], dshape[0]), dst=outputImage, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
        return outputImage

    def correctColours(self, im1, im2, landmarks1):
        blurAmount = self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                  np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
                                  np.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blurAmount = int(blurAmount)
        if blurAmount % 2 == 0:
            blurAmount += 1
        im1Blur = cv2.GaussianBlur(im1, (blurAmount, blurAmount), 0)
        im2Blur = cv2.GaussianBlur(im2, (blurAmount, blurAmount), 0)

        # Avoid divide-by-zero errors.
        im2Blur += (128 * (im2Blur <= 1.0)).astype(im2Blur.dtype)

        return (im2.astype(np.float64) * im1Blur.astype(np.float64) /
                                                    im2Blur.astype(np.float64))

    def drawConvexHull(self, img, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(img, points, color)

    def getFaceMask(self, img, landmarks):
        img = np.zeros(img.shape[:2], dtype=np.float64)
        for group in self.OVERLAY_POINTS:
            self.drawConvexHull(img, landmarks[group], color=1)

        img = np.array([img, img, img]).transpose((1, 2, 0)) 

        img = (cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        img = cv2.GaussianBlur(img, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return img

if __name__ == '__main__':
    from FaceChanger import *
    fc = FaceChanger()
    fc.load_images('ibrahimovic.jpg', 'pique.jpg')
    fc.run(showProcedure=True)