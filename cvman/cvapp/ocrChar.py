# -*-coding:utf8-*-#
import cv2
import sys
import operator
import imutils
import numpy as np

# module level variables
MIN_CONTOUR_AREA = 800
RESIZED_IMAGE_WIDTH = 24
RESIZED_IMAGE_HEIGHT = 40

def trainChars(trainImg):
    # read in training numbers image
    imgTrainingNumbers = cv2.imread(trainImg)

    if imgTrainingNumbers is None:
        print ("error: image not read from file \n\n")
        return

    # get grayscale image
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    # blur
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                      # input image
                                      255,                             # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,           # invert so foreground will be white, background will be black
                                      11,                              # size of a pixel neighborhood used to calculate threshold value
                                      2)                               # constant subtracted from the mean or weighted mean
    # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,          # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)    # compress horizontal, vertical, and diagonal segments and leave only their end points

    # declare empty numpy array, we will use this to write to file later
    # zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end
    intClassifications = []

    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,          # draw rectangle on original training image
                          (intX, intY),                # upper left corner
                          (intX + intW, intY + intH),  # lower right corner
                          (0, 0, 255),                 # red
                          2)                           # thickness

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)                     # get key press
            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:                                                                  # else if the char is in the list of chars we are looking for
                intClassifications.append(intChar)                                                          # append classification char to integer list of chars (we will convert to float later before writing to file)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
    
    # convert classifications list of ints to numpy array of floats
    fltClassifications = np.array(intClassifications, np.float32)
    # flatten numpy array of floats to 1d so we can write to file later
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print ("\n\ntraining complete !!\n")

    # write flattened images to file
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    # remove windows from memory
    cv2.destroyAllWindows()
    return

class ContourWithData():
    # member variables
    npaContour = None       # contour
    boundingRect = None     # bounding rect for contour
    intRectX = 0            # bounding rect top left corner x location
    intRectY = 0            # bounding rect top left corner y location
    intRectWidth = 0        # bounding rect width
    intRectHeight = 0       # bounding rect height
    fltArea = 0.0           # area of contour
    intrect =0
    imgWidth = 0
    imgHeight = 0

    # calculate bounding rect info
    def calculateRectTopLeftPointAndWidthAndHeight(self, cols):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        tolerance_factor = 18
        self.intrect = (((intY+intHeight) // tolerance_factor) * tolerance_factor) * cols + intX + intWidth

    # this is oversimplified, for a production grade program
    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        if (self.intRectWidth / self.imgWidth) > 0.2: return False
        if (self.intRectHeight / self.imgHeight) > 0.5: return False
        if (self.intRectHeight / self.imgHeight) < 0.2: return False 
        return True

def ocrChars(ocrImg):
    allContoursWithData = []
    validContoursWithData = []

    try:
        # read in training classifications
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        return

    try:
        # read in training images
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        return

    # reshape numpy array to 1d, necessary to pass to call to train
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    # instantiate KNN object
    kNearest = cv2.ml.KNearest_create()                   
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # read in testing numbers image
    # constant subtracted from the mean or weighted mean
    imgTestingNumbers = cv2.imread(ocrImg)                 # read in training numbers image

    if imgTestingNumbers is None:                          # if image was not read successfully
        print ("error: image not read from file \n\n"  )   # print error message to std out
        return                                             # and exit function (which exits program)
    imgTestingNumbers = imutils.resize(imgTestingNumbers, height=500)
    # imgTestingNumbers = imutils.rotate(imgTestingNumbers, angle=90)

    # get grayscale image
    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    # blur
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference
    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           #   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight(imgTestingNumbers.shape[1])                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        contourWithData.imgWidth = imgTestingNumbers.shape[1]
        contourWithData.imgHeight = imgTestingNumbers.shape[0]
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
            
   
    validContoursWithData.sort(key = operator.attrgetter("intrect","intRectX"))
    for contourWithData in validContoursWithData:                 # for all contours
        print(contourWithData.intrect,"==",contourWithData.intRectY,"__",contourWithData.intRectX)      # if so, append to valid contour list

    # sort contours from left to right
    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:
        # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))           # resize image, this will be more consistent for recognition and storage
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))    # flatten image into 1d numpy array
        npaROIResized = np.float32(npaROIResized)                                                 # convert from 1d numpy array of ints to 1d numpy array of floats
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)        # call KNN function find_nearest
        strCurrentChar = str(chr(int(npaResults[0][0])))                                          # get character from results
        strFinalString = strFinalString + strCurrentChar                                          # append current char to full string

    print ("\n" + strFinalString + "\n"   )                 # show the full string
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.waitKey(0)                                          # wait for user key press
    cv2.destroyAllWindows()                                 # remove windows from memory

    return

def run(img):
    # trainChars(img)
    ocrChars(img)

if __name__ == '__main__':
    main()