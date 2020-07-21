# -*-coding:utf8-*-#

import cv2
import imutils
import numpy as np

import cvlib.core as core
import cvlib.canvas as canvas
import cvlib.image as image
from cvlib.video import CVCapture, CVWriter

def backgroundSubtractor2(videoFile):
    cap = cv2.VideoCapture(videoFile)
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    maxValue = 0
    while(1):
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 13)
        ret, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cnts = cv2.findContours(fgmask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        charCount = 0
        for c in cnts:
            area = cv2.contourArea(c)
            # print(area)
            if area > 500 and area < 3200:
                [intX, intY, intW, intH] = cv2.boundingRect(c)
                # draw rectangle around each contour as we ask user for input
                cv2.rectangle(frame,          # draw rectangle on original training image
                            (intX, intY),                # upper left corner
                            (intX + intW, intY + intH),  # lower right corner
                            (0, 0, 255),                 # red
                            2)                           # thickness
            if area >= 3200:
                charCount = charCount + 1
                [intX, intY, intW, intH] = cv2.boundingRect(c)
                # draw rectangle around each contour as we ask user for input
                cv2.rectangle(frame,          # draw rectangle on original training image
                            (intX, intY),                # upper left corner
                            (intX + intW, intY + intH),  # lower right corner
                            (0, 255, 0),                 # Green
                            2)                           # thickness
        print(maxValue, charCount)
        maxValue = maxValue if((maxValue > charCount) and (maxValue <= 10)) else charCount

        # cv2.imshow("", frame)
        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 640, 360)
        cv2.moveWindow("frame", 0, 0)
        cv2.imshow("frame", frame)

        cv2.namedWindow("output", 0)
        cv2.resizeWindow("output", 640, 360)
        cv2.moveWindow("output", 640, 0)
        cv2.imshow("output", fgmask)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    print("the result is ", maxValue)
    cap.release()
    cv2.destroyAllWindows()

    return maxValue

def run(videoFile):
    return backgroundSubtractor2(videoFile)

if __name__ == '__main__':
    main()

# using namespace std;
# using namespace cv;

# int main(int argc, const char** argv)
# {
#     VideoCapture cap;
#     bool update_bg_model = true;

#     //cap.open(0);
#     cap.open("People.mp4");

#     if( !cap.isOpened() )
#     {
#         printf("can not open camera or video file\n");
#         return -1;
#     }

#     namedWindow("image", WINDOW_AUTOSIZE);
#     namedWindow("foreground mask", WINDOW_AUTOSIZE);
#     namedWindow("foreground image", WINDOW_AUTOSIZE);
#     namedWindow("mean background image", WINDOW_AUTOSIZE);

#     BackgroundSubtractorMOG2 bg_model;//(100, 3, 0.3, 5);建立背景模型

#     Mat img, fgmask, fgimg;
#     int i = 0;

#     for(;;)
#     {
#         i++;
#         cap >> img;

#         if( img.empty() )
#             break;

#         img = img(Rect(40, 0, 300, img.rows));

#         if( fgimg.empty() )
#             fgimg.create(img.size(), img.type());

#         //更新模型
#         bg_model(img, fgmask, update_bg_model ? -1 : 0);
#         medianBlur(fgmask, fgmask, 13);
#         threshold(fgmask, fgmask, 150, 255, THRESH_BINARY);

#         Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

#         /*erode(fgmask, fgmask, element, Point(0, 0), 3);
#         dilate(fgmask, fgmask, element, Point(0, 0), 3);*/

#         Mat srcGrayImage = fgmask.clone();
#         vector<vector<Point>> vContours;
#         vector<Vec4i> vHierarchy;
#         findContours(srcGrayImage, vContours, vHierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));

#         int count = 0;
#         RNG rng(12345);

#         for (int i = 0; i < vContours.size(); i++)
#         {
#             double area = contourArea(vContours[i], false);

#             RotatedRect smallRect = minAreaRect(vContours[i]);
#             Point2f smallRect_center = smallRect.center;
#             float smallRect_width = smallRect.size.width;
#             float smallRect_height = smallRect.size.height;
#             float smallRect_angle = 0;

#             smallRect = RotatedRect(smallRect_center, Size2f(smallRect_height, smallRect_width), smallRect_angle);
#             Point2f P[4];
#             smallRect.points(P);

#             if (area>1000 && area < 4200)
#             {
#                 count++;
#                 for (int j = 0; j <= 3; j++)
#                 {
#                     line(img, P[j], P[(j + 1) % 4], Scalar(255, 0, 0), 2);
#                 }
#             }
#             if (area>4200 && area < 6000)
#             {
#                 count+=2;
#                 for (int j = 0; j <= 3; j++)
#                 {
#                     line(img, P[j], P[(j + 1) % 4], Scalar(255, 0, 0), 2);
#                 }
#             }

#         }

#         Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));//任意值
#         putText(img, int2str(count), Point(220, 40), FONT_HERSHEY_TRIPLEX, 1, color, 2);

#         fgimg = Scalar::all(0);
#         img.copyTo(fgimg, fgmask);

#         Mat bgimg;
#         bg_model.getBackgroundImage(bgimg);

#         imshow("image", img);

#         /*string windows_name = "Video/image_" + int2str(i);
#         string windows_name_ext = windows_name + ".jpg";
#         imwrite(windows_name_ext, img);*/

#         imshow("foreground mask", fgmask);


#         imshow("foreground image", fgimg);
#         if(!bgimg.empty())
#             imshow("mean background image", bgimg );

#         char k = (char)waitKey(1);
#         if( k == 27 ) break;
#         if( k == ' ' )
#         {
#             update_bg_model = !update_bg_model;
#             if(update_bg_model)
#                 printf("\t>背景更新(Background update)已打开\n");
#             else
#                 printf("\t>背景更新(Background update)已关闭\n");
#         }
#     }

#     return 0;
# }
