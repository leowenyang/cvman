# -*-coding:utf8-*-#
import cv2

# 画线
# linetype : 
#    - cv.FILLED
#    - cv.LINE_4
#    - cv.LINE_8
#    - cv.LINE_AA (防锯齿)
def line(img, pt1, pt2, color, thickness=1, linetype=cv2.LINE_8, shift=0):
    return cv2.line(img, pt1, pt2, color, thickness, linetype, shift)

def polylines(img, pts, isClosed, color, thickness=1, lineType=cv2.LINE_8, shift=0):
    return cv2.line(img, pts, isClosed, color, thickness, lineType, shift)

def circle(img, center, radius, color, thickness=4, linetype=cv2.LINE_8, shift=0):
    return cv2.circle(img, center, radius, color, thickness, linetype, shift)

def rectangle(img, pt1, pt2, color, thickness=1, linetype=cv2.LINE_8, shift=0):
    return cv2.rectangle(img, pt1, pt2, color, thickness, linetype, shift)

def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, linetype=cv2.LINE_8, shift=0):
    return cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, linetype, shift)

# fontFace :
#    - cv.FONT_HERSHEY_SIMPLEX
#    - cv.FONT_HERSHEY_PLAIN
#    - cv.FONT_HERSHEY_DUPLEX
#    - cv.FONT_HERSHEY_COMPLEX
#    - cv.FONT_HERSHEY_TRIPLEX
#    - cv.FONT_HERSHEY_COMPLEX_SMALL
#    - cv.FONT_HERSHEY_SCRIPT_SIMPLEX
#    - cv.FONT_HERSHEY_SCRIPT_COMPLEX
#    - cv.FONT_ITALIC
# 中文字体支持有问题
def text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False):
    return cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)