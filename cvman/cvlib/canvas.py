# -*-coding:utf8-*-#
import cv2

# 画线
# linetype : 
FILLED  = cv2.FILLED
LINE_4  = cv2.LINE_4
LINE_8  = cv2.LINE_8
LINE_AA = cv2.LINE_AA # 防锯齿

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
# 中文字体支持有问题
FONT_HERSHEY_SIMPLEX        = cv2.FONT_HERSHEY_SIMPLEX
FONT_HERSHEY_PLAIN          = cv2.FONT_HERSHEY_PLAIN
FONT_HERSHEY_DUPLEX         = cv2.FONT_HERSHEY_DUPLEX
FONT_HERSHEY_COMPLEX        = cv2.FONT_HERSHEY_COMPLEX
FONT_HERSHEY_TRIPLEX        = cv2.FONT_HERSHEY_TRIPLEX
FONT_HERSHEY_COMPLEX_SMALL  = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_HERSHEY_SCRIPT_SIMPLEX = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_HERSHEY_SCRIPT_COMPLEX = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
FONT_ITALIC                 = cv2.FONT_ITALIC

def text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False):
    return cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)