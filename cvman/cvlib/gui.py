# -*-coding:utf8-*-#
import cv2

# 创建窗口
# title: 窗口名
# flag:
WINDOW_NORMAL       = cv2.WINDOW_NORMAL
WINDOW_AUTOSIZE     = cv2.WINDOW_AUTOSIZE
WINDOW_OPENGL       = cv2.WINDOW_OPENGL
WINDOW_FULLSCREEN   = cv2.WINDOW_FULLSCREEN
WINDOW_FREERATIO    = cv2.WINDOW_FREERATIO
WINDOW_KEEPRATIO    = cv2.WINDOW_KEEPRATIO
WINDOW_GUI_EXPANDED = cv2.WINDOW_GUI_EXPANDED
WINDOW_GUI_NORMAL   = cv2.WINDOW_GUI_NORMAL
def namedWindow(title, flag = WINDOW_NORMAL):
    cv2.namedWindow(title, flag)

# 改变窗口大小
def resizeWindow(title, width, height):
    cv2.resizeWindow(title, width, height)

# 修改参数窗口
def createTrackbar(trackbarName, winName, value, count, onChange):
    cv2.createTrackbar(trackbarName, winName, value, count, onChange)

def getTrackbarPos(trackbarName, winName):
    return cv2.getTrackbarPos(trackbarName, winName)

# 关闭指定窗口
def destroyWindow(title):
    cv2.destroyWindow(title)

# 关闭所有窗口
def destroyAllWindows():
    cv2.destroyAllWindows()

# 等待按键输入
def waitKey(during=0):
    return cv2.waitKey(during) & 0xFF

# 读入文件
# filename : 文件位置
# flag : 
IMREAD_UNCHANGED           = cv2.IMREAD_UNCHANGED # (加载原图)
IMREAD_GRAYSCALE           = cv2.IMREAD_GRAYSCALE # (加载原图的灰度图)
IMREAD_COLOR               = cv2.IMREAD_COLOR     # (把原图作为RGB图加载)
IMREAD_ANYDEPTH            = cv2.IMREAD_ANYDEPTH
IMREAD_ANYCOLOR            = cv2.IMREAD_ANYCOLOR
IMREAD_LOAD_GDAL           = cv2.IMREAD_LOAD_GDAL
IMREAD_REDUCED_GRAYSCALE_2 = cv2.IMREAD_REDUCED_GRAYSCALE_2
IMREAD_REDUCED_COLOR_2     = cv2.IMREAD_REDUCED_COLOR_2
IMREAD_REDUCED_GRAYSCALE_4 = cv2.IMREAD_REDUCED_GRAYSCALE_4
IMREAD_REDUCED_COLOR_4     = cv2.IMREAD_REDUCED_COLOR_4
IMREAD_REDUCED_GRAYSCALE_8 = cv2.IMREAD_REDUCED_GRAYSCALE_8
IMREAD_REDUCED_COLOR_8     = cv2.IMREAD_REDUCED_COLOR_8
IMREAD_IGNORE_ORIENTATION  = cv2.IMREAD_IGNORE_ORIENTATION
def imread(filename, flag = IMREAD_UNCHANGED):
    return cv2.imread(filename, flag)

# 保存文件
# filename: 文件路径
# img : 文件内容
# params : [flag, value, ..., flagn, valuen]
#   flag:
IMWRITE_JPEG_QUALITY        =  cv2.IMWRITE_JPEG_QUALITY
IMWRITE_JPEG_PROGRESSIVE    =  cv2.IMWRITE_JPEG_PROGRESSIVE
IMWRITE_JPEG_OPTIMIZE       =  cv2.IMWRITE_JPEG_OPTIMIZE
IMWRITE_JPEG_RST_INTERVAL   =  cv2.IMWRITE_JPEG_RST_INTERVAL
IMWRITE_JPEG_LUMA_QUALITY   =  cv2.IMWRITE_JPEG_LUMA_QUALITY
IMWRITE_JPEG_CHROMA_QUALITY =  cv2.IMWRITE_JPEG_CHROMA_QUALITY
IMWRITE_PNG_COMPRESSION     =  cv2.IMWRITE_PNG_COMPRESSION
IMWRITE_PNG_STRATEGY        =  cv2.IMWRITE_PNG_STRATEGY
IMWRITE_PNG_BILEVEL         =  cv2.IMWRITE_PNG_BILEVEL
IMWRITE_PXM_BINARY          =  cv2.IMWRITE_PXM_BINARY
IMWRITE_EXR_TYPE            =  cv2.IMWRITE_EXR_TYPE
IMWRITE_WEBP_QUALITY        =  cv2.IMWRITE_WEBP_QUALITY
IMWRITE_PAM_TUPLETYPE       =  cv2.IMWRITE_PAM_TUPLETYPE
def imwrite(filename, img, flag=[]):
    return cv2.imwrite(filename, img, flag)

# 显示图片
def imshow(img, title = ''):
    cv2.imshow(title, img)


