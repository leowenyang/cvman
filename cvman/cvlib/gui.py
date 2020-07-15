# -*-coding:utf8-*-#
import cv2

# 创建窗口
# title: 窗口名
# flag:
#    - cv.WINDOW_NORMAL
#    - cv.WINDOW_AUTOSIZE
#    - cv.WINDOW_OPENGL
#    - cv.WINDOW_FULLSCREEN
#    - cv.WINDOW_FREERATIO
#    - cv.WINDOW_KEEPRATIO
#    - cv.WINDOW_GUI_EXPANDED
#    - cv.WINDOW_GUI_NORMAL
def namedWindow(title, flag=cv2.WINDOW_NORMAL):
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
#     - cv.IMREAD_UNCHANGED (加载原图)
#     - cv.IMREAD_GRAYSCALE (加载原图的灰度图)
#     - cv.IMREAD_COLOR     (把原图作为RGB图加载)
#     - cv.IMREAD_ANYDEPTH
#     - cv.IMREAD_ANYCOLOR
#     - cv.IMREAD_LOAD_GDAL
#     - cv.IMREAD_REDUCED_GRAYSCALE_2
#     - cv.IMREAD_REDUCED_COLOR_2
#     - cv.IMREAD_REDUCED_GRAYSCALE_4
#     - cv.IMREAD_REDUCED_COLOR_4
#     - cv.IMREAD_REDUCED_GRAYSCALE_8
#     - cv.IMREAD_REDUCED_COLOR_8
#     - cv.IMREAD_IGNORE_ORIENTATION
def imread(filename, flag=cv2.IMREAD_UNCHANGED):
    return cv2.imread(filename, flag)

# 保存文件
# filename: 文件路径
# img : 文件内容
# params : [flag, value, ..., flagn, valuen]
#   flag:
#    - cv.IMWRITE_JPEG_QUALITY
#    - cv.IMWRITE_JPEG_PROGRESSIVE
#    - cv.IMWRITE_JPEG_OPTIMIZE
#    - cv.IMWRITE_JPEG_RST_INTERVAL
#    - cv.IMWRITE_JPEG_LUMA_QUALITY
#    - cv.IMWRITE_JPEG_CHROMA_QUALITY
#    - cv.IMWRITE_PNG_COMPRESSION
#    - cv.IMWRITE_PNG_STRATEGY
#    - cv.IMWRITE_PNG_BILEVEL
#    - cv.IMWRITE_PXM_BINARY
#    - cv.IMWRITE_EXR_TYPE
#    - cv.IMWRITE_WEBP_QUALITY
#    - cv.IMWRITE_PAM_TUPLETYPE
#    - cv.IMWRITE_TIFF_RESUNIT
#    - cv.IMWRITE_TIFF_XDPI
#    - cv.IMWRITE_TIFF_YDPI
#    - cv.IMWRITE_TIFF_COMPRESSION
#    - cv.IMWRITE_JPEG2000_COMPRESSION_X1000
def imwrite(filename, img, flag=[]):
    return cv2.imwrite(filename, img, flag)

# 显示图片
def imshow(img):
    cv2.imshow('', img)


