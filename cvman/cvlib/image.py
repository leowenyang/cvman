# -*-coding:utf8-*-#
import cv2
import numpy as np

# 归一化
# alpha: min
# beta: max
def normalize(img, alpha=1, beta=0, norm_type=cv2.NORM_L2, dtype=-1):
    return cv2.normalize(img, None, alpha, beta, norm_type, dtype)

## 亮度 vs 对比度
# 亮度调整: 图像像素强度整体变高/变低
# 对比度调整： 图像暗处像素强度变低，图像亮处像素变高，从而拉大中间某个区域范围的显示精度 (对比度有明暗分界点的指定)
#
# 对比公式： g(x) = a*f(x)+b (a, b 控制对比度，b 控制亮度) 

# 颜色转换
# color :
#   - cv.COLOR_BGR2BGRA
#   - cv.COLOR_RGB2RGBA
#   - cv.COLOR_BGRA2BGR
#   - cv.COLOR_RGBA2RGB
#   - cv.COLOR_BGR2RGBA
#   - cv.COLOR_RGB2BGRA
#   - cv.COLOR_RGBA2BGR
#   - cv.COLOR_BGRA2RGB
#   - cv.COLOR_BGR2RGB
#   - cv.COLOR_RGB2BGR
#   - cv.COLOR_BGR2GRAY
#   - cv.COLOR_BGR2HSV
#
# 注： HSV - 在opencv中， H:[0, 179], S:[0, 255], V:[0, 255]
COLOR_BGR2BGRA = cv2.COLOR_BGR2BGRA
COLOR_RGB2RGBA = cv2.COLOR_RGB2RGBA
COLOR_BGRA2BGR = cv2.COLOR_BGRA2BGR
COLOR_RGBA2RGB = cv2.COLOR_RGBA2RGB
COLOR_BGR2RGBA = cv2.COLOR_BGR2RGBA
COLOR_RGB2BGRA = cv2.COLOR_RGB2BGRA
COLOR_RGBA2BGR = cv2.COLOR_RGBA2BGR
COLOR_BGRA2RGB = cv2.COLOR_BGRA2RGB
COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
COLOR_RGB2BGR = cv2.COLOR_RGB2BGR
COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
COLOR_BGR2HSV = cv2.COLOR_BGR2HSV
def cvtColor(img, color):
    return cv2.cvtColor(img, color)

# return : height width channel
def shape(img):
    return img.shape

# 像素数
def size(img):
    return img.size

# 数据类型
def type(img):
    return img.dtype

# return : B G R
def split(img):
    return cv2.split(img)
def merge(b, g, r):
    return cv2.merge(b, g, r)

# dst(I) = saturate(src1(I) + src2(I))  if mask(I) <> 0
def add(img1, img2, mask):
    return cv2.add(img1, img2, None, mask)

# dst(I) = saturate(src1(I) * alpha + src2(I) ∗ beta + gamma)
def addWeighted(img1, alpha, img2, beta, gamma):
    return cv2.addWeighted(img1, alpha, img2, beta, gamma, dtype=-1)

# type :
#   - cv.THRESH_BINARY        => maxval if (img > thresh) else 0
#   - cv.THRESH_BINARY_INV    => maxval if (img < thresh) else 0 
#   - cv.THRESH_TRUNC         => thresh if (img > thresh) else img  (截断)
#   - cv.THRESH_TOZERO        => img if (img > thresh) else 0
#   - cv.THRESH_TOZERO_INV    => img if (img < threah) else 0
#   - cv.THRESH_MASK (不支持)
#   - cv.THRESH_OTSU         自动寻找阈值，需和上面配合，如 cv.THRESH_BINARY | cv.THRESH_OTSU
#   - cv.THRESH_TRIANGLE     同上 (适用于 单峰图像)
def threshold(img, thresh, maxval, type):
    return cv2.threshold(img, thresh, maxval, type)

# method :
#   - cv.ADAPTIVE_THRESH_MEAN_C 局部邻域块的平均值, 该算法是先求出块中的均值，再减去常数C
#   - cv.ADAPTIVE_THRESH_GAUSSIAN_C 局部邻域块的高斯加权和, 该算法是在区域中(x, y)周围的像素根据高斯函数按照他们离中心点的距离进行加权计算，再减去常数C
# type :
#   - cv.THRESH_BINARY
#   - cv.THRESH_BINARY_INV
def adaptiveThreshold(img, maxval, method, type, blockSize, C):
    return cv2.adaptiveThreshold(img, maxval, method, type, blockSize, C)

def inRange(img, lowb, upperb):
    return cv2.inRange(img, lowb, upperb)

def bitwiseAND(img1, img2, mask):
    return cv2.bitwise_and(img1, img2, mask=mask)

def bitwiseOR(img1, img2, mask):
    return cv2.bitwise_or(img1, img2, mask=mask)

def bitwiseNOT(img1, img2, mask):
    return cv2.bitwise_not(img1, img2, mask=mask)

def bitwiseXOR(img1, img2, mask):
    return cv2.bitwise_xor(img1, img2, mask=mask)

# interpolation :
#    - cv.INTER_NEAREST
#    - cv.INTER_LINEAR
#    - cv.INTER_CUBIC
#    - cv.INTER_AREA
#    - cv.INTER_LANCZOS4
#    - cv.INTER_LINEAR_EXACT
#    - cv.INTER_MAX
#    - cv.WARP_FILL_OUTLIERS
#    - cv.WARP_INVERSE_MAP
def resize(img, size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(img, size, None, fx, fy, interpolation)

# 图片移动
def move(img, dx, dy):
    height, width, channel = img.shape
    matShift = np.float32([ [1, 0, dx],
                            [0, 1, dy] ])
    return cv2.warpAffine(img, matShift, (width, height))

# 图片旋转
def rotate(img, center, angle, scale):
    height, width, channel = img.shape
    matShift = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, matShift, (width, height))

# 映射
# map1: x 映射表 CV_32FC1/CV_32FC2
# map2: y 映射表
# interpolation: 选择的插值方法, 常见线性插值、可选择立方等
def remap(img, map1, map2, interpolation):
    return cv2.remap(img, map1, map2, interpolation)

# 仿射变换
# srcPoints: 3 points in source image
# dstPoints: 3 points in dest image
def affineTransform(img, srcPoints, dstPoints):
    matShift = cv2.getAffineTransform(srcPoints, dstPoints)
    height, width, channel = img.shape
    return cv2.warpAffine(img, matShift, (width, height))

# 透视变换矩阵
# 透视关系变换的前提是 : 两个图像对应区域必须是同一平面
# srcPoints : 源点集 (至少 4 个点)
# dstPoints : 目的点集(一一对应)
# method : 
#   - 0 - a regular method using all the points, i.e., the least squares method
#   - cv.RANSAC - RANSAC-based robust method
#   - cv.LMEDS - Least-Median robust method
#   - cv.RHO - PROSAC-based robust method
def findHomograpy(srcPoints, dstPoints, method=0, ransacReprojThreshold=3, maxIters=2000, confidence=0.995):
    retval, mask = cv2.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold, None, maxIters, confidence)
    return retval, mask

def perspectiveTransform(img, m):
    return cv2.perspectiveTransform(img, m)

# 透视变换
# srcPoints: 4 points in source image
# dstPoints: 4 points in dest image
def perspectiveTransform2(img, srcPoints, dstPoints):
    matShift = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    height, width, channel = img.shape
    return cv2.warpPerspective(img, matShift, (width, height))

# 平均模糊 (取点周围数据的平均值)
#
# ksize: (x, y) 卷积核的大小 (x,y 为奇数值). 决定模糊程度, 值越大越模糊
# anchor: 取代元素的位置
# borderType: 
#    - cv2.BORDER_CONSTANT = 0, 
#    - cv2.BORDER_REPLICATE = 1, 
#    - cv2.BORDER_REFLECT = 2, 
#    - cv2.BORDER_WRAP = 3, 
#    - cv2.BORDER_REFLECT_101 = 4, 
#    - cv2.BORDER_TRANSPARENT = 5, 
#    - cv2.BORDER_REFLECT101 = BORDER_REFLECT_101, 
#    - cv2.BORDER_DEFAULT = BORDER_REFLECT_101, 
#    - cv2.BORDER_ISOLATED = 16 
def blur(img, ksize):
    return cv2.blur(img, ksize, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

# 高斯模糊
#
# ksize: (x, y) 卷积核的大小 (x,y 为奇数值). 决定模糊程度, 值越大越模糊
# sigmaX: 高斯方程 X 方向的标准差
# sigmaY: 高斯方程 Y 方向的标准差
# borderType: 
# 无法完全避免边缘信息的丢失，因为没有考虑像素值的不同
#
# 注: 
#  - sigmaX 在当 ksize 大小不为零时，直接从 ksize 大小计算
#  - sigmaY 在当 sigmX 不为零时, 从 sigmaX 计算， sigmaX 为零时，从 ksize 中计算 
def gaussianBlur(img, ksize, sigmaX, sigmaY=0):
    return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY=sigmaY, borderType=cv2.BORDER_DEFAULT)

# 中值模糊
#
# ksize: (x 卷积核的大小, 为奇数值). 决定模糊程度, 值越大越模糊
# 优点：中值对椒盐噪声有很好的抑制作用(处理黑白点)
# 缺点： 无法克服边缘信息丢失
def medianBlur(img, ksize):
    return cv2.medianBlur(img, ksize)

# 双边模糊
#
# d: 计算的半径，半径之内的像数都会被纳入计算，如果提供-1 则根据 sigmaSpace 参数取值
# sigmaColor: 高斯核中颜色值标准方差
# sigmaSpace: 高斯核中空间的标准方差
#
# 注: 指高斯双边滤波(用于美颜)。两像素差值太大的不予考虑，像素差值有一个阈值范围，
# 在这个范围的才考虑，其他原样输出，这个叫双边滤波
# 可以保留图像轮廓不变
def bilateralFilter(img, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace, borderType=cv2.BORDER_DEFAULT)

# 滤波函数
#
# ddepth: 输出图像的深度, -1 表示跟输入图像深度相同
# kernel: 卷积核或算子
# anchor: 卷积核中心位置
# delta: 卷积处理之后，每个像素值加上常量 delta
# borderType: 边缘处理
#  - cv.BORDER_CONSTANT       iiiiii|abcdefgh|iiiiiii with some specified i
#  - cv.BORDER_REPLICATE      aaaaaa|abcdefgh|hhhhhhh
#  - cv.BORDER_REFLECT        fedcba|abcdefgh|hgfedcb
#  - cv.BORDER_WRAP           cdefgh|abcdefgh|abcdefg
#  - cv.BORDER_REFLECT_101    gfedcb|abcdefgh|gfedcba
#  - cv.BORDER_TRANSPARENT    uvwxyz|abcdefgh|ijklmno
#  - cv.BORDER_REFLECT101     same as BORDER_REFLECT_101
#  - cv.BORDER_DEFAULT        same as BORDER_REFLECT_101
#  - cv.BORDER_ISOLATED       do not look outside of ROI
#
# 常见算子：
#  - Robert 算子
#          X:   [1  0]      Y: [ 0 1]
#               [0 -1]         [-1 0]
#  - Sobel 算子
#          X:   [-1 0 1]    Y: [-1 -2 -1]
#               [-2 0 2]       [ 0  0  0]
#               [-1 0 1]       [ 1  2  1]
#  - 拉普拉斯算子
#          [ 0 -1  0]
#          [-1  4 -1]
#          [ 0 -1  0]
def filter2D(img, ddepth, kernel, anchor, delta, borderType=cv2.BORDER_DEFAULT):
    return cv2.filter2D(img, ddepth, kernel, None, anchor, delta, borderType)

# borderType:
#  - cv.BORDER_CONSTANT       iiiiii|abcdefgh|iiiiiii with some specified i
#  - cv.BORDER_REPLICATE      aaaaaa|abcdefgh|hhhhhhh
#  - cv.BORDER_REFLECT        fedcba|abcdefgh|hgfedcb
#  - cv.BORDER_WRAP           cdefgh|abcdefgh|abcdefg
#  - cv.BORDER_REFLECT_101    gfedcb|abcdefgh|gfedcba
#  - cv.BORDER_TRANSPARENT    uvwxyz|abcdefgh|ijklmno
#  - cv.BORDER_REFLECT101     same as BORDER_REFLECT_101
#  - cv.BORDER_DEFAULT        same as BORDER_REFLECT_101
#  - cv.BORDER_ISOLATED       do not look outside of ROI
def makeBorder(img, top, bottom, left, right, borderType):
    return cv2.copyMakeBorder(img, top, bottom, left, right, borderType)

#####
# 边缘: 像素值发生跃迁的地方，是图像的显著特征
# 如何获取边缘: 对图像求一阶导数
#####

# 图像梯度
# 高斯平滑和微分求导
def sobel(img, dx, dy, ksize):
    return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize)
#  - scharr 算子
#          X:   [-3  0  3]    Y: [-3 -10 -3]
#               [-10 0 10]       [ 0   0  0]
#               [-2  0  3]       [ 3  10  3]
def scharr(img, dx, dy, scale, delta):
    return cv2.Scharr(img, cv2.CV_64F, dx, dy, None, scale, delta)
# 二阶导数，最大变化处为零，即边缘为零。因此，可以通过二阶导数求边缘
def laplacian(img, ksize):
    return cv2.Laplacian(img, cv2.CV_64F, None, ksize)
# 边缘检测
# 处理步骤：
#   - 高斯模糊
#   - 灰度转换
#   - 计算梯度
#   - 非最大信号抑制 (非边缘数值，就减少)
#   - 高低阈值输出二值图像
def canny(img, threshold1, threshold2):
    return cv2.Canny(img, threshold1, threshold2)

# 查找轮廓
# mode : 
#   - cv.RETR_EXTERNA 只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
#   - cv.RETR_LIST 检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立，没有等级关系
#   - cv.RETR_CCOMP 检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层
#   - cv.RETR_TREE 检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓
#   - cv.RETR_FLOODFILL
# method :(近似方法)
#   - cv.CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
#   - cv.CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留
#   - cv.CHAIN_APPROX_TC89_L1 使用teh-Chinl chain 近似算法
#   - cv.CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
def findContours(img, mode, method):
    image, contours, hierarchy = cv2.findContours(img, mode, method)
    return image, contours, hierarchy
def drawContours(img, contours, contourIdx, color, thickness=1, lineType=cv2.LINE_8):
    return cv2.drawContours(img, contours, contourIdx, color, thickness, lineType)

# 凸包
# 创建凸包
# points : 点集 来自 findContours
# clockwise : 顺时针方向
# returnPoints : 返回点个数
def convexHull(points, clockwise=False, returnPoints=True):
    return cv2.convexHull(points, None, clockwise, returnPoints)

# 矩
# 轮廓的最小外接矩形(无方向角)
def boundingRect(contours):
    x, y, w, h = cv2.boundingRect(contours)
    return x, y, w, h

# 轮廓的最小外接矩形(方向角)
def minAreaRect(contours):
    rect = cv2.minAreaRect(contours)
    return cv2.boxPoints(rect)

# 轮廓的最小外接圆
def minEnclosingCircle(contours):
    (x, y), radius = cv2.minEnclosingCircle(contours)
    return (x, y), radius

# 轮廓的最小外接椭圆
def fitEllipse(contours):
    return cv2.fitEllipse(cnt)

# 轮廓面积
def contourArea(contours, oriented=False):
    return cv2.contourArea(contour, oriented)

# 轮廓弧长
def arcLength(curve, closed):
    return cv2.arcLength(curve, closed)

# 多边形近似轮廓
# epsilon : 表示逼近曲率，越小表示相似逼近越厉害
# close : 是否闭合
def approxPolyDP(curve, epsilon, closed):
    return cv2.approxPolyDP(curve, epsilon, closed)

# 点、多边形检测
# contour: 输入轮廓
# pt: 点
# measureDist: 
#   true - 点到多边形最近的距离。 正值在多边形内，负值在多边形外，零在多边形边上
#   false - 返回 (+1, 0, -1)
def pointPolygonTest(contour, pt, measureDist):
    return cv2.pointPolygonTest(contour, pt, measureDist)

# 返回指定形状和尺寸的结构元素
# shape:
#     - MORPH_RECT 矩形
#     - MORPH_CROSS 交叉形
#     - MORPH_ELLIPSE 椭圆形
# anchor: 锚点位于中心点
def getStructuringElement(shape, ksize, anchor=(-1, -1)):
    return cv2.getStructuringElement(shape, ksize, anchor)

# 形态学四个基本操作: 腐蚀、膨胀、开、闭
# 腐蚀 (原理：卷积核覆盖下取图像的最小值)
def erode(img, kernel, anchor=(-1, -1), iterations=1):
    return cv2.erode(img, kernel, None, anchor, iterations)
# 膨胀 (原理：卷积核覆盖下取图像的最大值)
def dilate(img, kernel, anchor=(-1, -1), iterations=1):
    return cv2.dilate(img, kernel, None, anchor, iterations)
# 形态学操作
# op :
#  - cv.MORPH_ERODE     腐蚀
#  - cv.MORPH_DILATE    膨胀
#  - cv.MORPH_OPEN      开运算(先腐蚀再膨胀，去白点)
#  - cv.MORPH_CLOSE     闭运算(先膨胀在腐蚀，去黑点)
#  - cv.MORPH_GRADIENT  梯度(膨胀减去腐蚀)
#  - cv.MORPH_TOPHAT    礼帽(原图和开运算的差别)
#  - cv.MORPH_BLACKHA   黑帽(原图和闭运算的差别)  
def morphologyEx(img, op, kernel, anchor=(-1, -1), iterations=1):
    return cv2.morphologyEx(img, op, kernel, None, anchor, iterations)
#####
# 图像金字塔
# 高斯金字塔: 对图像进行降采样
#            1. 当前层高斯模糊
#            2. 删除当前层的偶数行和列
# 拉普拉斯金字塔: 用来重构一张图片根据它的上层降采样图片
# DOG(高斯不同): 不同参数高斯模糊后相减。得到图像内在特征，在灰度图像增强、角点检测中经常用到
#####

# 上采样(放大)
def pyrUp(img, size):
    return cv2.pyrUp(img, None, size)

# 降采样(缩小)
def pyrDown(img, size):
    return cv2.pyrDown(img, None, size)

# 傅里叶变换
def npfft(img):
    # 快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)
    # 默认结果中心点位置在左上角，转移到中间
    fshift = np.fft.fftshift(f)
    # 取绝对值：将复数变成实数
    # 取对数: 将数据变化到0-255
    return 20*np.log(np.abs(fshift))

# 霍夫直线变换
# 前提条件: 边缘检测已经完成
# 原理: 空间任意一点的直线方程可以用极坐标(p, 0)表示， 因此，两点共线的依据就是
#      极坐标曲线相交
#
# img: 8-bit 灰度图像
# rho: 生成极坐标时候的像素扫描步长
# theta: 生成极坐标时候的角度步长
# threshold: 阈值 只有获得足够交点的极坐标点才被看成直线
# return: [(rho, theta) ...] 
def houghLines(img, rho, theta, threshold):
    return cv2.HoughLines(img, rho, theta, threshold)
# img: 8-bit 灰度图像
# rho: 生成极坐标时候的像素扫描步长
# theta: 生成极坐标时候的角度步长
# threshold: 阈值 只有获得足够交点的极坐标点才被看成直线
# minLineLength: 最小直线长度
# maxLineGap: 最大间隔
# return: [(x1, y1, x2, y2) ...]
def houghLinesP(img, rho, theta, threshold, minLineLength=0, maxLineGap=0):
    return cv2.HoughLinesP(img, rho, theta, threshold, None, minLineLength, maxLineGap)
# 霍夫圆形变换
# 噪声敏感，要做中值滤波
# 边缘检测后
# img: 8-bit 灰度图像
# dp: 
# minDist: 最小距离可以区分两个圆
# minRadius: 圆的最小半径
# maxRadius: 圆的最大半径
def houghCircles(img, dp, minDist, param1, param2, minRadius, maxRadius):
    return cv2.HoughCircles(img, dp, minDist, param1, param2, minRadius, maxRadius)

#####
# 直方图
# 直方图的属性:
#   - dims 表示维度
#   - bins 表示在维度中子区域大小的划分 ( X 值 )
#   - range 表示值的范围 ( X 范围 )
#####

# 统计直方图
# imgs: 图像列表
# channels: 图像通道数列表
# mask: 输入mask 可选 不用
# histSize: 直方图级数列表 (bins 个数)
# ranges: 值域范围 [0, 256]
def calcHist(imgs, channels, histSize, ranges):
    return cv2.calcHist(imgs, channels, None, histSize, ranges)

# 直方图均衡化
# img : 8-bit 单通道图像
def equalizeHist(img):
    return cv2.equalizeHist(img)

# 直方图反向投影
# 输出与输入图像同样大小的图像，其中的每一个像素值代表了输入图像上对应点属于目标对象的概率
# imgs: 图像列表
# channels: 图像通道数列表
# hist: 查找目标物体的直方图
# ranges: 值域范围 [0, 256]
def calcBackProject(imgs, channels, hist, ranges):
    return cv2.calcBackProject(imgs, channels, hist, ranges, scale=1)

# 直方图比较
# H1 : 直方图 1
# H2 : 直方图 2
# method : (评估相似性的数学方法)
#   - cv.HISTCMP_CORREL
#   - cv.HISTCMP_CHISQR
#   - cv.HISTCMP_INTERSECT
#   - cv.HISTCMP_BHATTACHARYYA
#   - cv.HISTCMP_HELLINGER
#   - cv.HISTCMP_CHISQR_ALT
#   - cv.HISTCMP_KL_DIV
def compareHist():
    return cv2.compareHist(H1, H2, method)

