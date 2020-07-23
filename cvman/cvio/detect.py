# -*-coding:utf8-*-#
import cv2

######
#  检测三大方向
#  D : 检测
#  D : 描述
#  M : 匹配
######

###############
# 运动目标检测 : 帧间差分法、背景减法、光流场法
#
# 1. 帧间差分法: 对时间上连续的两帧或三帧图像进行差分运算，不同帧对应的像素点相减，
#    判断灰度差的绝对值，当绝对值超过一定阈值时，即可判断为运动目标，从而实现目标的检测功能
# 2. 背景减法 : 利用不同图像的差分运算提取目标区域。
#    将当前帧图像与一个不断更新的背景模型相减，在差分图像中提取运动目标
# 3. 光流场法 : 首先计算图像中每一个像素点的运动向量，即建立整幅图像的光流场。
#    如果场景中没有运动目标，则图像中所有像素点的运动向量应该是连续变化的；
#    如果有运动目标，由于目标和背景之间存在相对运动，目标所在位置处的运动向量必然和邻域(背景)的运动向量不同，从而检测出运动目标
#############
# 角点检测
# Harris : 检测角点
# SIFT : 检测斑点
# SURF : 检测斑点
# FAST : 检测角点
# BRIFF : 检测斑点
# ORB : 
# HOG :
# LBP
# KAZE
# AKAZE
# BRISK
############

# Harris
# ======
# img : 8 bit 单通道图像 或 float32 图像
# blockSize : 邻域窗口边长(搜索角点的区域)
# ksize : 求导窗口边长, 为奇数值(3, 5, ...), 越大边缘越模糊
# k : 检测方程自由参数 [0.04, 0.06], k 越大, 检测精度越高
#
# 评价标准 R = r1r2 - k*((r1+r2)**2)
# e.g cornerHarris(gray, 2, 3, 0.04)
def cornerHarris(img, blockSize, ksize, k):
    return cv2.cornerHarris(img, blockSize, ksize, k)

# Shi-Tomasi
# ==========
# img: 8 bit 单通道图像
# maxCorners : 返回的最多角点数
# qualityLevel : 质量水平系数（小于1.0的正数，一般在0.01-0.1之间）
# minDistance : 最小距离，小于此距离的点忽略
# mask : mask=0的点忽略
# blockSize : 使用的邻域数
# useHarrisDetector : false ='Shi Tomasi metric'
# k : Harris角点检测时使用 (0.04)
#
# 评价标准 R = min(r1, r2)
# goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, None, mask, blockSize, useHarrisDetector, k
def goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance):
    corners = cv2.goodFeaturesToTrack(img, maxConners, qualityLevel, minDistance)
    return corners

# 自定义 Harris
# 输出: 每一个像素的 (r1, r2)
# 然后自定义 R = r1r2 - k*((r1+r2)**2)
def cornerEigenValsAndVecs(img, blockSize, ksize):
    return cv2.cornerEigenValsAndVecs(img, blockSize, ksize)
# 输出: 每一个像素的 (r1, r2)
# 然后自定义 R = min(r1, r2)
def cornerMinEigenVal(img, blockSize, ksize)
    return cv2.cornerMinEigenVal(img, blockSize, None, ksize)

# 亚像素
# 所有的角点都不是一个真正准确像素点 如(100.234, 5.789) -> (100, 5)
# 亚像素定位的方法
#   - 插值法
#   - 基于图像矩计算
#   - 曲线拟合方法(高斯曲面，多项式， 椭圆曲面)
def cornerSubPix(img, corners, winSize, zeroZone, criteria):
    corners = cv2.cornerSubPix(img, corners, winSize, zeroZone, criteria)
    return corners

# SIFT
# 特性: 旋转不变性、尺寸不变性
# 尺寸不变性可以通过尺寸空间滤波器实现（尺寸空间滤波器可以使用一些列具有不同方差的高斯卷积构成）
# LOG: 高斯拉普拉斯算子
# - 建立尺度空间，寻找极值 (取 DOG 近似 LOG)
# - 关键点(极值)定位
# - 关键点方向指定
# - 关键点描述子
# - 关键点匹配
def SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma):
    return cv.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

# SURF (快速 SIFT)
# 特点： 1 特征检测 2 尺度空间 (大小) 3 选择不变性(光照 旋转) 4 特征向量
# 工作原理:
#   - 选择图像中 POI (points of interest)
#   - 在不同的尺度空间发现关键点，非最大信号压制 (取 盒子滤波器 近似 LOG)
#   - 发现特征方法，旋转不变性要求
#   - 生成特征向量
#
# hessianThreshold : 检测阈值 [300 - 500]
# nOctaves : 表示尺度空间个数
# nOctaveLayers : 表示每个尺度层数
# upright : 表示计算选择不变性, 1 不计算
def SURF_create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=false, upright=false):
    return cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
def detectAndCompute(img, mask, descriptors, useProvidedKeypoints):
    keypoints, descriptors = cv2.Feature2D.detectAndCompute(img, mask, descriptors, useProvidedKeypoints)

# HOG (基于图像梯度直方图寻找特征)
# 步骤：
#   - 灰度图像转换
#   - 梯度计算
#   - 分网格的梯度方向直方图
#   - 块描述子
#   - 块描述子归一化  (屏蔽光照影响)
#   - 特征数据与检测窗口
#   - 匹配方法
#
#  gray = R * 0.3 + 0.59 * G + 0.11 * B
#
#  cv::HOGDescriptor::HOGDescriptor(
#     Size _winSize,
#     Size _blockSize,
#     Size _blockStride,
#     Size _cellSize,
#     int _nbins,
#     int _derivAperture = 1,
#     double _winSigma = -1,
#     HOGDescriptor::HistogramNormType _histogramNormType = HOGDescriptor::L2Hys,
#     double _L2HysThreshold = 0.2,
#     bool _gammaCorrection = false,
#     int _nlevels = HOGDescriptor::DEFAULT_NLEVELS,
#     bool _signedGradient = false 
#  )

# LBP (局部二值特征)
# 原理：
#  - 以中心点为阈值，跟周围像素做二值化
#  - 由二值化的值 确定 点、平坦区、线、边缘、角点

# 积分图
# 积分图: 图像上的任意一点的值是它左上角像素值的总和
#

# Haar 特征
#   - 高类间变异性
#   - 低类间变异性
#   - 局部强度差
#   - 不同尺度
#   - 计算效率高

# 动态目标检测
# learningRate: (模型的学习率)
#    -n: 负值表示自适应学习率
#     0: 学习率不更新
#    (0, 1): 学习率
#     1: 根据最后数据, 动态更新学习率
def backgroundSubtractorMOG(img, learningRate=-1):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(frame, learningRate)
    return fgmask
# 动态目标检测
# 是基于自适应混合高斯背景建模的背景减除法, 具有更好的抗干扰能力，特别是光照变化
# learningRate: (模型的学习率)
#    -n: 负值表示自适应学习率
#     0: 学习率不更新
#    (0, 1): 学习率
#     1: 根据最后数据, 动态更新学习率
def backgroundSubtractorMOG2(img, learningRate=-1):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(frame, learningRate)
    return fgmask

# 模板匹配
# img : 源图像
# templ : 待匹配图像
# method :
#   - cv.TM_SQDIFF
#   - cv.TM_SQDIFF_NORMED
#   - cv.TM_CCORR
#   - cv.TM_CCORR_NORMED
#   - cv.TM_CCOEFF
#   - cv.TM_CCOEFF_NORMED
# 输出的图像大小： (w+1, h+1)
def matchTemplate(img):
    return cv2.matchTemplate(img, templ, method)

# 暴力匹配
# normType : 使用的距离测试类型
#   - cv.NORM_L1
#   - cv.NORM_L2
#   - cv.NORM_HAMMING
#   - cv.NORM_HAMMING2
# crossCheck : 匹配是否严格
def BFMatcher_create(normType, crossCheck = false):
    return cv2.BFMatcher_create(normType, crossCheck = false)

# FLANN 匹配

# KAZE 局部匹配


# CascadeClassifier
# detectMultiScale
# scaleFactor : 每次图像尺寸减小的比例
# minNeighbors : 表示每一个目标至少要被检测到 N 次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
# flag : Not used
# minSize : 目标的最小尺寸
# maxSize : 目标的最大尺寸
# objects, numDetections = cv.CascadeClassifier.detectMultiScale2(image, scaleFactor=1.1, minNeighbors=3, flags, minSize, maxSize)
# 人脸生物特征 :
# - 两个眼睛之间的宽度大致等于一个眼睛的距离
# - 左右对称
# - 眼睛到嘴巴之间的距离大致在两个眼睛的宽度大小左右
# - 鼻子到嘴唇距离，大致等于两个嘴唇的厚度

#####
# 图像分割
#####
# K-means
#
# K : 聚类的最终数目 
# bestLabels : 
# criteria : 终止迭代的条件 (type, max_iter, epsilon)
#    - type
#       cv.TERM_CRITERIA_EPS 只有精确度 epsilon 满足时停止迭代
#       cv.TERM_CRITERIA_MAX_ITER 当迭代次数超过阈值时停止迭代
#       cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER 任何一个条件满足时停止迭代
#    - max_iter 表示最大迭代次数
#    - epsilon 精确度阈值
# attempts : 使用不同的起始标记来执行算法的次数。
# flag : 设置如何选择起始重心
#   - cv.KMEANS_PP_CENTERS
#   - cv.KMEANS_RANDOM_CENTERS
def kmeans((data, K, bestLabels, criteria, attempts, flags, centers):
    retval, bestLabels, centers = cv.kmeans(data, K, bestLabels, criteria, attempts, flags, centers)
    return retval, bestLabels, centers

# GMM 高斯混合模型
# 自动聚类， 抗干扰性强
# 
# samples : 输入的样本，一个单通道的矩阵。从这个样本中，进行高斯混和模型估计
# logLikelihoods : 可选项，输出一个矩阵，里面包含每个样本的似然对数值
# labels : 可选项，输出每个样本对应的标注
# probs: 可选项，输出一个矩阵，里面包含每个隐性变量的后验概率
def trainEM(samples, logLikelihoods, labels, probs):
    retval, logLikelihoods, labels, probs = cv.ml_EM.trainEM(samples, logLikelihoods, labels, probs)
    return retval, logLikelihoods, labels, probs

# 距离变换 (计算图像中每一个非零点距离离自己最近的零点的距离 创立小山头)
# distanceType:
#   - cv.DIST_USER    User defined distance
#   - cv.DIST_L1      distance = |x1-x2| + |y1-y2|
#   - cv.DIST_L2      the simple euclidean distance
#   - cv.DIST_C       distance = max(|x1-x2|,|y1-y2|)
#   - cv.DIST_L12     L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
#   - cv.DIST_FAIR    distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
#   - cv.DIST_WELSCH  distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
#      cv.DIST_HUBER  distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
# maskSize: 
#   - cv.DIST_MASK_3
#   - cv.DIST_MASK_5
#   - cv.DIST_MASK_PRECISE
def distanceTransform(img, distanceType, maskSize):
    dst = cv2.distanceTransform(img, distanceType, maskSize)
    return dst
def distanceTransformWithLabels(img, distanceType, maskSize, labels, labelType):
    dst, labels = cv2.distanceTransformWithLabels(img, distanceType, maskSize, None, labels, labelType)
    return dst, labels

# 分水岭:
# 把跟临近像素间的相似性作为重要的参考依据，从而将在空间位置上相近并且灰度值相近的像素点互相连接起来构成一个封闭的轮廓
# img : Input 8-bit 3-channel image.
# markers : Input/output 32-bit single-channel image (map) of markers. It should have the same size as image
def watershed(img, markers):
    return cv2.watershed(img, markers)

# BSM (背景消除)

# 光流跟踪
# 移动跟踪三要素 : 图像表示、外观模型、移动模型

# 稀疏光流 - KLT
# 前提条件 :
#   - 亮度恒定
#   - 近距离移动
#   - 空间一致性


# 稠密光流 - HF


# DNN 
# bufferProto : 网络结构模型描述文件 (.prototxt file)
# bufferModel : 网络结构模型 (.caffemodel file)
#
# 返回模型
def readNetFromCaffe():
    return cv2.dnn.readNetFromCaffe(bufferProto, bufferModel)

def blobFromImage(img, scalefactor, size, mean, swapRB, crop, ddepth):
	return cv2.dnn.blobFromImage(img, scalefactor, size, mean, swapRB, crop, ddepth)

# None = cv2.dnn_Net.setInput(blob, name, scalefactor, mean)
# retval = cv2.dnn_Net.forward(outputName)




