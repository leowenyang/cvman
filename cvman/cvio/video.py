# -*-coding:utf8-*-#
import cv2

class CVCapture():
    CAP_ANY          = cv2.CAP_ANY
    CAP_V4L          = cv2.CAP_V4L
    CAP_V4L2         = cv2.CAP_V4L2
    CAP_FIREWIRE     = cv2.CAP_FIREWIRE
    CAP_FIREWARE     = cv2.CAP_FIREWARE
    CAP_IEEE1394     = cv2.CAP_IEEE1394
    CAP_DC1394       = cv2.CAP_DC1394
    CAP_CMU1394      = cv2.CAP_CMU1394
    CAP_DSHOW        = cv2.CAP_DSHOW
    CAP_PVAPI        = cv2.CAP_PVAPI
    CAP_OPENNI       = cv2.CAP_OPENNI
    CAP_OPENNI_ASUS  = cv2.CAP_OPENNI_ASUS
    CAP_ANDROID      = cv2.CAP_ANDROID
    CAP_XIAPI        = cv2.CAP_XIAPI
    CAP_AVFOUNDATION = cv2.CAP_AVFOUNDATION
    CAP_GIGANETIX    = cv2.CAP_GIGANETIX
    CAP_MSMF         = cv2.CAP_MSMF
    CAP_WINRT        = cv2.CAP_WINRT
    CAP_INTELPERC    = cv2.CAP_INTELPERC
    CAP_OPENNI2      = cv2.CAP_OPENNI2
    CAP_OPENNI2_ASUS = cv2.CAP_OPENNI2_ASUS
    CAP_GPHOTO2      = cv2.CAP_GPHOTO2
    CAP_GSTREAMER    = cv2.CAP_GSTREAMER
    CAP_FFMPEG       = cv2.CAP_FFMPEG
    CAP_IMAGES       = cv2.CAP_IMAGES
    CAP_ARAVIS       = cv2.CAP_ARAVIS
    CAP_OPENCV_MJPEG = cv2.CAP_OPENCV_MJPEG
    CAP_INTEL_MFX    = cv2.CAP_INTEL_MFX
    def __init__(self, filename='', apiPreference=cv2.CAP_ANY):
        # VideoCapture(filename, apiPreference)
        # filename: 
        # apiPreference : 
        if filename == '':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
        
        # 判断视频是否打开成功
        if not self.cap.isOpened():
            if filename == '':
                self.cap.open(0)
            else:
                self.cap.open(filename, cv2.CAP_FFMPEG)

    def read(self):
        return self.cap.read()

    # 获取视频信息
    # propId :
    CAP_PROP_POS_MSEC             = cv2.CAP_PROP_POS_MSEC
    CAP_PROP_POS_FRAMES           = cv2.CAP_PROP_POS_FRAMES
    CAP_PROP_POS_AVI_RATIO        = cv2.CAP_PROP_POS_AVI_RATIO
    CAP_PROP_FRAME_WIDTH          = cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT         = cv2.CAP_PROP_FRAME_HEIGHT
    CAP_PROP_FPS                  = cv2.CAP_PROP_FPS
    CAP_PROP_FOURCC               = cv2.CAP_PROP_FOURCC
    CAP_PROP_FRAME_COUNT          = cv2.CAP_PROP_FRAME_COUNT
    CAP_PROP_FORMAT               = cv2.CAP_PROP_FORMAT
    CAP_PROP_MODE                 = cv2.CAP_PROP_MODE
    CAP_PROP_BRIGHTNESS           = cv2.CAP_PROP_BRIGHTNESS
    CAP_PROP_CONTRAST             = cv2.CAP_PROP_CONTRAST
    CAP_PROP_SATURATION           = cv2.CAP_PROP_SATURATION
    CAP_PROP_HUE                  = cv2.CAP_PROP_HUE
    CAP_PROP_GAIN                 = cv2.CAP_PROP_GAIN
    CAP_PROP_EXPOSURE             = cv2.CAP_PROP_EXPOSURE
    CAP_PROP_CONVERT_RGB          = cv2.CAP_PROP_CONVERT_RGB
    CAP_PROP_WHITE_BALANCE_BLUE_U = cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
    CAP_PROP_RECTIFICATION        = cv2.CAP_PROP_RECTIFICATION
    CAP_PROP_MONOCHROME           = cv2.CAP_PROP_MONOCHROME
    CAP_PROP_SHARPNESS            = cv2.CAP_PROP_SHARPNESS
    CAP_PROP_AUTO_EXPOSURE        = cv2.CAP_PROP_AUTO_EXPOSURE
    CAP_PROP_GAMMA                = cv2.CAP_PROP_GAMMA
    CAP_PROP_TEMPERATURE          = cv2.CAP_PROP_TEMPERATURE
    CAP_PROP_TRIGGER              = cv2.CAP_PROP_TRIGGER
    CAP_PROP_TRIGGER_DELAY        = cv2.CAP_PROP_TRIGGER_DELAY
    CAP_PROP_WHITE_BALANCE_RED_V  = cv2.CAP_PROP_WHITE_BALANCE_RED_V
    CAP_PROP_ZOOM                 = cv2.CAP_PROP_ZOOM
    CAP_PROP_FOCUS                = cv2.CAP_PROP_FOCUS
    CAP_PROP_GUID                 = cv2.CAP_PROP_GUID
    CAP_PROP_ISO_SPEED            = cv2.CAP_PROP_ISO_SPEED
    CAP_PROP_BACKLIGHT            = cv2.CAP_PROP_BACKLIGHT
    CAP_PROP_PAN                  = cv2.CAP_PROP_PAN
    CAP_PROP_TILT                 = cv2.CAP_PROP_TILT
    CAP_PROP_ROLL                 = cv2.CAP_PROP_ROLL
    CAP_PROP_IRIS                 = cv2.CAP_PROP_IRIS
    CAP_PROP_SETTINGS             = cv2.CAP_PROP_SETTINGS
    CAP_PROP_BUFFERSIZE           = cv2.CAP_PROP_BUFFERSIZE
    CAP_PROP_AUTOFOCUS            = cv2.CAP_PROP_AUTOFOCUS
    def get(self, propId):
        return self.cap.get(propId)

    def set(self, propId, value):
        self.cap.set(propId, value)

    def release(self):
        self.cap.release()

class CVWriter():
    # 输入输出视频的分辨率应该一致，否则，保存视频有问题
    def __init__(self, filename, strFourcc, width=1920, height=1080, frameRate=25.0):
        fourcc = cv2.VideoWriter_fourcc(*strFourcc)
        self.writer = cv2.VideoWriter(filename, fourcc, frameRate, (width, height))

    def write(self, frame):
        self.writer.write(frame)
    
    def set(self, propId, value):
        self.writer.set(propId, value)

    def release(self):
        self.writer.release()
