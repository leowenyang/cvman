# -*-coding:utf8-*-#
import cv2

class CVCapture():
    def __init__(self, filename='', apiPreference=cv2.CAP_ANY):
        # VideoCapture(filename, apiPreference)
        # filename: 
        # apiPreference : 
        #   - cv.CAP_ANY
        #   - cv.CAP_V4L
        #   - cv.CAP_V4L2
        #   - cv.CAP_FIREWIRE
        #   - cv.CAP_FIREWARE
        #   - cv.CAP_IEEE1394
        #   - cv.CAP_DC1394
        #   - cv.CAP_CMU1394
        #   - cv.CAP_DSHOW
        #   - cv.CAP_PVAPI
        #   - cv.CAP_OPENNI
        #   - cv.CAP_OPENNI_ASUS
        #   - cv.CAP_ANDROID
        #   - cv.CAP_XIAPI
        #   - cv.CAP_AVFOUNDATION
        #   - cv.CAP_GIGANETIX
        #   - cv.CAP_MSMF
        #   - cv.CAP_WINRT
        #   - cv.CAP_INTELPERC
        #   - cv.CAP_REALSENSE
        #   - cv.CAP_OPENNI2
        #   - cv.CAP_OPENNI2_ASUS
        #   - cv.CAP_GPHOTO2
        #   - cv.CAP_GSTREAMER
        #   - cv.CAP_FFMPEG
        #   - cv.CAP_IMAGES
        #   - cv.CAP_ARAVIS
        #   - cv.CAP_OPENCV_MJPEG
        #   - cv.CAP_INTEL_MFX
        #   - cv.CAP_XINE
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
    #     - cv.CAP_PROP_POS_MSEC
    #     - cv.CAP_PROP_POS_FRAMES
    #     - cv.CAP_PROP_POS_AVI_RATIO
    #     - cv.CAP_PROP_FRAME_WIDTH
    #     - cv.CAP_PROP_FRAME_HEIGHT
    #     - cv.CAP_PROP_FPS
    #     - cv.CAP_PROP_FOURCC
    #     - cv.CAP_PROP_FRAME_COUNT
    #     - cv.CAP_PROP_FORMAT
    #     - cv.CAP_PROP_MODE
    #     - cv.CAP_PROP_BRIGHTNESS
    #     - cv.CAP_PROP_CONTRAST
    #     - cv.CAP_PROP_SATURATION
    #     - cv.CAP_PROP_HUE
    #     - cv.CAP_PROP_GAIN
    #     - cv.CAP_PROP_EXPOSURE
    #     - cv.CAP_PROP_CONVERT_RGB
    #     - cv.CAP_PROP_WHITE_BALANCE_BLUE_U
    #     - cv.CAP_PROP_RECTIFICATION
    #     - cv.CAP_PROP_MONOCHROME
    #     - cv.CAP_PROP_SHARPNESS
    #     - cv.CAP_PROP_AUTO_EXPOSURE
    #     - cv.CAP_PROP_GAMMA
    #     - cv.CAP_PROP_TEMPERATURE
    #     - cv.CAP_PROP_TRIGGER
    #     - cv.CAP_PROP_TRIGGER_DELAY
    #     - cv.CAP_PROP_WHITE_BALANCE_RED_V
    #     - cv.CAP_PROP_ZOOM
    #     - cv.CAP_PROP_FOCUS
    #     - cv.CAP_PROP_GUID
    #     - cv.CAP_PROP_ISO_SPEED
    #     - cv.CAP_PROP_BACKLIGHT
    #     - cv.CAP_PROP_PAN
    #     - cv.CAP_PROP_TILT
    #     - cv.CAP_PROP_ROLL
    #     - cv.CAP_PROP_IRIS
    #     - cv.CAP_PROP_SETTINGS
    #     - cv.CAP_PROP_BUFFERSIZE
    #     - cv.CAP_PROP_AUTOFOCUS
    #     - cv.CAP_PROP_SAR_NUM
    #     - cv.CAP_PROP_SAR_DEN
    #     - cv.CAP_PROP_BACKEND
    #     - cv.CAP_PROP_CHANNEL
    #     - cv.CAP_PROP_AUTO_WB
    #     - cv.CAP_PROP_WB_TEMPERATURE
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
        self.cap.set(propId, value)

    def release(self):
        self.writer.release()
