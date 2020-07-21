# -*-coding:utf8-*-#

import cv2
import imutils
import numpy as np

def run(imgFile):
    img = cv2.imread(imgFile)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 黑 : H -> [0, 180]  S -> [0, 255]  V -> [  0,  46]
    # 灰 : H -> [0, 180]  S -> [0,  43]  V -> [ 46, 220]
    # 白 : H -> [0, 180]  S -> [0,  30]  V -> [221, 255]
    # 红 : H -> [0, 10] [156, 180]  S -> [43,  255]  V -> [46, 255]
    # 橙 : H -> [11, 25]  S -> [43,  255]  V -> [46, 255]
    # 黄 : H -> [26, 34]  S -> [43,  255]  V -> [46, 255]
    # 绿 : H -> [35, 77]  S -> [43,  255]  V -> [46, 255]
    # 青 : H -> [78, 99]  S -> [43,  255]  V -> [46, 255]
    # 蓝 : H -> [100, 124]  S -> [43,  255]  V -> [46, 255]
    # 紫 : H -> [125, 155]  S -> [43,  255]  V -> [46, 255]
    height = hsv.shape[0]
    width  = hsv.shape[1]
    for x in range(height):
        for y in range(width):
            h = hsv[x, y][0]
            s = hsv[x, y][1]
            v = hsv[x, y][2]
            # 黑
            # if (h >= 0 and h <= 180) and (s >= 0 and s <= 255) and (v >= 0 and v <= 46):
            if (h >= 0 and h <= 180) and (s >= 0 and s <= 255) and (v >= 0 and v <= 60):
                hsv[x, y][0] = 0
                hsv[x, y][1] = 0
                hsv[x, y][2] = 0
            # 灰 1
            elif (h >= 0 and h <= 180) and (s >= 0 and s <= 43) and (v >= 46 and v <= 199):
                hsv[x, y][0] = 0
                hsv[x, y][1] = 0
                hsv[x, y][2] = 0
            # 灰 2
            elif (h >= 0 and h <= 180) and (s >= 0 and s <= 43) and (v >= 200 and v <= 220):
                hsv[x, y][0] = 255
                hsv[x, y][1] = 255
                hsv[x, y][2] = 255
            # 白
            elif (h >= 0 and h <= 180) and (s >= 0 and s <= 30) and (v >= 221 and v <= 255):
                hsv[x, y][0] = 255
                hsv[x, y][1] = 255
                hsv[x, y][2] = 255
            # 红
            # elif ((h >= 0 and h <= 10) or (h >= 156 and h <= 180)) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif ((h >= 4 and h <= 6) or (h >= 160 and h <= 176)) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 0
                hsv[x, y][1] = 0
                hsv[x, y][2] = 255
            # 橙
            # elif (h >= 11 and h <= 25) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif (h >= 13 and h <= 23) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 0
                hsv[x, y][1] = 127
                hsv[x, y][2] = 255
            # 黄
            # elif (h >= 26 and h <= 34) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif (h >= 28 and h <= 32) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 0
                hsv[x, y][1] = 255
                hsv[x, y][2] = 255
            # 绿
            # elif (h >= 35 and h <= 77) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif (h >= 40 and h <= 70) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 0
                hsv[x, y][1] = 255
                hsv[x, y][2] = 0
            # 青
            # elif (h >= 78 and h <= 99) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif (h >= 80 and h <= 97) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 255
                hsv[x, y][1] = 255
                hsv[x, y][2] = 0
            # 蓝
            # elif (h >= 100 and h <= 124) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif (h >= 102 and h <= 122) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 255
                hsv[x, y][1] = 0
                hsv[x, y][2] = 0
            # 紫
            # elif (h >= 125 and h <= 155) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
            elif (h >= 127 and h <= 153) and (s >= 43 and s <= 255) and (v >= 46 and v <= 255):
                hsv[x, y][0] = 255
                hsv[x, y][1] = 0
                hsv[x, y][2] = 139
            else:
                hsv[x, y] = img[x, y]
    cv2.imshow("img", img)
    cv2.imshow("result", hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()