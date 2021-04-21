from __future__ import print_function
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab
import random
import json
import os
from PIL import Image
import numpy.ma as npm
from skimage import measure,draw
import cv2
import math
from shapely.geometry import Polygon
import glob
zoom = 20
tileSize = 300
initialResolution = 2 * math.pi * 6378137 / tileSize
originShift = 2 * math.pi * 6378137 / 2.0
earthc = 6378137 * 2 * math.pi
factor = math.pow(2, zoom)
map_width = 300 * (2 ** zoom)

def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #颜色转换（参数一是需要转换的图片，参数二是转换成何种格式）
    #cv2.COLOR_BGR2YCR_CB将BGR格式转换成ycr格式
    channels = cv2.split(ycrcb)#分离通道
    cv2.equalizeHist(channels[0], channels[0])#直方图均衡化，[0]灰度图像
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def solar_panel_params():
    panel_lens = input("Number of panels together: ")
    # panel_wids = input("Number of panels in breadth: ")
    panel_wids = 1
    length_s = input("Enter length of panel in mm: ")
    width = input("Enter width of panel in mm: ")
    angle = input("Rotation Angle for Solar Panels: ")
    return panel_lens, panel_wids, length_s, width, angle


def rotation(center_x, center_y, points, ang):
    angle = ang * math.pi / 180
    rotated_points = []
    for p in points:
        x, y = p
        x, y = x - center_x, y - center_y
        x, y = (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))
        x, y = x + center_x, y + center_y
        rotated_points.append((x, y))
    return rotated_points


def createLineIterator(P1, P2, img):
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(float) / dY.astype(float)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(float) / dX.astype(float)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer

def panel_rotation(panels_series, solar_roof_area):

    # high_reso = cv2.pyrUp(solar_roof_area)
    high_reso = solar_roof_area
    # rows,cols = high_reso.shape    #rows=362 cols=362 图solar_roof_area的行列数
    # rows,cols = solar_roof_area.shape    #rows=362 cols=362 图solar_roof_area的行列数
    size = high_reso.shape  # rows=362 cols=362 图solar_roof_area的行列数
    rows = size[0]
    cols = size[1]

    high_reso_new = cv2.pyrUp(new_image)

    for _ in range(panels_series - 2):
        for col in range(0, cols, l + 1):
            for row in range(0, rows, w + 1):

                # Rectangular Region of interest for solar panel area
                #PW为一组板子的宽，w为一块板子的宽
                solar_patch = high_reso[row:row + (w + 1) * pw + 1, col:col + ((l * pl) + 3)]
                # print('solar_patch类型：',type(solar_patch))
                # r, c = solar_patch.shape#太阳能电板图的行r列c数
                size_solar_patch = solar_patch.shape
                r = size_solar_patch[0]
                c = size_solar_patch[1]

                # Rotation of rectangular patch according to the angle provided
                patch_rotate = np.array([[col, row], [c + col, row], [c + col, r + row], [col, r + row]], np.int32)
                rotated_patch_points = rotation((col + c) / 2, row + r / 2, patch_rotate, solar_angle)
                #存储每个板子的坐标点patch_location
                # patch_location_list[i++]=rotated_patch_points

                rotated_patch_points = np.array(rotated_patch_points, np.int32)
                # print('坐标', type(rotated_patch_points))
                # temp=np.append(temp,rotated_patch_points)
                # temp.append(rotated_patch_points)


                # 剔除超出图像大小的点
                if (rotated_patch_points > 0).all():
                    solar_polygon = Polygon(rotated_patch_points)#传入板子位置参数（x,y）,画不规则四边形 (坐标按照一维数组从左上角→左下角→右下角逆时针旋转到左上角）
                    polygon_points = np.array(solar_polygon.exterior.coords, np.int32)

                    # Appending points of the image inside the solar area to check the intensity
                    patch_intensity_check = []

                    # Point polygon test for each rotated solar patch area
                    for j in range(rows):
                        for k in range(cols):
                            if cv2.pointPolygonTest(polygon_points, (k, j), False) == 1:
                                patch_intensity_check.append(high_reso[j, k])
                                patch_intensity_check.append(high_reso[j, k])

                    # Check for the region available for Solar Panels
                    if np.mean(patch_intensity_check) == 255:

                        # Moving along the length of line to segment solar panels in the patch
                        solar_line_1 = createLineIterator(rotated_patch_points[0], rotated_patch_points[1], high_reso)
                        solar_line_1 = solar_line_1.astype(int)
                        solar_line_2 = createLineIterator(rotated_patch_points[3], rotated_patch_points[2], high_reso)
                        solar_line_2 = solar_line_2.astype(int)
                        line1_points = []
                        line2_points = []
                        if len(solar_line_2) > 10 and len(solar_line_1) > 10:

                            # Remove small unwanted patches
                            cv2.fillPoly(high_reso, [rotated_patch_points], 0)#填充任意形状图像high_reso
                            cv2.fillPoly(high_reso_new, [rotated_patch_points], 0)
                            cv2.polylines(high_reso_orig, [rotated_patch_points], 1, 0, 2)
                            cv2.polylines(high_reso_new, [rotated_patch_points], 1, 0, 2)

                            cv2.fillPoly(high_reso_orig, [rotated_patch_points], (0, 0, 255))
                            cv2.fillPoly(high_reso_new, [rotated_patch_points], (0, 0, 255))

                            for i in range(5, len(solar_line_1), 5):
                                line1_points.append(solar_line_1[i])
                            for i in range(5, len(solar_line_2), 5):
                                line2_points.append(solar_line_2[i])

                        # Segmenting Solar Panels in the Solar Patch
                        for points1, points2 in zip(line1_points, line2_points):
                            x1, y1, _ = points1#起点坐标
                            x2, y2, _ = points2#终点坐标
                            cv2.line(high_reso_orig, (x1, y1), (x2, y2), (0, 0, 0), 1)#
                            cv2.line(high_reso_new, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # print("output:arr",high_reso_orig)
        # Number of Solar Panels in series (3/4/5)
        panels_series = panels_series - 1   #pa
    # result = Image.fromarray(high_reso_orig)
    # resut_2 = Image.fromarray(high_reso_new)#实现array到image的转换
    # result.save('output' + fname )#将处理结果保存为彩色图像格式（PNG,JPG等）
    # resut_2.save('panels' + fname)
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(high_reso_orig)
    # # plt.figure()
    # # plt.axis('off')
    # # plt.imshow(high_reso_new)
    # plt.show()
    return high_reso_orig
def white_image(im):
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))

def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #颜色转换（参数一是需要转换的图片，参数二是转换成何种格式）
    #cv2.COLOR_BGR2YCR_CB将BGR格式转换成ycr格式
    channels = cv2.split(ycrcb)#分离通道
    cv2.equalizeHist(channels[0], channels[0])#直方图均衡化，[0]灰度图像
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def delete_zero_bfstr(ss):
    for i in range(len(ss)):
        if ss[i]=='0':
            continue
        else:
            ss=ss[i:]
            break
    return ss

def find_id_ann(ann,imgid):
    l=[]
    for anni in ann:
        if str(anni['image_id'])==imgid:
            l.append(anni)
    return l


with open('/home/whx/Desktop/d_roof/predictions.json','r') as f:
    prediction_json=json.load(f)

testimages_dir='/home/whx/Desktop/d_roof/test_image'
testimages_list=os.listdir(testimages_dir)

pl, pw, l, w, solar_angle = 4, 1, 8, 5, 30
for image_id in testimages_list:
    img_filepath=os.path.join(testimages_dir,image_id)
    img=mpimg.imread(img_filepath)
    img_real=mpimg.imread(img_filepath)
    # high_reso_orig = cv2.pyrUp(img)
    new_image = white_image(img)
    mask=np.zeros(img.shape)[:,:,0]
    img_id=delete_zero_bfstr(image_id.split('.')[0])
    img_annlist=find_id_ann(prediction_json,img_id)
    for ann in img_annlist:
        m=cocomask.decode(ann['segmentation'])
        mask+=m
    mask=mask>0 # 是房顶所有像素点都变成true
    contours = measure.find_contours(mask, 0.5)
    img.flags.writeable=True
    img[:,:,0][mask]=255

    # mask = np.reshape(mask, [300, 300, 1])
    plt.figure()
    plt.subplot(1,5,1)
    plt.title('origin image')
    plt.imshow(img_real)
    plt.axis('off')

    plt.subplot(1,5,2)
    plt.title('masked image')
    plt.imshow(img)
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], color='red',linewidth=1)
    plt.axis('off')
    plt.subplot(1,5,3)
    plt.title('pannel image')
    newcontours = [np.int32(contour)[:,[1,0]] for contour in contours]
    cv2.fillPoly(img_real, newcontours, (255, 255, 255))
    plt.imshow(img_real)

    new_image = white_image(img)
    im = equalize(img_real)
    # 均衡化

    shifted1 = cv2.pyrMeanShiftFiltering(img_real, 21, 51)  # 均值漂移，图像在色彩层面的平滑滤波，它可以中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
    # 漂移物理半径大小sp=21,漂移色彩空间半径大小sr=51
    high_reso_orig = cv2.pyrUp(img_real)
    # convert the mean shift image to grayscale, then apply
    gray1 = cv2.cvtColor(shifted1, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.threshold(gray1, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    solar_roof = thresh1
    high_reso_orig = img_real
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.title('solar_roof_area image')
    plt.imshow(thresh1)
    end_image=panel_rotation(pl, solar_roof)
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.title('solar_roof_area image')
    plt.imshow(end_image)
    plt.show()
