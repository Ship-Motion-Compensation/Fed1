import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
from aip import AipImageClassify
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import time


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def sig():
    im = cv2.imread('test.jpg')

    bbox, label, conf = cv.detect_common_objects(im)

    output_image = draw_bbox(im, bbox, label, conf)

    # print(bbox)
    # print(label)

    plt.imshow(output_image)

    plt.show()

    result = Counter(label)
    result0 = dict(result)
    # print(result0)
    data_key = "person"
    i = 0
    for key in result0.keys():
        if key == data_key:
            i +=1
    if i == 1:
        print(result0[key])
    if i == 0:
        print("没有相应物品！")

    # 挑选目标物品
    I = []
    for i in range(len(label)):
        if label[i] == 'cell phone':
            I.append(i)

    # 提取目标中心点
    C = []
    for i in I:
        x = (bbox[i][2] - bbox[i][0])/2
        y = (bbox[i][3] - bbox[i][1])/2
        D = [x, y]
        C.append(D)

    # 与图片中心点比较
    D_X = []
    D_XY = []
    for d in C:
        t_c = [320, 240]
        d_x = (np.array(d) - np.array(t_c))
        ab_x = abs(d_x)
        # print(d_x)
        D_X.append(ab_x[0])
        D_XY.append(d_x)
    # print(D_X)

    # 选择与中心点近的物品，以x为选择标准
    if D_X != []:
        z = D_X.index(min(D_X))
        print(D_XY[z])
        if D_XY[z][0] < -10:
            if D_XY[z][1] < -10:
                print("左上移动")
            if D_XY[z][1] > 10:
                print("右上移动")
            if D_XY[z][1] >= -10 and D_XY[z][1] <= 10:
                print('不动')
        if D_XY[z][0] > 10:
            if D_XY[z][1] < -10:
                print("左下移动")
            if D_XY[z][1] > 10:
                print("右下移动")
            if D_XY[z][1] >= -10 and D_XY[z][1] <= 10:
                print('不动')
        if D_XY[z][0] >= -10 and D_XY[z][0] <= 10:
            if D_XY[z][1] < -10:
                print("右移动")
            if D_XY[z][1] > 10:
                print("左移动")
            if D_XY[z][1] >= -10 and D_XY[z][1] <= 10:
                print('不动')
    else:
        print("移动镜头，寻找物品！")

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)

        if cv2.waitKey(1):
            cv2.imwrite("test.jpg", frame)
            time.sleep(0.01)

        sig()
    cap.release()
    cv2.destroyAllWindows()
    # chack_image()
    # sig()