import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
from aip import AipImageClassify
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def get_image():
    cap = cv2.VideoCapture(0)
    while (1):
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            cv2.imwrite("test.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 上下翻转
def RotateClockWise180(img):
    new_img=np.zeros_like(img)
    h,w=img.shape[0],img.shape[1]
    for i in range(h): #上下翻转
        new_img[i]=img[h-i-1]
    return new_img


def chack_image():
    APP_ID = '24780223'
    API_KEY = 'pGYz7e105PWRevwGM4CYZqGq'
    SECRET_KEY = 'uOoXyejA9Vrkc1g08zNU2EL2xaFAXtqu'

    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    image = get_file_content('test.jpg')

    keyword = client.advancedGeneral(image)

    # print(keyword)

    options = {}
    options["baike_num"] = 5

    client.advancedGeneral(image, options)

    print("num:"), print(keyword['result_num'])
    for i in range(keyword['result_num']):
        print(keyword['result'][i])

    list1 = []
    for i in range(keyword['result_num']):
        list1.append(keyword['result'][i]['keyword'])
    print(list1)

def sig():
    im = cv2.imread('test.jpg')
    # w = im.shape[0:2]
    # print(w)

    im = RotateClockWise180(im)
    im = im[0:, 0:320]  # 裁剪坐标为[y0:y1, x0:x1]

    bbox, label, conf = cv.detect_common_objects(im)

    output_image = draw_bbox(im, bbox, label, conf)

    print(bbox)
    print(label)

    plt.imshow(output_image)

    plt.show()

    result = Counter(label)
    result0 = dict(result)
    print(result0)
    data_key = "wine glass"
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
        if label[i] == 'wine glass':
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
        t_c = [160, 125]
        d_x = (np.array(d) - np.array(t_c))
        ab_x = abs(d_x)
        print(d_x)
        D_X.append(ab_x[0])
        D_XY.append(d_x)
    print(D_X)

    # 选择与中心点近的物品，以x为选择标准
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

if __name__ == '__main__':
    get_image()
    # chack_image()
    sig()