import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
from aip import AipImageClassify
import matplotlib.pyplot as plt

from collections import Counter

dict_0 = {'人':'person','自行车':'bicycle','车':'car','摩托车':'motorbike','飞机':'aeroplane','公共汽车':'bus','火车':'train','卡车':'truck','船':'boat','红绿灯':'traffic light',
'消防栓':'fire hydrant','停车标志':'stop sign','停车费':'parking meter','板凳':'bench','鸟':'bird','猫':'cat','狗':'dog','马':'horse','羊':'sheep',
'牛':'cow','大象':'elephant','熊':'bear','斑马':'zebra','长颈鹿':'giraffe','背包':'backpack','伞':'umbrella','手提包':'handbag','领带':'tie','手提箱':'suitcase',
'飞盘':'frisbee','溜冰鞋':'skis','滑雪板':'snowboard','体育球':'sports ball','风筝':'kite','棒球棒':'baseball bat','棒球手套':'baseball glove','滑板':'skateboard',
'冲浪板':'surfboard','网球拍':'tennis racket','瓶':'bottle','酒杯':'wine glass','杯':'cup','叉':'fork','刀':'knife','勺子':'spoon','碗':'bowl','香蕉':'banana','苹果':'apple',
'三明治':'sandwich','橙色':'orange','西兰花':'broccoli','胡萝卜':'carrot','热狗':'hot dog','披萨':'pizza','甜甜圈':'donut','蛋糕':'cake','椅子':'chair',
'沙发':'sofa','盆栽':'potted plant','床':'bed','餐桌':'dining table','厕所':'toilet','电视':'tv','电脑':'laptop','鼠标':'mouse','远程':'remote',
'键盘':'keyboard','手机':'cell phone','微波':'microwave','烤箱':'oven','烤面包机':'toaster','水槽':'sink','冰箱':'refrigerator','书':'book','时钟':'clock',
'花瓶':'vase','剪刀':'scissors','泰迪熊':'teddy bear','吹风机':'hair drier','牙刷':'toothbrush'}

def C_E(data_key_0):
    for key_0 in dict_0.keys():
        if key_0 == data_key_0:
            data_key_1 = dict_0[data_key_0]
            return data_key_1

def get_image():
    cap = cv2.VideoCapture(0)
    while (1):
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        if cv2.waitKey(1):
            cv2.imwrite("test.jpg", frame)
            break
    cap.release()
    cv2.destroyAllWindows()


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


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

    bbox, label, conf = cv.detect_common_objects(im)

    output_image = draw_bbox(im, bbox, label, conf)

    plt.imshow(output_image)

    plt.show()

    result = Counter(label)
    result0 = dict(result)
    print(result0)
    data_key_0 = "电视"

    data_key = C_E(data_key_0)
    i = 0
    for key in result0.keys():
        if key == data_key:
            i +=1
    if i == 1:
        print(result0[data_key])
    if i == 0:
        print("没有相应物品！")




if __name__ == '__main__':
    get_image()
    # chack_image()
    sig()