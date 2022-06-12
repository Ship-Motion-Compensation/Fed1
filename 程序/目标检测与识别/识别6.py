import cv2
import time
import numpy as np

AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2  # 自动拍照间隔

cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0)

# 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率1280x480；分割为两个640x480、640x480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

counter = 0
utc = time.time()
folder = "./SaveImage/"  # 拍照文件目录


def shot(pos, frame):
    global counter
    path = folder + pos + "_" + str(counter) + ".jpg"

    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)

# 上下翻转
def RotateClockWise180(img):
    new_img = np.zeros_like(img)
    h, w = img.shape[0], img.shape[1]
    for i in range(h): #上下翻转
        new_img[i] = img[h-i-1]
    return new_img

# 镜像反转
def video_mirror_output(video):
    new_img = np.zeros_like(video)
    h, w = video.shape[0], video.shape[1]
    for row in range(h):
        for i in range(w):
            new_img[row, i] = video[row, w-i-1]
    return new_img

while True:
    ret, frame = camera.read()
    # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
    frame = RotateClockWise180(frame)
    frame = video_mirror_output(frame)
    left_frame = frame[0:240, 0:320]
    right_frame = frame[0:240, 320:640]

    cv2.line(img=left_frame, pt1=(120, 0), pt2=(50, 240), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
    cv2.line(img=left_frame, pt1=(200, 0), pt2=(270, 240), color=(255, 0, 0), thickness=5, lineType=8, shift=0)

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)


    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
        utc = now

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")