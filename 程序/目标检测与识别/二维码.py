import cv2
from pyzbar import pyzbar
import csv
found = set()
capture = cv2.VideoCapture(0)
PATH = "test.csv"
i = 0
while True:
    ret,frame = capture.read()
    test = pyzbar.decode(frame)
    for tests in test:
        testdate = tests.data.decode('utf-8')
        print(testdate)
        if testdate not in found:
            with open(PATH,'a+') as f:
                csv_write = csv.writer(f)
                date = [testdate]
                csv_write.writerow(date)
                i += 1
            found.add(testdate)
        break
    cv2.imshow('Test',frame)
    cv2.waitKey(1)
    if i != 0:
        break
