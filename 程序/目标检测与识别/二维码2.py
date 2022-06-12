import cv2
from pyzbar import pyzbar
import csv
import json


def ma(Goods):
    found = set()
    capture = cv2.VideoCapture(0)
    PATH = "test.csv"
    while (1):
        ret, frame = capture.read()
        test = pyzbar.decode(frame)
        for tests in test:
            testdate = tests.data.decode('utf-8')
            # print(testdate)
            A = json.loads(testdate)
            for i in A:
                # print(i)
                if i == Goods:
                    B = A[i]
            if testdate not in found:
                with open(PATH, 'a+') as f:
                    csv_write = csv.writer(f)
                    date = [testdate]
                    csv_write.writerow(date)
                found.add(testdate)
        # cv2.imshow('Test',frame)
        # if cv2.waitKey(1) == ord('q'):
        if test != []:
            break
    return B

if __name__ == "__main__":
    print(ma("苹果"))