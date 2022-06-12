import cv2
from aip import AipImageClassify

def get_image():
    cap = cv2.VideoCapture(0)
    while(1):
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



if __name__ == '__main__':
    get_image()
    chack_image()