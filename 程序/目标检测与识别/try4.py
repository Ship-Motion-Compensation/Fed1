import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print("Class number: ", len(classNames))

# Declare path variable
modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

threshold = 0.4

# Load model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    height, width, channel = img.shape
    boundingBox = []
    classIDs = []
    confidences = []
    for outputs in outputs:
        for det in outputs:
            scores = det[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > threshold:
                w, h = int(det[2]*width), int(det[3]*height)  #Box Width and Height
                x, y = int((det[0]*width) - w/2), int(det[1]*height - h/2)  # Center point
                boundingBox.append([x, y, w, h])
                classIDs.append(classID)
                confidences.append(float(confidence))
    # print(len(boundingBox))
    indices = cv2.dnn.NMSBoxes(boundingBox, confidences, threshold, nms_threshold=0.2)
    for i in indices:
        i = i[0]
        box = boundingBox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w, y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIDs[i]].upper()} {int(confidences[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)


while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)  # Find the output layers [3 different output layers]
    # Set the outputs
    outputs = net.forward(outputNames)
    # print(outputs[0].shape)  # Bounding boxes
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    '''The first 4 values of 85 are: Center X, Center Y, Width, Height
          The 5th value is 'Confidence' that there is an object in the box
          The other 80 values are the prediction probabilities of 80 classes'''

    findObjects(outputs, img)
    cv2.imshow('image', img)
    cv2.waitKey(1)
