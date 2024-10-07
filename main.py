import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    sys.exit()

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

classesFile = "coco.names"
try:
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: {classesFile} not found.")
    cap.release()
    cv.destroyAllWindows()
    sys.exit()

modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
try:
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
except cv.error as e:
    print(f"Error: {e}")
    cap.release()
    cv.destroyAllWindows()
    sys.exit()

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                       (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    findObjects(outputs, img)

    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Image', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    cv.imshow('Image', img)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
