# import the necessary packages
import numpy as np
import face_recognition
import cv2
import os
import sys, datetime
from time import sleep
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


PRO = True

# Path where the images are
path = "./registered_users"

imgList = []
imgLabels = []

# List all the files in that directory
listFiles = os.listdir(path)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Load the faceNet files
prototxtPath = r"./face_detector/deploy.prototxt"
weightsPath = r"./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Read all the images in the image folder
for cls in listFiles:
    curImg = cv2.imread(f'{path}/{cls}')
    imgList.append(curImg)
    imgLabels.append(os.path.splitext(cls)[0])

# Function to add data to the database
def markTheData(name, mask):
    with open('database.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        now = datetime.datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtString},{mask}')


# Function that detect and predict mask
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs, loccs = [], []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX-startX, endY-startY)), loccs.append((startY, endX-startX, endY-startY, startX))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds, loccs)

# This function find the encoding of an image
def findEncodings(images):
    encodeList = []
    for img in images:
        (locs, preds, loccs) = detect_and_predict_mask(img, faceNet, maskNet)
        startY, endX, endY, startX = loccs[0]
        endX = endX+startX
        endY = endY + startY
        loccs = [(startY, endX, endY, startX)]
        # encode = face_recognition.face_encodings(img, loccs)[0]
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(imgList)

# This function recognize a person face.
def face_recognize(frame, locs):
    startY, endX, endY, startX = locs[0]
    cropped = frame[startY:endY, startX:endX]
    # encodesCurFrame = face_recognition.face_encodings(frame, locs)
    encodesCurFrame = face_recognition.face_encodings(frame)
    for encodeFace in encodesCurFrame:
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        face_distance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(face_distance)
        matchIndex = np.argmin(face_distance)
        # print(face_distance)
        # print(matchIndex)
        # print(matches)
        if matches[matchIndex]:
            name = imgLabels[matchIndex].upper()
            return name
        else:
            return "UnKnown"

# Another function to add data to the database. This function add data of mask or no mask with date
def markTheData1(mask):
    with open('database1.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        now = datetime.datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{dtString},{mask}')


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

# This function draw boxes around the face
def draw_boxes(frame, boxes, preds, color):
    for (x, y, w, h), pred in zip(boxes, preds):
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask-0.10 else "No Mask"
        # color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return frame

# This function is used to resize the image
def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

# A class that is used to detect faces and predict mask
class FaceDetector():

    def __init__(self):
        pass

    def detect(self, frame):
        (locs, preds, loccs) = detect_and_predict_mask(frame, faceNet, maskNet)
        return (locs, preds, locs)

# A class used to track a face
class FaceTracker():
    
    def __init__(self, frame, face):
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face

# This is used to control the tracker wether to detect a new face or keep tracking
class Controller():
    
    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self, faces):
        """Return True if should trigger event"""
        return (faces == [(0, 0, 0, 0)] or faces == []) and (self.get_seconds_since() > self.event_interval)
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

# This class combine all the other classes
class Pipeline():
    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = FaceDetector()
        self.trackers = []
    
    def detect_and_track(self, frame):
        # get faces 
        (faces, preds, locs) = self.detector.detect(frame)

        self.preds = preds
        if faces != [] and preds != []:
            faces, preds = [faces[0]], [preds[0]]

        # reset timer
        self.controller.reset()

        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces]

        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        new = type(faces) is not tuple
        self.faces = faces
        return faces, new, preds
    
    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        self.faces = boxes
        return boxes, False, self.preds
    
    def boxes_for_frame(self, frame):
        if self.controller.trigger(self.faces):
            return self.detect_and_track(frame)
        else:
            return self.track(frame)

