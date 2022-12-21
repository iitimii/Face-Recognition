import cv2 as cv
from facenet_keras_model import InceptionResNetV2
import numpy as np
import os
from sklearn.preprocessing import Normalizer
import pickle
import mediapipe as mp

face_encoder = InceptionResNetV2()
face_encoder.load_weights('facenet_keras_weights.h5')
l2_normalizer = Normalizer(norm='l2')
encodings = []
encodings_dict = {}
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(0.9)

def normalize(image):
    mean, std = image.mean(), image.std()
    image = (image - mean)/std
    return image

def get_encoding(image_path):
    image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    results = face_detector.process(image)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = image.shape
            x1, y1, width, height = int(bboxC.xmin*w), int(bboxC.ymin*h)-20, int(bboxC.width*w), int(bboxC.height*h)+25
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]
        face = normalize(face)
        face = cv.resize(face, (160, 160), cv.INTER_LINEAR)
        face = np.expand_dims(face, axis=0)
        encoding = face_encoder.predict(face)
    else:
        encoding = []
    return encoding


IMAGES_PATH = 'IMAGES'
for person in os.listdir(IMAGES_PATH):
    print(person)
    person_path = os.path.join(IMAGES_PATH, person)
    for image in os.listdir(person_path):
        image_path = os.path.join(person_path, image)
        encoding = get_encoding(image_path)
        if np.array(encoding).shape != (0,):
            encodings.append(encoding)
        else:
            print(f'No face in image: {image_path}')
    if encodings:
        encoding_sum = np.sum(encodings, axis=0)
        encoding_sum = l2_normalizer.transform(encoding_sum)[0]
        encodings_dict[person] = encoding_sum

encodings_path = 'ENCODINGS/encodings.pkl'
with open(encodings_path, 'wb') as file:
    pickle.dump(encodings_dict, file)

print('Done')
[print(encodings_dict.keys())]