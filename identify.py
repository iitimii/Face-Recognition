import cv2 as cv
from facenet_keras_model import InceptionResNetV2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
import pickle

def normalize(image):
    mean, std = image.mean(), image.std()
    image = (image - mean)/std
    return image

def get_enc(face, face_encoder):
    face = normalize(face)
    face = cv.resize(face, (160, 160), cv.INTER_LINEAR)
    face = np.expand_dims(face, axis=0)
    encoding = face_encoder.predict(face)
    return encoding

def main():
    encodings_path = 'ENCODINGS/encodings.pkl'
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights('facenet_keras_weights.h5')
    l2_normalizer = Normalizer(norm='l2')
    with open(encodings_path, 'rb') as file:
        database = pickle.load(file)
    confidence = 0.9
    recognition = 0.5
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(confidence)

    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print('CAM NOT OPENED')
            break
        image = cv.flip(image, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_detector.process(image)
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                x1, y1, width, height = int(bboxC.xmin*w), int(bboxC.ymin*h)-20, int(bboxC.width*w), int(bboxC.height*h)+20
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = image[y1:y2, x1:x2]
                encoding = get_enc(face, face_encoder)
                encoding = l2_normalizer.transform(encoding)
                name = 'idk'
                prev_dist = float(1000)
                for database_name, database_encoding in database.items():
                    dist = cosine(np.reshape(database_encoding, (1, -1))[0], np.reshape(encoding, (1, -1))[0])
                    if dist<recognition and dist<prev_dist:
                        name = database_name
                        prev_dist = dist
                if name == 'idk':
                    cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv.putText(image, 'unknown', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(image, f'{name} {prev_dist:.2f}', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        cv.putText(image, 'press q to quit', (470, 460), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow('Camera', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


main()