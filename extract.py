# Initialize the face detector and facial landmark detector
import cv2
import numpy as np
from lbf import lbf_model_path
from load_dataset import train_images, val_images, test_images
from lable_encode import train_labels_encoded, val_labels_encoded, test_labels_encoded
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(lbf_model_path)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    _, landmarks = facemark.fit(gray, faces)
    features = landmarks[0].flatten()
    return features

def process_images(images, labels):
    feature_list = []
    label_list = []
    for image, label in zip(images, labels):
        features = extract_features(image)
        if features is not None:
            feature_list.append(features)
            label_list.append(label)
    return np.array(feature_list), np.array(label_list)    
   

X_train, y_train = process_images(train_images, train_labels_encoded)
X_val, y_val = process_images(val_images, val_labels_encoded)
X_test, y_test = process_images(test_images, test_labels_encoded)


print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")