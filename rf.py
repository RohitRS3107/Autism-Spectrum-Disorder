import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import requests
import joblib
from imgaug import augmenters as iaa

# Download the LBF Model
lbf_model_url = 'https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml'
lbf_model_path = 'lbfmodel.yaml'

response = requests.get(lbf_model_url)
with open(lbf_model_path, 'wb') as file:
    file.write(response.content)

# Initialize the face detector and facial landmark detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(lbf_model_path)

# Define data augmentation
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.GaussianBlur(sigma=(0, 0.5)), # blur images
    iaa.LinearContrast((0.75, 1.5)), # improve or worsen contrast
    iaa.Multiply((0.8, 1.2), per_channel=0.2), # change brightness
])

def load_dataset(folder_path):
    images = []
    labels = []
    for label in ['autistic', 'non_autistic']:
        label_path = os.path.join(folder_path, label)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(label)
    return images, labels

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    _, landmarks = facemark.fit(gray, faces)
    features = landmarks[0].flatten()
    return features

def process_images(images, labels, augment=False):
    feature_list = []
    label_list = []
    for image, label in zip(images, labels):
        if augment:
            image = augmenter(image=image)
        features = extract_features(image)
        if features is not None:
            feature_list.append(features)
            label_list.append(label)
    return np.array(feature_list), np.array(label_list)

# Load dataset
train_path = '/home/thor/Downloads/asdClassification/train'
val_path = '/home/thor/Downloads/asdClassification/valid'
test_path = '/home/thor/Downloads/asdClassification/test'


train_images, train_labels = load_dataset(train_path)
val_images, val_labels = load_dataset(val_path)
test_images, test_labels = load_dataset(test_path)

# Encode labels
label_encoder = LabelEncoder()

train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Process images with augmentation for training data
X_train, y_train = process_images(train_images, train_labels_encoded, augment=True)
# Process images without augmentation for validation and test data
X_val, y_val = process_images(val_images, val_labels_encoded, augment=False)
X_test, y_test = process_images(test_images, test_labels_encoded, augment=False)

# Ensure the sizes match
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Define parameter grid for RandomizedSearchCV
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
}

# Use RandomizedSearchCV for broader search space
model = SVC()
clf = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)

print("Best Parameters:", clf.best_params_)

# Use the best model found by RandomizedSearchCV
model = clf.best_estimator_

# Validate the model
y_val_pred = model.predict(X_val)
print("Validation Results:\n", classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))

# Evaluate the model
y_test_pred = model.predict(X_test)
print("Test Results:\n", classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# Save the model
model_filename = 'asd_classification_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")
