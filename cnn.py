import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image dimensions
image_height, image_width = 224, 224

# Load and preprocess images
def load_and_preprocess_images(folder_path, image_height, image_width):
    images = []
    labels = []
    for label in ['autistic', 'non_autistic']:
        label_path = os.path.join(folder_path, label)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (image_height, image_width))  # Resize images
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load datasets
train_path = '/home/thor/Downloads/asdClassification/train'
val_path = '/home/thor/Downloads/asdClassification/valid'
test_path = '/home/thor/Downloads/asdClassification/test'

X_train, y_train = load_and_preprocess_images(train_path, image_height, image_width)
X_val, y_val = load_and_preprocess_images(val_path, image_height, image_width)
X_test, y_test = load_and_preprocess_images(test_path, image_height, image_width)

# Normalize image data
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(y_train)
val_labels_encoded = label_encoder.transform(y_val)
test_labels_encoded = label_encoder.transform(y_test)

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(train_labels_encoded)
y_val_cat = to_categorical(val_labels_encoded)
y_test_cat = to_categorical(test_labels_encoded)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    validation_data=(X_val, y_val_cat),
    epochs=61
)

# Evaluate the model
val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)

print(f'Validation accuracy: {val_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save('asd_classification_model.h5.h5.h5')

# Optionally, save the label encoder for decoding predictions later
import pickle
with open('label_encoder.pkl.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
