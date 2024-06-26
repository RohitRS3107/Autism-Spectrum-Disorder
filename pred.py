import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the pre-trained model and label encoder
model = load_model('asd_classification_model.h5')
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Function to preprocess a single face image
def preprocess_face(face_image, image_height, image_width):
    face_image = cv2.resize(face_image, (image_height, image_width))
    face_image = face_image.astype('float32') / 255.0
    return np.expand_dims(face_image, axis=0)  # Add batch dimension

# Path to the new full-size image
full_image_path = input()


# Load the full-size image
full_image = cv2.imread(full_image_path[1:-2])

# Check if the image is loaded correctly
if full_image is not None:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for i, (x, y, w, h) in enumerate(faces):
        face_image = full_image[y:y+h, x:x+w]
        processed_face = preprocess_face(face_image, 64, 64)
        
        # Predict the class of the face
        prediction = model.predict(processed_face)
        predicted_class = np.argmax(prediction, axis=1)
        class_label = label_encoder.inverse_transform(predicted_class)[0]
        print(class_label)
        # Draw rectangle and label on the original image
        cv2.rectangle(full_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(full_image, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Save each detected face with label to a file
        face_filename = f'detected_face_{i}.jpg'
        cv2.imwrite(face_filename, face_image)

    # Save the final image with bounding boxes and labels
    output_image_path = 'detected_faces.jpg'
    cv2.imwrite(output_image_path, full_image)

    print(f"Detected faces and their labels saved to {output_image_path} and individual face files.")

else:
    print("Error: Full-size image not loaded correctly.")
