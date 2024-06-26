import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('index (copy).html.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load the full-size image
    full_image = cv2.imread(filepath)

    if full_image is None:
        return jsonify({'error': 'Error loading image'}), 400

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    results = []
    for i, (x, y, w, h) in enumerate(faces):
        face_image = full_image[y:y+h, x:x+w]
        processed_face = preprocess_face(face_image, 64, 64)
        
        # Predict the class of the face
        prediction = model.predict(processed_face)
        predicted_class = np.argmax(prediction, axis=1)
        class_label = label_encoder.inverse_transform(predicted_class)[0]
        
        results.append({'label': class_label, 'box': [int(x), int(y), int(w), int(h)]})

        # Draw rectangle and label on the original image
        cv2.rectangle(full_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(full_image, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the final image with bounding boxes and labels
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_faces.jpg')
    cv2.imwrite(output_image_path, full_image)

    return jsonify({'results': results, 'image_file': 'detected_faces.jpg'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5000)
