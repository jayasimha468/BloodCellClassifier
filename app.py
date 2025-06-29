
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'Blood Cell.h5'
model = load_model(MODEL_PATH)

# Class labels
class_labels = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400

    if file:
        # Save the uploaded file
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return render_template('result.html', prediction=predicted_class, image_file=filepath)

if __name__ == '__main__':
    app.run(debug=True)
