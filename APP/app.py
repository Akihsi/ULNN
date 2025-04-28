from flask import Flask, render_template, request, send_from_directory, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# --- Flask app setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load model ---
model = load_model('detection.h5')

# --- Image Preprocessing ---
IMG_SIZE = (128, 128)

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image

def predict_image(image_path):
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    img = preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)[0][0]

    print(f"Raw model prediction: {prediction}")

    if prediction > 0.5:
        return "AI Generated"
    else:
        return "Real Image"

# --- Serve uploaded files ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = predict_image(filepath)

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
