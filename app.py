from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model
model = load_model('CNN_model.h5')

# Class labels
class_labels = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

def model_predict(img_path):
    img = Image.open(img_path).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file.")
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction, confidence = model_predict(file_path)
            return render_template('index.html',
                                   prediction=prediction,
                                   confidence=f"{confidence:.2f}",
                                   image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
