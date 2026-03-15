import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# --- CONFIGURATION ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "surface_defect_high_accuracy.h5"
model = None

# Function to load model only when needed to save initial boot memory
def get_model():
    global model
    if model is None:
        # Loading with compile=False avoids version-specific optimizer errors
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

class_names = [
    "Crazing", "Inclusion", "Patches", 
    "Pitted Surface", "Rolled-in Scale", "Scratches"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detector", methods=["GET"])
def detector_page():
    return render_template("detector.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Matches the 'name="image"' attribute in your HTML
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)
        
        # Run Prediction
        predictions = get_model().predict(img, batch_size=1)
        idx = np.argmax(predictions)
        predicted_class = class_names[idx]
        confidence = round(float(np.max(predictions)) * 100, 2)

        # Return the result back to the detector page
        return render_template(
            "detector.html",
            prediction=predicted_class,
            confidence=confidence,
            image_path=filepath
        )
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    # Correct port binding for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
