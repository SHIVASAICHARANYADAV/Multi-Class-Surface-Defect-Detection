import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# --- OPTIMIZATION ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = "surface_defect_high_accuracy.h5"
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

class_names = ["Crazing", "Inclusion", "Patches", "Pitted Surface", "Rolled-in Scale", "Scratches"]

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

# FIXED: Added the missing route for the detector page
@app.route("/detector", methods=["GET"])
def detector_page():
    return render_template("detector.html")

# FIXED: Unified the route and file keys
@app.route("/predict", methods=["POST"])
def predict():
    # Changed "file" to "image" to match your HTML
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)
        predictions = get_model().predict(img, batch_size=1)
        
        # Calculate results
        idx = np.argmax(predictions)
        predicted_class = class_names[idx]
        confidence = round(float(np.max(predictions)) * 100, 2)

        # We stay on detector.html to show results
        return render_template(
            "detector.html",
            prediction=predicted_class,
            confidence=confidence,
            image_path=filepath
        )
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
