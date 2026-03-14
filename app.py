import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# --- OPTIMIZATION: Disable GPU & Reduce Logging ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU only to save memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

# Load model - Load it globally but ensure it's compiled for inference
MODEL_PATH = "surface_defect_high_accuracy.h5"
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# Class labels
class_names = [
    "Crazing", "Inclusion", "Patches", 
    "Pitted Surface", "Rolled-in Scale", "Scratches"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    # Using 'with' ensures the file is closed properly to save RAM
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)
        
        # Inference
        predictions = get_model().predict(img, batch_size=1)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template(
            "index.html",
            prediction=predicted_class,
            img_path=filepath
        )
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    # Render uses the PORT environment variable. 
    # If not present (local testing), it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
