import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("surface_defect_high_accuracy.h5", compile=False)

# Class labels
class_names = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches"
]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template(
        "index.html",
        prediction=predicted_class,
        img_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)
