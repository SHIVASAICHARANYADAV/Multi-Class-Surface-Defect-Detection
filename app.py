from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Create upload folder if it does not exist
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the fixed model
model = tf.keras.models.load_model("surface_defect_fixed.h5", compile=False)

# Class names
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detector", methods=["GET", "POST"])
def detector():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        # Save uploaded image
        path = os.path.join(UPLOAD_FOLDER, "test.jpg")
        file.save(path)

        # Preprocess image
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)

        class_index = np.argmax(pred)

        prediction = classes[class_index]
        confidence = round(float(np.max(pred)) * 100, 2)

        image_path = path

    return render_template(
        "detector.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
