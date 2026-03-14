from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = tf.keras.models.load_model(
    "surface_defect_high_accuracy.h5",
    compile=False
)

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
    error = None

    if request.method == "POST":
        try:
            if "image" not in request.files:
                error = "No image uploaded"
                return render_template("detector.html", error=error)

            file = request.files["image"]

            if file.filename == "":
                error = "No file selected"
                return render_template("detector.html", error=error)

            path = os.path.join(UPLOAD_FOLDER, "test.jpg")
            file.save(path)

            # Image preprocessing
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))

            img = np.array(img).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            pred = model.predict(img)

            class_index = int(np.argmax(pred))
            prediction = classes[class_index]
            confidence = round(float(np.max(pred)) * 100, 2)

            image_path = path

        except Exception as e:
            error = str(e)

    return render_template(
        "detector.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
