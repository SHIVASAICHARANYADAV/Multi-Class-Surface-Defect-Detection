from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# create uploads folder
os.makedirs("static/uploads", exist_ok=True)

model = None

classes = [
"crazing",
"inclusion",
"patches",
"pitted_surface",
"rolled_in_scale",
"scratches"
]

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(
            "surface_defect_high_accuracy.h5",
            compile=False
        )
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detector", methods=["GET","POST"])
def detector():

    prediction=None
    confidence=None
    image_path=None

    if request.method=="POST":

        if "image" not in request.files:
            return "No image uploaded"

        file=request.files["image"]

        if file.filename=="":
            return "No selected file"

        path="static/uploads/test.jpg"
        file.save(path)

        # FIX: convert image to RGB
        img=Image.open(path).convert("RGB")
        img=img.resize((224,224))

        img=np.array(img)/255.0
        img=np.expand_dims(img,axis=0)

        m = load_model()

        pred=m.predict(img)

        class_index=np.argmax(pred)

        prediction=classes[class_index]

        confidence=round(float(np.max(pred))*100,2)

        image_path=path

    return render_template(
        "detector.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
