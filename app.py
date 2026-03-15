import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "model.tflite"
interpreter = None
input_details = None
output_details = None

def get_model():
    global interpreter, input_details, output_details
    if interpreter is None:
        # Load the TFLite model using the standard TF library
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    return interpreter

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

@app.route("/detector", methods=["GET"])
def detector_page():
    return render_template("detector.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400
    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)
        interp = get_model()
        
        # Run TFLite Inference
        interp.set_tensor(input_details[0]['index'], img)
        interp.invoke()
        predictions = interp.get_tensor(output_details[0]['index'])
        
        idx = np.argmax(predictions[0])
        predicted_class = class_names[idx]
        confidence = round(float(np.max(predictions[0])) * 100, 2)

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
