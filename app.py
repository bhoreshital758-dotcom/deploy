"""
app.py

Flask API that loads your trained fabric_model.keras, predicts the
fabric type of an uploaded image, and looks up its sustainability
category from material_database.csv (kept as a separate lookup table,
per the project's two-stage design: image -> fabric type -> lookup ->
eco label).

SETUP:
    pip install flask flask-cors tensorflow pillow --break-system-packages

Expects these three files in the SAME folder as app.py:
    fabric_model.keras
    fabric_model_labels.txt
    material_database.csv

RUN:
    python app.py

Then POST an image to http://localhost:5000/predict as multipart form
field "image", e.g. from the provided frontend/index.html.
"""

import os
import csv
import io

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "fabric_model.keras")
LABELS_PATH = os.path.join(APP_DIR, "fabric_model_labels.txt")
MATERIAL_DB_PATH = os.path.join(APP_DIR, "material_database.csv")
IMG_SIZE = (224, 224)

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]
print(f"Classes: {CLASS_NAMES}")

MATERIAL_DB = {}
with open(MATERIAL_DB_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        MATERIAL_DB[row["Material"].strip().lower()] = row
print(f"Loaded {len(MATERIAL_DB)} materials from material_database.csv")


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # model applies its own preprocess_input internally
    return arr


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    file = request.files["image"]
    try:
        arr = preprocess_image(file.read())
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    predictions = model.predict(arr)[0]  # shape: (num_classes,)
    top_idx = int(np.argmax(predictions))
    top_label = CLASS_NAMES[top_idx]
    top_confidence = float(predictions[top_idx])

    # Full ranked breakdown, useful for showing composition-style results
    ranked = sorted(
        [{"fabric": CLASS_NAMES[i], "confidence": float(predictions[i])} for i in range(len(CLASS_NAMES))],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    material_info = MATERIAL_DB.get(top_label.lower())
    if material_info:
        sustainability = {
            "fibre_type": material_info["Fibre Type"],
            "category": material_info["Sustainability Category"],
            "reason": material_info["Reason/Criteria"],
        }
    else:
        sustainability = {
            "fibre_type": "Unknown",
            "category": "Unknown",
            "reason": f"No entry for '{top_label}' in material_database.csv",
        }

    return jsonify({
        "fabric_type": top_label,
        "confidence": top_confidence,
        "ranked_predictions": ranked,
        "sustainability": sustainability,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
