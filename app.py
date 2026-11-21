"""
app.py

Flask app that loads the trained fabric_model.keras, predicts the
fabric type of an uploaded image, and looks up its sustainability
category from material_database.csv (two-stage design:
image -> fabric type -> lookup -> eco label).

Serves the frontend (index.html) at "/" and the prediction API at "/predict".
"""

import os
import csv
import io

from flask import Flask, request, jsonify, send_from_directory
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

with open(LABELS_PATH, encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]
print(f"Classes: {CLASS_NAMES}")

MATERIAL_DB = {}
with open(MATERIAL_DB_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        MATERIAL_DB[row["Material"].strip().lower()] = row
print(f"Loaded {len(MATERIAL_DB)} materials from material_database.csv")

# Fast inference function + warm-up so the first real request isn't slow
infer = tf.function(lambda x: model(x, training=False))
infer(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))
print("Model warmed up")


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img.draft("RGB", (IMG_SIZE[0] * 2, IMG_SIZE[1] * 2))  # fast downscale while decoding JPEGs
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # model applies its own preprocess_input internally
    return arr


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(APP_DIR, "index.html")


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

    predictions = infer(arr).numpy()[0]  # shape: (num_classes,)
    top_idx = int(np.argmax(predictions))
    top_label = CLASS_NAMES[top_idx]
    top_confidence = float(predictions[top_idx])

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)