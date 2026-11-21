"""
convert_to_tflite.py

Run this ONCE on your PC, in the folder that contains fabric_model.keras.
It creates fabric_model.tflite (a much lighter version of the same model)
and checks that its predictions match the original.

    python convert_to_tflite.py
"""

import os
import numpy as np
import tensorflow as tf

KERAS_PATH = "fabric_model.keras"
TFLITE_PATH = "fabric_model.tflite"

print("Loading Keras model...")
model = tf.keras.models.load_model(KERAS_PATH)
print("Input shape:", model.input_shape)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Saved {TFLITE_PATH} ({os.path.getsize(TFLITE_PATH) / 1024 / 1024:.1f} MB)")

# ---- Check: both models should give (almost) the same answer ----
print("Checking predictions match...")
x = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
expected = model.predict(x, verbose=0)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
interpreter.set_tensor(inp["index"], x)
interpreter.invoke()
actual = interpreter.get_tensor(out["index"])

diff = float(np.max(np.abs(expected - actual)))
print(f"Max difference between Keras and TFLite: {diff:.6f}")
if diff < 1e-3:
    print("OK - the converted model matches. You can deploy fabric_model.tflite.")
else:
    print("WARNING - the outputs differ more than expected. Send me this output.")
