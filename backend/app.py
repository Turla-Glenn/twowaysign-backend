# app.py (updated for JPG + PNG support)
import os
import base64
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from predictor import predict_gesture
from PIL import Image, UnidentifiedImageError

# Flask initialization
app = Flask(__name__)
CORS(app)

PORT = 5000
public_url = ngrok.connect(PORT)
print(f"🔥 Public URL: {public_url}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sign Language Backend is running!",
        "public_url": str(public_url)
    })

def decode_image(image_bytes):
    """Try to decode an image (JPG or PNG) using OpenCV, fallback to PIL."""
    # Try OpenCV
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is not None:
        return image

    # Fallback to PIL
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image
    except UnidentifiedImageError:
        return None

# 🔹 Base64 JSON endpoint
@app.route("/predict", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_base64 = data["image"]

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            return jsonify({"error": "Failed to decode Base64 string"}), 400

        image = decode_image(image_bytes)

        if image is None:
            return jsonify({"error": "Invalid JPG/PNG image data"}), 400

        letter, confidence, suggestion = predict_gesture(image)

        if letter is None:
            return jsonify({"error": "No hand detected"}), 200

        return jsonify({
            "letter": letter,
            "confidence": confidence,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 File upload endpoint
@app.route("/predict-file", methods=["POST"])
def predict_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        filename = file.filename.lower()

        # 🔒 Allow only JPG/JPEG/PNG
        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            return jsonify({"error": "Only JPG and PNG files are allowed"}), 400

        image_bytes = file.read()
        image = decode_image(image_bytes)

        if image is None:
            return jsonify({"error": "Invalid JPG/PNG file"}), 400

        letter, confidence, suggestion = predict_gesture(image)

        if letter is None:
            return jsonify({"error": "No hand detected"}), 200

        return jsonify({
            "letter": letter,
            "confidence": confidence,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=PORT)
