# predictor.py
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant

# Initialize components
model = load_model("model/cnn8grps_rad1_model.h5")
detector = HandDetector(maxHands=1)
d = enchant.Dict("en-US")

# Class labels
labels = list(ascii_uppercase)

def preprocess_frame(image_bgr):
    """
    Preprocess the frame for the CNN model
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (400, 400))  # Match your model input size
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def predict_gesture(image_bgr):
    """
    Detect hand, preprocess, and predict the gesture
    Returns predicted letter and confidence
    """
    hands, img = detector.findHands(image_bgr)
    if not hands:
        return None, 0.0

    # Preprocess for CNN model
    processed = preprocess_frame(image_bgr)
    prediction = model.predict(processed)[0]
    max_idx = np.argmax(prediction)
    confidence = float(prediction[max_idx])

    letter = labels[max_idx]

    # Suggest a word if it's a valid English character
    suggestion = None
    if d.check(letter):
        suggestion = letter

    return letter, confidence, suggestion
