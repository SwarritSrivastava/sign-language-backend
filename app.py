from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
model = tf.keras.models.load_model("pretrained_model.h5")
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Label Mapping (Modify if necessary)
labels = {i: chr(65 + i) for i in range(25)}

def normalize_landmarks(landmarks):
    """Normalize landmarks using the wrist (first landmark)."""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    if landmarks.shape[0] == 21:
        wrist = landmarks[0]
        landmarks -= wrist  
        return landmarks.ravel().reshape(1, -1)
    return None

@app.route('/')
def index():
    return "Sign Language Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']  # Base64-encoded image from React
        image_data = base64.b64decode(data.split(",")[1])
        
        image = Image.open(BytesIO(image_data))  # Open image
        image = np.array(image)
        
        # Convert to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                normalized_data = normalize_landmarks(hand_landmarks.landmark)
                if normalized_data is not None:
                    prediction = model.predict(normalized_data)
                    class_index = np.argmax(prediction)
                    predicted_label = labels.get(class_index, '?')
                    return jsonify({"prediction": predicted_label})
        
        return jsonify({"prediction": "No hand detected"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
