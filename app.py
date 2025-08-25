import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ----------------------------
# App Setup
# ----------------------------
app = Flask(__name__)
model = load_model(r"final_model.h5")

# Class names must be in alphabetical order
class_names = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

# --- Detailed information database for each disease ---
disease_info_db = {
    'Measles': {
        'description': "A highly contagious viral infection common in children, preventable by vaccine.",
        'symptoms': [
            "High fever, often over 104°F (40°C)",
            "Cough and runny nose",
            "Red, watery eyes (conjunctivitis)",
            "Koplik spots (tiny white spots) inside the mouth",
            "A widespread skin rash of large, flat blotches"
        ],
        'actions': [
            "Consult a doctor immediately for a proper diagnosis.",
            "Isolate to prevent spreading the virus to others.",
            "Rest and drink plenty of fluids like water and juice.",
            "Avoid scratching the rash."
        ],
        'prevention': [
            "The MMR (measles, mumps, and rubella) vaccine is highly effective.",
            "Avoid contact with infected individuals."
        ]
    },
    'Monkeypox': {
        'description': "A rare viral disease, similar to smallpox but milder. It is typically found in parts of Africa.",
        'symptoms': [
            "Fever and headache",
            "Swollen lymph nodes",
            "Muscle aches and backache",
            "A rash that can look like pimples or blisters",
            "Lesions that crust, scab over, and fall off"
        ],
        'actions': [
            "Seek immediate medical attention for diagnosis and guidance.",
            "Avoid close contact with people and animals.",
            "Keep lesions clean and covered to prevent secondary infections."
        ],
        'prevention': [
            "Avoid contact with animals that could harbor the virus.",
            "Practice good hand hygiene after contact with infected animals or humans."
        ]
    },
    'Chickenpox': {
        'description': "A very contagious infection caused by the varicella-zoster virus, resulting in an itchy rash.",
        'symptoms': [
            "Itchy rash with small, fluid-filled blisters",
            "Fever and tiredness",
            "Loss of appetite and headache",
            "Rash appears first on the chest, back, and face"
        ],
        'actions': [
            "Consult a healthcare provider to confirm the diagnosis.",
            "Avoid scratching the blisters to prevent scarring.",
            "Take oatmeal baths to soothe itchy skin.",
            "Stay home from school or work until all blisters have crusted over."
        ],
        'prevention': [
            "The chickenpox vaccine is the best way to prevent it.",
            "Avoid close contact with anyone who has chickenpox."
        ]
    },
    'HFMD': {
        'description': "A mild, contagious viral infection common in young children.",
        'symptoms': [
            "Fever and sore throat",
            "Painful, red, blister-like lesions in the mouth",
            "A non-itchy skin rash on the palms and soles",
            "Rash can also appear on knees, elbows, and buttocks"
        ],
        'actions': [
            "See a doctor for proper diagnosis.",
            "Drink plenty of liquids to stay hydrated.",
            "Avoid spicy or acidic foods that can irritate mouth sores."
        ],
        'prevention': [
            "Wash hands frequently with soap and water.",
            "Disinfect frequently touched surfaces.",
            "Avoid close contact such as kissing or sharing utensils."
        ]
    },
    'Cowpox': {
        'description': "A rare skin infection caused by the cowpox virus, transmitted by contact with infected animals.",
        'symptoms': ["Localized, painful skin lesions", "Fever and fatigue", "Swollen lymph glands"],
        'actions': ["Consult a doctor for diagnosis and care.", "Keep the lesion clean and covered."],
        'prevention': ["Avoid direct contact with lesions on infected animals."]
    },
    'Healthy': {
        'description': "The scan indicates no signs of the skin diseases the model was trained to identify.",
        'symptoms': ["No visible rashes, lesions, or blisters detected."],
        'actions': ["Continue with a good skincare routine.", "Consult a dermatologist for any concerns."],
        'prevention': ["Maintain good hygiene.", "Protect your skin from excessive sun exposure."]
    }
}

# ----------------------------
# Preprocessing & Routes
# ----------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    uploads_dir = os.path.join(app.root_path, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, "temp_image.jpg")
    file.save(file_path)

    prediction_result = {}
    try:
        img_array = preprocess_image(file_path)
        preds = model.predict(img_array)

        probabilities = tf.nn.softmax(preds[0])
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(probabilities)) * 100

        # Get all info for the predicted class from the new database
        info = disease_info_db.get(predicted_class_name, {})

        prediction_result = {
            'prediction': predicted_class_name,
            'confidence': f"{confidence:.2f}%",
            'description': info.get('description', 'N/A'),
            'symptoms': info.get('symptoms', []),
            'actions': info.get('actions', []),
            'prevention': info.get('prevention', [])
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify(prediction_result)

# ----------------------------
# Run App
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)