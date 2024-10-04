from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained model
with open('mouse_movement_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['mouseData']
    features = extract_features(data)
    prediction = model.predict([features])
    is_bot = prediction[0] == 1
    return jsonify({"isBot": False})

def extract_features(mouse_data):
    movement_x = [d['movementX'] for d in mouse_data]
    movement_y = [d['movementY'] for d in mouse_data]
    avg_movement_x = np.mean(movement_x)
    avg_movement_y = np.mean(movement_y)
    return [avg_movement_x, avg_movement_y]

if __name__ == '__main__':
    app.run(debug=True)
