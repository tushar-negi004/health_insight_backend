from flask import Flask, request, jsonify
from google.genai import types
from google import genai
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Securely get API key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("error")

# Initialize the Gemini client
client = genai.GenerativeModel(model_name="gemini-2.0-flash", api_key=API_KEY)

sys_instruct = """only provide responses in paragraphs, no bullet points no list, no table and no special symbols should be used in the response. don't provide the code in any programming language(if someone asks) in such cases you can provide algorithm for such programs"""

@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = client.generate_content(prompt)  # Fixed the API call
        return jsonify({"generatedText": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load ML Model
MODEL_PATH = 'Chrono_tree_model_refined.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

FEATURE_ORDER = ['sex', 'BMI', 'cigars_weekly', 'drinks_weekly', 'exercise', 'chronic_disease', 'diet_quality', 'healthcare', 'current_age']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)
    
    try:
        missing_keys = [key for key in FEATURE_ORDER if key not in data]
        if missing_keys:
            return jsonify({'error': f'Missing keys: {missing_keys}'}), 400

        input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
        prediction = model.predict(input_df)[0]

        return jsonify({'health_score': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# No need for app.run() when using Gunicorn
