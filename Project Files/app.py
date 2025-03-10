import numpy as np
import os
import cv2
import joblib
from flask import Flask, request, render_template

# Load the trained model
model = joblib.load("eye_disease_model.pkl")

# Define categories
CATEGORIES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Initialize Flask app
app = Flask(__name__,template_folder="templates")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inp')
def inp():
    return render_template('img_input.html')

# Function to extract features from uploaded images
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (100, 100))  # Resize to match training size
    features = img.flatten()  # Convert to a 1D array
    return np.array([features])

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        f = request.files['image']
        filepath = "uploads/" + f.filename  # Save uploaded file
        f.save(filepath)

        # Extract features and predict
        img_features = extract_features(filepath)
        prediction = model.predict(img_features)[0]  # Get class index
        result = CATEGORIES[prediction]  # Convert to label

        return render_template('output.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)