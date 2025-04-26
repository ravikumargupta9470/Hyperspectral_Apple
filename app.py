import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
xception_model = load_model('ensemble_xception.h5')
resnet_model = load_model('ensemble_mobilenet.h5')
mobilenet_model = load_model('ensemble_resnet.h5')

# Categories mapping
categories = ["Fresh", "Low", "High"]

# Ensemble prediction function
def ensemble_predict(models, X):
    predictions = [model.predict(X) for model in models]
    avg_prediction = np.mean(predictions, axis=0)  # Average predictions
    return avg_prediction

# Prediction function for a single image
def predict_single_image(image_path, models):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (299, 299))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Use ensemble to predict
    preds = ensemble_predict(models, img)
    category = np.argmax(preds)
    confidence = np.max(preds) * 100

    # Printing result to match your desired output format
    print(f"Prediction: {categories[category]} with confidence {confidence:.2f}%")
    return category, confidence

# Route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict for a single image
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Save the image to a temporary path
    image_path = "temp_image.jpg"
    file.save(image_path)

    # List of models for ensemble prediction
    models = [xception_model, resnet_model, mobilenet_model]
    
    # Predict using the models
    category, confidence = predict_single_image(image_path, models)
    
    # Clean up the temporary image
    os.remove(image_path)

    # Return the prediction result
    return jsonify({
        'category': categories[category],
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
