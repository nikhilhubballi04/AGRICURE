import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys

# Load the model
model_path = 'models/model.h5'  # Assuming the model is saved as model.h5 in models folder
model = load_model(model_path)

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Image size
IMG_SIZE = (224, 224)

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    # Print all probabilities
    for i, label in enumerate(labels):
        print(f"{label}: {predictions[0][i]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found.")
        sys.exit(1)
    predict_image(image_path)
