"""
Brain MRI Single Image Inference & Diagnosis CLI
==================================================
Usage:
  python src/predict.py --image data/test/glioma_tumor/glioma_tumor_test_0001.jpg
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

from preprocessing import preprocess_image
from model import CLASS_NAMES


def predict_tumor(image_path: str, model_path: str = None):
    """
    Predicts tumor class and confidence score for a single brain MRI scan.
    """
    if model_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "best_vgg16_mri_model.keras"
        if not model_path.exists():
            model_path = base_dir / "models" / "final_vgg16_mri_model.keras"
            
    if not os.path.exists(str(model_path)):
        print(f"[Error] Trained model not found at {model_path}.")
        print("Please run `python src/train.py` first to train the model.")
        return
        
    print(f"Loading MRI scan from: {image_path}")
    # Run OpenCV preprocessing
    processed_img = preprocess_image(image_path)
    input_tensor = np.expand_dims(processed_img, axis=0)
    
    print(f"Loading VGG16 model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    # Predict probabilities
    predictions = model.predict(input_tensor, verbose=0)[0]
    best_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[best_idx].replace('_', ' ').title()
    confidence = float(predictions[best_idx]) * 100.0
    
    print("\n" + "=" * 55)
    print("           BRAIN MRI CLASSIFICATION RESULT           ")
    print("=" * 55)
    print(f"Predicted Diagnosis :  {predicted_class}")
    print(f"Confidence Score    :  {confidence:.2f}%")
    print("-" * 55)
    print("Detailed Probability Breakdown:")
    for cls_name, prob in zip(CLASS_NAMES, predictions):
        readable_name = cls_name.replace('_', ' ').title().ljust(20)
        bar = "#" * int(prob * 30)
        print(f"  {readable_name}: {prob*100:6.2f}% | {bar}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain MRI tumor type")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI scan image")
    parser.add_argument("--model", type=str, default=None, help="Optional path to model .keras file")
    args = parser.parse_args()
    
    predict_tumor(args.image, args.model)
