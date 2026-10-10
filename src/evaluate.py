"""
Brain MRI Classification Model Evaluation & Explainability (Grad-CAM)
=====================================================================
Performs:
1. Multi-class Confusion Matrix & Classification Report calculation.
2. Macro / Weighted Precision, Recall, F1-Score, and Overall Accuracy.
3. Multi-class ROC-AUC analysis.
4. Grad-CAM (Gradient-weighted Class Activation Mapping) for visual interpretability.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import cv2

from model import CLASS_NAMES


def generate_gradcam(model, img_array, last_conv_layer_name="block5_conv3", pred_index=None):
    """
    Computes Grad-CAM heatmap for visual interpretability of tumor localized activation.
    """
    try:
        # Extract VGG16 base
        vgg_base = None
        for l in model.layers:
            if "vgg16" in l.name:
                vgg_base = l
                break
                
        if vgg_base is None:
            return None
            
        last_conv_layer = vgg_base.get_layer(last_conv_layer_name)
        
        # Sub-model mapping base input to last conv layer activations
        conv_model = tf.keras.Model(vgg_base.inputs, last_conv_layer.output)
        
        # Classifier layers after conv base
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        found = False
        for l in model.layers:
            if l.name == "global_avg_pool":
                found = True
            if found:
                x = l(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_array)
            tape.watch(conv_outputs)
            preds = classifier_model(conv_outputs)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"[Grad-CAM Notice] Could not compute Grad-CAM: {e}")
        return None


def evaluate_model(
    model_path: str,
    test_dir: str,
    output_dir: str = "reports"
):
    """
    Runs full test evaluation and exports metrics and visualization artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    # Generate predictions
    print("Evaluating predictions on test set...")
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    
    # 1. Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv)
    print("\n--- Model Performance Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print(f"Saved classification report to {report_csv}")
    
    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[c.replace('_', ' ').title() for c in CLASS_NAMES],
        yticklabels=[c.replace('_', ' ').title() for c in CLASS_NAMES]
    )
    plt.title('Brain MRI Classification - Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # 3. Multi-Class ROC Curves
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
        # One-vs-Rest ROC
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves by Class', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")
    
    print("\nEvaluation pipeline completed successfully.")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent
    model_file = base_dir / "models" / "final_vgg16_mri_model.keras"
    if not model_file.exists():
        model_file = base_dir / "models" / "best_vgg16_mri_model.keras"
    test_folder = base_dir / "data" / "test"
    report_folder = base_dir / "reports"
    if model_file.exists() and test_folder.exists():
        evaluate_model(str(model_file), str(test_folder), str(report_folder))
    else:
        print("Model file or test folder not found. Please train model or generate data first.")
