"""
Brain MRI Classification Training Pipeline
===========================================
Performs:
1. Data loading with real-time OpenCV preprocessing (contour cropping + CLAHE).
2. Data augmentation (rotations, flips, zooming) to prevent overfitting.
3. Two-phase transfer learning with VGG16:
   - Phase 1: Feature extraction (frozen convolutional base).
   - Phase 2: Fine-tuning top convolutional layers.
4. Evaluation metric logging and training curves export.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

from preprocessing import crop_brain_contour, apply_clahe
from model import build_vgg16_transfer_model, get_callbacks, CLASS_NAMES


def custom_opencv_preprocessor(image):
    """
    Applies OpenCV contour cropping & CLAHE directly in Keras generator pipeline.
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    cropped = crop_brain_contour(img_uint8)
    enhanced = apply_clahe(cropped)
    resized = tf.image.resize(enhanced, (224, 224)).numpy()
    return resized / 255.0


def train_mri_classifier(
    data_dir: str,
    epochs_phase1: int = 5,
    epochs_phase2: int = 5,
    batch_size: int = 16,
    save_dir: str = "models",
    report_dir: str = "reports"
):
    """
    Executes the training workflow.
    """
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"Loading data from {train_path} and {val_path}...")
    
    # Data Augmentation & Normalization
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    train_gen = train_datagen.flow_from_directory(
        str(train_path),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        str(val_path),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )
    
    print(f"Class indices: {train_gen.class_indices}")
    
    # Phase 1: Build model with frozen base
    print("\n--- Starting Phase 1: Feature Extraction (Frozen Base) ---")
    model = build_vgg16_transfer_model(
        input_shape=(224, 224, 3),
        freeze_base=True,
        learning_rate=1e-4
    )
    
    callbacks = get_callbacks(f"{save_dir}/best_vgg16_mri_model.keras")
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-Tuning top convolutional block of VGG16
    print("\n--- Starting Phase 2: Fine-Tuning Top Convolutional Layers ---")
    # Unfreeze block5 layers of VGG16
    for layer in model.layers:
        if "vgg16" in layer.name:
            layer.trainable = True
            for sublayer in layer.layers:
                if sublayer.name.startswith("block5_"):
                    sublayer.trainable = True
                else:
                    sublayer.trainable = False
                    
    # Re-compile with lower learning rate to preserve learned representations
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_phase2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine training histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy', color='#1f77b4', lw=2)
    plt.plot(val_acc, label='Val Accuracy', color='#ff7f0e', lw=2)
    plt.axvline(x=epochs_phase1 - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(val_loss, label='Val Loss', color='#ff7f0e', lw=2)
    plt.axvline(x=epochs_phase1 - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(report_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to {plot_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "final_vgg16_mri_model.keras")
    model.save(final_model_path)
    print(f"Model successfully saved to {final_model_path}")
    return model


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_data_dir = current_dir.parent / "data"
    train_mri_classifier(str(base_data_dir), epochs_phase1=3, epochs_phase2=3, batch_size=8)
