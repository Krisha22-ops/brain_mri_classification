"""
VGG16 Transfer Learning Model Architecture for Brain MRI Classification
========================================================================
Implements a 4-class convolutional neural network using VGG16 pretrained
on ImageNet with custom regularization, batch normalization, and fine-tuning.

Target Classes:
1. Glioma Tumor
2. Meningioma Tumor
3. Pituitary Tumor
4. No Tumor (Healthy Control)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Tuple, List

# Class labels in consistent alphanumeric order
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES = len(CLASS_NAMES)


def build_vgg16_transfer_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    freeze_base: bool = True,
    fine_tune_at_layer: str = "block5_conv1",
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-4
) -> tf.keras.Model:
    """
    Builds and compiles the VGG16 Transfer Learning model.
    
    Args:
        input_shape: Input tensor dimensions (height, width, channels).
        num_classes: Number of target categories (4).
        freeze_base: Whether to freeze initial convolutional weights.
        fine_tune_at_layer: Layer name from which to unfreeze for fine-tuning.
        dropout_rate: Dropout rate for regularization.
        learning_rate: Initial learning rate for Adam optimizer.
        
    Returns:
        Compiled Keras Model.
    """
    try:
        # Load VGG16 backbone without top fully connected layers
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    except Exception as e:
        print(f"[Warning] Could not fetch online ImageNet weights ({e}). Initializing architecture.")
        base_model = tf.keras.applications.VGG16(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )

    # Freeze or selectively unfreeze base layers
    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        set_trainable = False
        for layer in base_model.layers:
            if layer.name == fine_tune_at_layer:
                set_trainable = True
            layer.trainable = set_trainable

    # Construct the custom classification head
    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = base_model(inputs, training=not freeze_base)
    
    # Feature pooling & dense representation
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation='relu', name="dense_fc1")(x)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    
    x = layers.Dense(128, activation='relu', name="dense_fc2")(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="dropout2")(x)
    
    # Final Softmax probability distribution over 4 classes
    outputs = layers.Dense(num_classes, activation='softmax', name="tumor_classification")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Brain_MRI_VGG16_Classifier")
    
    # Compile with Categorical Crossentropy & multi-class metrics
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_callbacks(model_save_path: str = "models/best_vgg16_mri_model.keras") -> List[tf.keras.callbacks.Callback]:
    """
    Returns standard production callbacks for training stability:
    1. ModelCheckpoint: saves weights with highest validation accuracy.
    2. EarlyStopping: halts training if val_loss fails to improve for 7 epochs.
    3. ReduceLROnPlateau: decreases learning rate by factor of 0.2 when validation stalls.
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        mode='min',
        verbose=1
    )
    
    return [checkpoint, early_stop, lr_reduction]
