"""
Synthetic Brain MRI Dataset Generator
======================================
Generates realistic simulated brain MRI images across 4 tumor classes:
- glioma_tumor (diffuse, irregular lesion with varying intensity)
- meningioma_tumor (well-defined, spherical mass near outer meningeal layers)
- pituitary_tumor (mass localized at base of brain / sella turcica)
- no_tumor (healthy brain structure with symmetric ventricles)

Creates standard train / val / test folder structure:
  data/
    train/
      glioma_tumor/
      meningioma_tumor/
      no_tumor/
      pituitary_tumor/
    val/
    test/
"""

import os
import cv2
import numpy as np
from pathlib import Path


def generate_brain_mri_slice(
    tumor_type: str,
    img_size: int = 224,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Generates a single synthetic brain MRI axial slice.
    """
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    
    # 1. Outer skull outline (dim oval)
    skull_axes = (int(img_size * 0.42), int(img_size * 0.45))
    cv2.ellipse(canvas, center, skull_axes, 0, 0, 360, 40, -1)
    
    # 2. Brain parenchymal tissue (oval inside skull)
    brain_axes = (int(img_size * 0.38), int(img_size * 0.42))
    cv2.ellipse(canvas, center, brain_axes, 0, 0, 360, 110, -1)
    
    # 3. Add sulci/gyri texture variations
    noise = np.random.normal(0, 15, (img_size, img_size)).astype(np.float32)
    brain_mask = canvas > 50
    temp = canvas.astype(np.float32)
    temp[brain_mask] += noise[brain_mask]
    canvas = np.clip(temp, 0, 255).astype(np.uint8)
    
    # 4. Ventricles (central butterfly-like dark cavities)
    v_left = (int(center[0] - img_size * 0.06), center[1])
    v_right = (int(center[0] + img_size * 0.06), center[1])
    cv2.ellipse(canvas, v_left, (int(img_size * 0.04), int(img_size * 0.12)), -15, 0, 360, 25, -1)
    cv2.ellipse(canvas, v_right, (int(img_size * 0.04), int(img_size * 0.12)), 15, 0, 360, 25, -1)
    
    # 5. Class-specific lesion injection
    if tumor_type == 'glioma_tumor':
        # Diffuse, infiltrating hyperintense mass in cerebral hemisphere
        tx = int(center[0] + np.random.choice([-1, 1]) * img_size * 0.18)
        ty = int(center[1] + np.random.uniform(-0.15, 0.15) * img_size)
        axes = (int(img_size * 0.12), int(img_size * 0.09))
        cv2.ellipse(canvas, (tx, ty), axes, np.random.randint(0, 180), 0, 360, 210, -1)
        # Infiltrative fuzzy edges
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        
    elif tumor_type == 'meningioma_tumor':
        # Dense, circumscribed hyperintense mass adjacent to dura/skull
        angle = np.random.uniform(0.2, 0.8) * np.pi
        dist = img_size * 0.32
        tx = int(center[0] + dist * np.cos(angle))
        ty = int(center[1] + dist * np.sin(angle))
        radius = int(img_size * 0.08)
        cv2.circle(canvas, (tx, ty), radius, 240, -1)
        # Dense core
        cv2.circle(canvas, (tx, ty), int(radius * 0.5), 255, -1)
        
    elif tumor_type == 'pituitary_tumor':
        # Sella turcica location: central lower midline
        tx = center[0] + np.random.randint(-3, 3)
        ty = int(center[1] + img_size * 0.19)
        axes = (int(img_size * 0.07), int(img_size * 0.06))
        cv2.ellipse(canvas, (tx, ty), axes, 0, 0, 360, 230, -1)
        
    # 'no_tumor' has no extra lesion added
    
    # 6. Global MRI acquisition noise (Rician/Gaussian artifact)
    rf_noise = np.random.normal(0, noise_level * 255, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + rf_noise, 0, 255).astype(np.uint8)
    
    # Convert to 3-channel (RGB representation)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return bgr


def create_dataset_structure(
    base_dir: str = "C:/Users/Krisha Jariwala/.gemini/antigravity/scratch/data_analyst_portfolio/01_brain_mri_classification/data",
    samples_per_class: dict = None
):
    """
    Builds the dataset directory and populates it with synthetic samples.
    """
    if samples_per_class is None:
        samples_per_class = {
            'train': 60,   # Generates 240 train images
            'val': 20,     # Generates 80 val images
            'test': 20     # Generates 80 test images
        }
        
    classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    base_path = Path(base_dir)
    
    print(f"Creating synthetic Brain MRI dataset at {base_path}...")
    
    for split, count in samples_per_class.items():
        for cls in classes:
            split_cls_dir = base_path / split / cls
            split_cls_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                img = generate_brain_mri_slice(cls)
                filename = split_cls_dir / f"{cls}_{split}_{i+1:04d}.jpg"
                cv2.imwrite(str(filename), img)
                
    print("Dataset generation completed successfully.")
    print("Classes generated:", classes)


if __name__ == "__main__":
    create_dataset_structure()
