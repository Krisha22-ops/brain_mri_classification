"""
Brain MRI Scan Preprocessing Pipeline
======================================
Applies OpenCV image processing techniques including:
1. Skull-stripping / extreme contour detection to crop black borders & skull artifacts.
2. Contrast Limited Adaptive Histogram Equalization (CLAHE) for lesion enhancement.
3. Resizing, aspect-ratio preservation, and normalization for VGG16 input.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional


def crop_brain_contour(image: np.ndarray, margin: int = 5) -> np.ndarray:
    """
    Finds the extreme points of the brain contour and crops out the black background
    and peripheral artifacts to focus model attention exclusively on brain tissue.
    
    Args:
        image: BGR or RGB image as numpy array.
        margin: Extra pixel padding around extreme contour boundaries.
        
    Returns:
        Cropped numpy array containing only the brain scan.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Gaussian blur to filter high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding: binary threshold to separate brain mass from black background
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to close small holes and eliminate noise speckles
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
        
    # Pick the largest contour by area (the brain structure)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate extreme boundary points (top, bottom, left, right)
    ext_left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    ext_right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    ext_top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    ext_bot = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
    
    # Bounding box with safety margin
    h, w = image.shape[:2]
    ymin = max(0, ext_top[1] - margin)
    ymax = min(h, ext_bot[1] + margin)
    xmin = max(0, ext_left[0] - margin)
    xmax = min(w, ext_right[0] + margin)
    
    # Crop if valid bounding box
    if (ymax > ymin) and (xmax > xmin):
        cropped = image[ymin:ymax, xmin:xmax]
        return cropped
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to amplify subtle tissue density variations and tumor boundary contrast.
    
    Args:
        image: BGR or RGB image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for local histogram equalization.
        
    Returns:
        Contrast-enhanced image array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3:
        # Convert to LAB color space to apply CLAHE only on Lightness channel (L)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = clahe.apply(l)
        merged_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224), apply_enhancement: bool = True) -> np.ndarray:
    """
    Full end-to-end preprocessing pipeline for a single MRI file.
    
    Steps:
    1. Read image from disk.
    2. Crop brain contour (skull strip / background removal).
    3. Apply CLAHE contrast enhancement.
    4. Resize to target dimension (e.g. 224x224 for VGG16).
    5. Normalize pixel intensity to [0, 1].
    
    Args:
        image_path: Absolute or relative path to MRI image file.
        target_size: (width, height) tuple for CNN input.
        apply_enhancement: Whether to apply CLAHE enhancement.
        
    Returns:
        Float32 array of shape (target_size[0], target_size[1], 3) scaled to [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")
        
    # Step 1: Crop brain contour
    cropped = crop_brain_contour(img)
    
    # Step 2: Contrast enhancement
    if apply_enhancement:
        enhanced = apply_clahe(cropped)
    else:
        enhanced = cropped
        
    # Step 3: Resize
    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    
    # Step 4: Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized
