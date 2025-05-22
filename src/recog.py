import os
import pandas as pd
import numpy as np
import cv2
import imagecodecs

def preprocess_image(img):
    """Preprocess finger vein image: convert to grayscale, enhance contrast, and apply thresholding."""
    # Convert to grayscale if necessary
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # Already grayscale
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding to highlight vein patterns
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def extract_features(img):
    """Extract features from preprocessed image using ORB."""
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two images using BFMatcher."""
    if desc1 is None or desc2 is None:
        return 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Calculate match score (number of good matches)
    good_matches = [m for m in matches if m.distance < 50]
    match_score = len(good_matches) / max(len(desc1), len(desc2)) if max(len(desc1), len(desc2)) > 0 else 0
    return len(good_matches), match_score

def load_image_with_imagecodecs(file_path):
    """
    Load an image using imagecodecs for unsupported formats (jxr, jxl).
    
    Parameters:
    file_path (str): Path to the image file.
    
    Returns:
    np.ndarray: Decoded image as a NumPy array.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        img = imagecodecs.imread(data)
        return img
    except Exception as e:
        print(f"Error loading image with imagecodecs: {file_path}, {e}")
        return None

def vein_recog(org_img, comp_dir, output_path, filename):
    """
    Perform finger vein recognition by comparing original image with compressed versions.
    
    Parameters:
    org_img (str): Path to the original image
    comp_dir (str): Directory containing compressed images (jpg, jp2, jxr, jxl)
    output_path (str): Path to save the result CSV
    filename (str): Base filename for the images (without extension)
    """
    # Load original image
    original = cv2.imread(org_img)
    if original is None:
        raise ValueError(f"Cannot load original image: {org_img}")
    
    # Preprocess original image
    org_preprocessed = preprocess_image(original)
    org_kp, org_desc = extract_features(org_preprocessed)
    
    # Supported compression formats
    formats = ['jpg', 'jp2', 'jxr', 'jxl']
    
    # Results storage
    results = {
        'format': [],
        'num_keypoints_original': [],
        'num_keypoints_compressed': [],
        'num_matches': [],
        'match_score': []
    }
    
    # Process each compressed image
    for fmt in formats:
        comp_path = os.path.join(comp_dir, f"{filename}.{fmt}")
        if not os.path.exists(comp_path):
            print(f"Compressed image not found: {comp_path}")
            continue
        
        # Load compressed image
        if fmt in ['jxr', 'jxl']:
            comp_img = load_image_with_imagecodecs(comp_path)
        else:
            comp_img = cv2.imread(comp_path)
        
        if comp_img is None:
            print(f"Cannot load compressed image: {comp_path}")
            continue
        
        # Convert to grayscale if necessary
        if len(comp_img.shape) == 3 and comp_img.shape[2] == 3:  # RGB
            comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2GRAY)
        
        # Preprocess compressed image
        comp_preprocessed = preprocess_image(comp_img)
        comp_kp, comp_desc = extract_features(comp_preprocessed)
        
        # Match features
        num_matches, match_score = match_features(org_desc, comp_desc)
        
        # Store results
        results['format'].append(fmt)
        results['num_keypoints_original'].append(len(org_kp))
        results['num_keypoints_compressed'].append(len(comp_kp))
        results['num_matches'].append(num_matches)
        results['match_score'].append(match_score)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage
    org_img_path = "path/to/original.png"
    comp_dir_path = "path/to/compressed/"
    output_dir = "path/to/output/"
    fname = "test_image"
    vein_recog(org_img_path, comp_dir_path, output_dir, fname)