import os
import pandas as pd
import numpy as np
import cv2
import imagecodecs
import re

def preprocess_image(img):
    """Preprocess finger vein image: enhance vein patterns using Gabor filters."""
    if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # Already grayscale
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Gabor filter
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(blurred, cv2.CV_8UC3, gabor_kernel)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    return enhanced

def extract_features(img):
    """Extract features from preprocessed image using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two images using FLANN-based matcher."""
    if desc1 is None or desc2 is None:
        return 0, 0
    
    # Convert descriptors to CV_32F for FLANN
    if desc1.dtype != np.float32:
        desc1 = np.float32(desc1)
    if desc2.dtype != np.float32:
        desc2 = np.float32(desc2)

    index_params = dict(algorithm=1, trees=5)  # FLANN parameters for KDTree
    search_params = dict(checks=100)  # Increase checks for more exhaustive search
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]  # Relax threshold
    match_score = len(good_matches) / ((len(desc1) + len(desc2)) / 2) if (len(desc1) + len(desc2)) > 0 else 0
    return len(good_matches), match_score

def load_image_with_imagecodecs(file_path):
    """Load an image using imagecodecs for unsupported formats (jxr, jxl)."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        img = imagecodecs.imread(data)
        return img
    except Exception as e:
        print(f"Error loading image with imagecodecs: {file_path}, {e}")
        return None

def parse_filename(filename):
    """Parse the filename to extract scanner, dorsal/palmar, session ID, user ID, finger ID, and image ID."""
    pattern = r"^(.*?)_(DORSAL|PALMAR)_(\d+)_(\d+)_(\d+)_(\d+)\.png$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match expected format: {filename}")
    return match.groups()

def vein_recog_baseline(org_img_path, org_dir, output_path):
    """
    Perform finger vein recognition by comparing images of the same finger.
    
    Parameters:
    org_img_path (str): Path to the original image
    org_dir (str): Directory containing all original images
    output_path (str): Path to save the result CSV
    """
    # Parse the original image filename
    org_filename = os.path.basename(org_img_path)
    scanner, hand, session_id, user_id, finger_id, _ = parse_filename(org_filename)
    
    # Load original image
    original = cv2.imread(org_img_path)
    if original is None:
        raise ValueError(f"Cannot load original image: {org_img_path}")
    
    # Preprocess original image
    org_preprocessed = preprocess_image(original)
    org_kp, org_desc = extract_features(org_preprocessed)
    
    # Find matching images in the directory
    matching_images = []
    for file in os.listdir(org_dir):
        if file == org_filename:
            continue
        try:
            _, _, _, uid, fid, _ = parse_filename(file)
            if uid == user_id and fid == finger_id:
                matching_images.append(os.path.join(org_dir, file))
        except ValueError:
            continue  # Skip files that don't match the expected format
    
    # Results storage
    results = {
        'comparison_image': [],
        'num_keypoints_original': [],
        'num_keypoints_comparison': [],
        'num_matches': [],
        'match_score': []
    }
    
    # Compare with matching images
    for comp_path in matching_images:
        comp_img = cv2.imread(comp_path)
        if comp_img is None:
            print(f"Cannot load comparison image: {comp_path}")
            continue
        
        # Align and preprocess comparison image
        aligned_comp_img = align_images(original, comp_img)
        comp_preprocessed = preprocess_image(aligned_comp_img)
        comp_kp, comp_desc = extract_features(comp_preprocessed)
        
        # Match features
        num_matches, match_score = match_features(org_desc, comp_desc)
        
        # Store results
        results['comparison_image'].append(os.path.basename(comp_path))
        results['num_keypoints_original'].append(len(org_kp))
        results['num_keypoints_comparison'].append(len(comp_kp))
        results['num_matches'].append(num_matches)
        results['match_score'].append(match_score)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

def align_images(img1, img2):
    """Align img2 to img1 using keypoints."""
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    
    # Check if descriptors are valid
    if desc1 is None or desc2 is None:
        print("Descriptors are None, skipping alignment.")
        return img2  # Return the original comparison image if alignment fails
    
    # Ensure descriptor types match
    if desc1.dtype != np.uint8:
        desc1 = np.uint8(desc1)
    if desc2.dtype != np.uint8:
        desc2 = np.uint8(desc2)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) > 10:  # Minimum matches required for alignment
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned_img = cv2.warpPerspective(img2, matrix, (img1.shape[1], img1.shape[0]))
        return aligned_img
    else:
        print("Not enough matches for alignment, skipping.")
    return img2

if __name__ == "__main__":
    # Example usage
    org_img_path = "path/to/original_image.png"
    org_dir_path = "path/to/original_images/"
    output_dir = "path/to/output/"
    output_file = os.path.join(output_dir, "results.csv")
    vein_recog_baseline(org_img_path, org_dir_path, output_file)