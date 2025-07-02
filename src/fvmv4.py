import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def match_finger_veins(img1_path, img2_path, method='template'):
    """
    Match two finger vein images and return a similarity score.

    Args:
        img1_path (str): Path to first image
        img2_path (str): Path to second image
        method (str): Matching method - 'template'

    Returns:
        float: Similarity score between 0 and 1 (1 = perfect match)
    """
    print(f"Matching {img1_path} with {img2_path}")
    try:
        # Load and preprocess images
        img1 = load_and_preprocess(img1_path)
        img2 = load_and_preprocess(img2_path)

        if img1 is None or img2 is None:
            return 0.0

        # Apply different matching methods
        if method == 'template':
            score = template_matching(img1, img2)
        else:
            raise ValueError("Method must be 'template'")

        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

    except Exception as e:
        print(f"Error in finger vein matching: {e}")
        return 0.0

def load_and_preprocess(img_path):
    """Load and preprocess finger vein image."""
    try:
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Resize to standard size
        img = cv2.resize(img, (500, 130))

        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
        img = clahe.apply(img)

        # Normalize pixel values
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Apply vein enhancement filter
        img = enhance_veins(img)

        return img

    except Exception as e:
        print(f"Error preprocessing image {img_path}: {e}")
        return None

def enhance_veins(img):
    """Enhance vein patterns using morphological operations."""
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Top-hat transform to enhance bright regions (veins)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Black-hat transform to enhance dark regions
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # Combine transforms
    enhanced = cv2.add(img, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)

    return enhanced

def template_matching(img1, img2):
    """Perform template matching between two images."""
    try:
        # Normalize images
        img1_norm = cv2.normalize(img1.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        img2_norm = cv2.normalize(img2.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        # Template matching using normalized cross correlation
        result = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCOEFF_NORMED)

        # Get maximum correlation value
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max(max_val, 0.0)

    except Exception:
        return 0.0

def match_finger_veins_from_arrays(img1_array, img2_array, method='hybrid'):
    """
    Match two finger vein images provided as numpy arrays.

    Args:
        img1_array (numpy.ndarray): First image as numpy array
        img2_array (numpy.ndarray): Second image as numpy array
        method (str): Matching method - 'template'

    Returns:
        float: Similarity score between 0 and 1 (1 = perfect match)
    """
    try:
        # Preprocess arrays
        img1 = preprocess_array(img1_array)
        img2 = preprocess_array(img2_array)

        if img1 is None or img2 is None:
            return 0.0

        # Apply matching method
        if method == 'template':
            score = template_matching(img1, img2)
        else:
            raise ValueError("Method must be 'template'")

        return min(max(score, 0.0), 1.0)

    except Exception as e:
        print(f"Error in finger vein matching: {e}")
        return 0.0

def preprocess_array(img_array):
    """Preprocess image array."""
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            img = img_array.copy()

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize to standard size
        img = cv2.resize(img, (256, 64))

        # Apply preprocessing steps
        img = cv2.GaussianBlur(img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = enhance_veins(img)

        return img

    except Exception as e:
        print(f"Error preprocessing array: {e}")
        return None