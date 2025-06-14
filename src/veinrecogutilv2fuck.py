import cv2
import numpy as np

def enhance_vein_image(image):
    """
    Enhances the finger vein image using adaptive histogram equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def find_vein_matches(
    image1_path,
    image2_path,
    min_match_count=10,
    ratio_thresh=0.75,
    ransac_thresh=3.0,
    inlier_ratio_thresh=0.5,
    distance_thresh=150
):
    """
    Compares two finger vein images and determines if they are a match.
    Returns (is_match, visualization_image or None).
    """
    try:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            raise FileNotFoundError("One of the images couldn't be loaded.")

        # Enhance contrast
        img1 = enhance_vein_image(img1)
        img2 = enhance_vein_image(img2)

        # Detect SIFT keypoints/descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return False#, None

        # FLANN matcher setup
        index_params = dict(algorithm=1, trees=5)  # KD-tree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance and m.distance < distance_thresh:
                good_matches.append(m)

        if len(good_matches) < min_match_count:
            return False#, None

        # Extract matching keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is None:
            return False#, None

        inliers = mask.ravel().tolist()
        inlier_count = sum(inliers)
        inlier_ratio = inlier_count / len(good_matches)

        if inlier_count >= min_match_count and inlier_ratio >= inlier_ratio_thresh:
            return True#, None
        else:
            return False#, None

    except Exception as e:
        print(f"Error: {e}")
        return False#, None

# Example usage
if __name__ == '__main__':
    image1 = 'finger1.png'
    image2 = 'finger2.png'
    is_match, _ = find_vein_matches(image1, image2)
    print("Match" if is_match else "No Match")
