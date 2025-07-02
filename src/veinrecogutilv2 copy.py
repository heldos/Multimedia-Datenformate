# main.py
import cv2
import numpy as np

def enhance_vein_image(image):
    """
    Enhances the finger vein image using adaptive histogram equalization.

    Args:
        image: The input finger vein image (grayscale).

    Returns:
        The enhanced image.
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def find_vein_matches(image1_path, image2_path, threshold=0.75, min_match_count=10):
    """
    Compares two finger vein images and determines if they are a match.

    Args:
        image1_path: Path to the first finger vein image.
        image2_path: Path to the second finger vein image.
        threshold: Lowe's ratio test threshold for filtering good matches.
        min_match_count: Minimum number of good matches required for a positive match.

    Returns:
        A tuple containing:
        - A boolean indicating if the images are a match.
        - The visualization of the matches.
    """
    try:
        # Load the images in grayscale
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None:
            raise FileNotFoundError(f"Error: Could not load image from {image1_path}")
        if img2 is None:
            raise FileNotFoundError(f"Error: Could not load image from {image2_path}")


        # Enhance the images to make the veins more prominent
        img1_enhanced = enhance_vein_image(img1)
        img2_enhanced = enhance_vein_image(img2)

        # Initialize the SIFT detector
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1_enhanced, None)
        kp2, des2 = sift.detectAndCompute(img2_enhanced, None)

        if des1 is None or des2 is None:
            print("Warning: Could not find descriptors in one or both images.")
            return False, None

        # Use FLANN based matcher for feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        # Draw the matches
        match_img = cv2.drawMatches(img1_enhanced, kp1, img2_enhanced, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        # Check if the number of good matches is above the minimum threshold
        if len(good_matches) > min_match_count:
            #print(f"Match Found! - Good Matches: {len(good_matches)}")
            return True#, match_img
        else:
            #print(f"Not a Match - Good Matches: {len(good_matches)}")
            return False#, match_img

    except FileNotFoundError as e:
        print(e)
        return False#, None
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return False#, None


if __name__ == '__main__':
    # --- Configuration ---
    # Replace with the paths to your finger vein images
    image1 = 'finger1.png'
    image2 = 'finger2.png'
    # To test a non-match, use a different finger image
    # image2 = 'finger3.png'

    # --- Run the matching ---
    is_match, result_image = find_vein_matches(image1, image2)

    # --- Save the results ---
    if result_image is not None:
        # Define the output filename
        output_filename = 'matches_result.png'
        
        # Resize the image for better visualization if it's too large
        scale_percent = 50 # percent of original size
        width = int(result_image.shape[1] * scale_percent / 100)
        height = int(result_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(result_image, dim, interpolation = cv2.INTER_AREA)

        # Save the resulting image to a file
        cv2.imwrite(output_filename, resized)
        print(f"Resulting image saved as {output_filename}")
