import os
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs('./output/visualize', exist_ok=True)

def save_plot(title, img, cmap='gray', step=None):
    """Save visualization of an image with optional step numbering"""
    ensure_output_dir()
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    if step is not None:
        filename = f"./output/visualize/{title.lower().replace(' ', '_')}.png"
    else:
        filename = f"./output/visualize/{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

def preprocess_vein_image(img, prefix=""):
    """Robust preprocessing with visualization and edge cropping"""
    try:
        step_counter = 1
        original_img = None
        
        if isinstance(img, str):
            original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                raise ValueError("Image loading failed")
            save_plot(f"{prefix}1 Original Image", original_img, step=step_counter)
            step_counter += 1
            img = original_img.copy()
        else:
            original_img = img.copy()
            save_plot(f"{prefix}1 Original Image", original_img, step=step_counter)
            step_counter += 1

        # Edge cropping (30 pixels from each side)
        crop_size = 30
        if img.shape[0] > 2*crop_size and img.shape[1] > 2*crop_size:
            img = img[crop_size:-crop_size, crop_size:-crop_size]
            original_img = original_img[crop_size:-crop_size, crop_size:-crop_size]
            save_plot(f"{prefix}2 Cropped Image", img, step=step_counter)
            step_counter += 1
        else:
            print(f"Warning: Image too small for {crop_size}px cropping. Original size: {img.shape}")
            save_plot(f"{prefix}2 Original (No Cropping)", img, step=step_counter)
            step_counter += 1

        # Noise reduction and thresholding
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        save_plot(f"{prefix}3 Gaussian Blurred", blurred, step=step_counter)
        step_counter += 1

        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 2
        )
        save_plot(f"{prefix}4 Adaptive Threshold", binary, step=step_counter)
        step_counter += 1

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        save_plot(f"{prefix}5 Morphology Kernel", kernel*255, step=step_counter)
        step_counter += 1

        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        save_plot(f"{prefix}6 After Morphological Close", morph, step=step_counter)
        step_counter += 1

        # Find largest contour
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, np.eye(3)

        # Visualize contours
        contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Use cropped image
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 0, 255), 3)
        save_plot(f"{prefix}7 Detected Contours", contour_img, step=step_counter)
        step_counter += 1

        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.float32(box)

        # Visualize bounding box
        box_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Use cropped image
        cv2.drawContours(box_img, [np.int0(box)], 0, (0, 255, 255), 2)
        save_plot(f"{prefix}8 Min Area Rectangle", box_img, step=step_counter)
        step_counter += 1

        # Get perspective transform
        width, height = int(rect[1][0]), int(rect[1][1])
        dst_pts = np.float32([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]])
        M = cv2.getPerspectiveTransform(box, dst_pts)

        # Warp and enhance
        warped = cv2.warpPerspective(img, M, (width, height))
        save_plot(f"{prefix}9 Warped Image", warped, step=step_counter)
        step_counter += 1

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        enhanced = clahe.apply(warped)
        save_plot(f"{prefix}10 CLAHE Enhanced", enhanced, step=step_counter)
        step_counter += 1

        return enhanced, M
        
    except Exception as e:
        print(f"Preprocessing warning: {str(e)}")
        return img, np.eye(3)

def extract_vein_pattern(img, prefix=""):
    """Improved vein pattern extraction with better parameter tuning"""
    try:
        step_counter = 1
        save_plot(f"{prefix}10 Input for Vein Extraction", img, step=step_counter)
        step_counter += 1

        # Multi-scale Frangi filter with enhanced parameters
        vein_enhanced = np.zeros_like(img, dtype=np.float32)
        
        # Adjusted sigma range and scale count
        sigmas = np.linspace(2.0, 4.0, 8)  # More scales with wider range
        beta = 0.5  # Frangi correction parameter
        gamma = 10   # Frangi correction parameter
        
        for i, sigma in enumerate(sigmas):
            frangi_result = frangi(
                img,
                sigmas=[sigma],
                black_ridges=True,
                beta=beta,
                gamma=gamma,
                alpha=0.5  # Sensibility to blob-like structures
            )
            vein_enhanced = np.maximum(vein_enhanced, frangi_result)
            save_plot(f"{prefix}11 Frangi Filter sigma={sigma:.1f}", frangi_result, step=step_counter+i)
        step_counter += len(sigmas)

        # Enhanced normalization with percentile clipping
        p_low, p_high = np.percentile(vein_enhanced, (1, 99))
        vein_enhanced = np.clip(vein_enhanced, p_low, p_high)
        vein_enhanced = cv2.normalize(vein_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        save_plot(f"{prefix}12 Combined Frangi Results", vein_enhanced, step=step_counter)
        step_counter += 1

        # Adaptive thresholding instead of Otsu
        binary = cv2.adaptiveThreshold(
            vein_enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=21,
            C=2
        )
        save_plot(f"{prefix}13 Adaptive Thresholded Veins", binary, step=step_counter)
        step_counter += 1

        # Improved morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        save_plot(f"{prefix}14 After Morphological Cleaning", cleaned, step=step_counter)
        step_counter += 1

        # Skeletonization with area filtering
        skeleton = skeletonize(cleaned // 255).astype(np.uint8) * 255
        
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 15:  # Remove small components
                skeleton[labels == i] = 0
        
        save_plot(f"{prefix}15 Final Skeletonized Veins", skeleton, step=step_counter)
        step_counter += 1

        return skeleton
        
    except Exception as e:
        print(f"Vein extraction warning: {str(e)}")
        return img

def align_vein_patterns(vein1, vein2, M1, M2, prefix1="", prefix2=""):
    """Safe pattern alignment with visualization"""
    try:
        step_counter = 1
        h, w = vein1.shape
        
        # Convert to proper types
        vein1 = vein1.astype(np.uint8)
        vein2 = vein2.astype(np.uint8)

        # Visualize inputs
        save_plot(f"{prefix1}17 Vein Pattern Before Alignment", vein1, step=step_counter)
        save_plot(f"{prefix2}17 Vein Pattern Before Alignment", vein2, step=step_counter)
        step_counter += 1

        # Warp images
        aligned1 = cv2.warpPerspective(
            vein1, 
            np.linalg.inv(M1), 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        aligned2 = cv2.warpPerspective(
            vein2, 
            np.linalg.inv(M2), 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        save_plot(f"{prefix1}18 Warped to Original Space", aligned1, step=step_counter)
        save_plot(f"{prefix2}18 Warped to Original Space", aligned2, step=step_counter)
        step_counter += 1

        # Find overlap (proper type conversion)
        mask1 = (aligned1 > 0).astype(np.uint8)
        mask2 = (aligned2 > 0).astype(np.uint8)
        mask = cv2.bitwise_and(mask1, mask2)

        save_plot(f"{prefix1}19 Overlap Mask", mask*255, step=step_counter)
        step_counter += 1

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return vein1, vein2

        # Get ROI
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w < 50 or h < 50:
            return vein1, vein2

        # Visualize ROI
        roi_img1 = cv2.cvtColor(aligned1, cv2.COLOR_GRAY2BGR)
        roi_img2 = cv2.cvtColor(aligned2, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(roi_img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(roi_img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        save_plot(f"{prefix1}20 Detected ROI", roi_img1, step=step_counter)
        save_plot(f"{prefix2}20 Detected ROI", roi_img2, step=step_counter)
        step_counter += 1

        result1 = aligned1[y:y+h, x:x+w]
        result2 = aligned2[y:y+h, x:x+w]

        save_plot(f"{prefix1}21 Final Aligned Veins", result1, step=step_counter)
        save_plot(f"{prefix2}21 Final Aligned Veins", result2, step=step_counter)

        return result1, result2
        
    except Exception as e:
        print(f"Alignment warning: {str(e)}")
        return vein1, vein2

def calculate_similarity(vein1, vein2, prefix=""):
    """Robust similarity calculation with visualization"""
    try:
        step_counter = 1
        # Ensure minimum size
        min_size = 50
        if min(vein1.shape[0], vein1.shape[1]) < min_size or min(vein2.shape[0], vein2.shape[1]) < min_size:
            return {'ncc': 0, 'ssim': 0, 'dice': 0, 'jaccard': 0}

        # Visualize inputs
        save_plot(f"{prefix}22 Vein Pattern 1 for Matching", vein1, step=step_counter)
        save_plot(f"{prefix}22 Vein Pattern 2 for Matching", vein2, step=step_counter)
        step_counter += 1

        # Resize maintaining aspect ratio
        #scale = min(300/vein1.shape[1], 300/vein1.shape[0])
        #size = (int(vein1.shape[1]*scale), int(vein1.shape[0]*scale))
        #v1 = cv2.resize(vein1, size).astype(np.float32) / 255.0
        #v2 = cv2.resize(vein2, size).astype(np.float32) / 255.0

        # Visualize resized
        #save_plot(f"{prefix}23 Resized Pattern 1", v1, step=step_counter)
        #save_plot(f"{prefix}23 Resized Pattern 2", v2, step=step_counter)
        #step_counter += 1

        v1 = vein1.astype(np.float32) / 255.0
        v2 = vein2.astype(np.float32) / 255.0

        # NCC
        ncc = cv2.matchTemplate(v1, v2, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # SSIM (simplified)
        ssim = cv2.SSIM(v1, v2) if hasattr(cv2, 'SSIM') else 0.5
        
        # Dice and Jaccard
        intersection = np.sum(v1 * v2)
        dice = (2. * intersection) / (np.sum(v1) + np.sum(v2) + 1e-8)
        jaccard = intersection / (np.sum(np.maximum(v1, v2)) + 1e-8)

        # Create visualization of overlap
        overlap = cv2.addWeighted(cv2.cvtColor(v1, cv2.COLOR_GRAY2BGR), 0.5, 
                                 cv2.cvtColor(v2, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.putText(overlap, f"NCC: {ncc:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(overlap, f"Dice: {dice:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        save_plot(f"{prefix}24 Similarity Visualization", overlap, step=step_counter)

        return {
            'ncc': float(ncc),
            'ssim': float(ssim),
            'dice': float(dice),
            'jaccard': float(jaccard)
        }
        
    except Exception as e:
        print(f"Similarity calculation warning: {str(e)}")
        return {'ncc': 0, 'ssim': 0, 'dice': 0, 'jaccard': 0}

def compare_finger_veins(img_path1, img_path2, threshold=0.35, display=False):
    """Main comparison function with visualization"""
    try:
        ensure_output_dir()
        
        # Preprocess
        img1, M1 = preprocess_vein_image(img_path1, "img1_")
        img2, M2 = preprocess_vein_image(img_path2, "img2_")
        
        if img1 is None or img2 is None:
            raise ValueError("Invalid image(s)")
        
        # Extract veins
        vein1 = extract_vein_pattern(img1, "img1_")
        vein2 = extract_vein_pattern(img2, "img2_")
        
        # Align
        aligned1, aligned2 = align_vein_patterns(vein1, vein2, M1, M2, "img1_", "img2_")
        
        # Calculate similarity
        metrics = calculate_similarity(aligned1, aligned2, "comparison_")
        
        # Dynamic threshold
        quality = min(np.mean(vein1)/255, np.mean(vein2)/255)
        adj_threshold = max(0.25, threshold * (1 + (0.5 - quality)))
        
        # Match decision
        match_score = (metrics['ncc'] + metrics['dice']) / 2
        decision = match_score >= adj_threshold
        confidence = min(max(0, (match_score - threshold + 0.5) * 2), 1)
        
        # Create final result visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(aligned1, cmap='gray')
        ax[0].set_title('Vein Pattern 1')
        ax[1].imshow(aligned2, cmap='gray')
        ax[1].set_title('Vein Pattern 2')
        plt.suptitle(f"Match: {decision} (Confidence: {confidence:.2f})\n"
                    f"NCC: {metrics['ncc']:.2f}, Dice: {metrics['dice']:.2f}")
        plt.savefig('./output/visualize/final_comparison_result.png')
        plt.close()

        return {
            'similarity_metrics': metrics,
            'match_decision': bool(decision),
            'match_confidence': float(confidence),
            'adjusted_threshold': float(adj_threshold),
            'original_threshold': float(threshold)
        }
        
    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return {
            'similarity_metrics': {'ncc': 0, 'ssim': 0, 'dice': 0, 'jaccard': 0},
            'match_decision': False,
            'match_confidence': 0.0,
            'error': str(e)
        }