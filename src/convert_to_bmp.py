import os
import cv2

def convert_images_to_bmp_grayscale(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            img = cv2.imread(input_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}.bmp")
                cv2.imwrite(output_path, gray)
                print(f"Saved grayscale BMP: {output_path}")
            else:
                print(f"Skipped (not an image): {input_path}")

# Example usage:
convert_images_to_bmp_grayscale('./input/genuineORG', './input/genuineORG_bmp_gray')