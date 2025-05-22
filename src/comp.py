import imagecodecs
from imagecodecs import (
    jpeg_encode,
    jpeg2k_encode,
    jpegxr_encode,
    jpegxl_encode
)
import numpy as np
import os


def adjust_quality_for_size(encoded_data: bytes, target_size: int, encode_function, image, **kwargs):
    """Adjusts quality level to reach target file size."""
    quality = 99  # Start with high quality
    step = 0.1  # Step size for quality adjustment
    
    while len(encoded_data) > target_size and quality > 1:
        quality -= step
        encoded_data = encode_function(image, level=quality, **kwargs)
    
    return encoded_data

def compress_image(input_path: str, output_prefix: str, target_size: int):
    # Load the PNG image as a NumPy array
    image = imagecodecs.imread(input_path)
    
    # Convert to 8-bit if necessary (some formats don't support higher bit depths)
    if image.dtype != np.uint8:
        image = (image / image.max() * 255).astype(np.uint8)
    
    # JPEG
    encoded_jpeg = imagecodecs.jpeg_encode(image, level=90)
    encoded_jpeg = adjust_quality_for_size(encoded_jpeg, target_size, imagecodecs.jpeg_encode, image)
    with open(f"{output_prefix}.jpg", "wb") as f:
        f.write(encoded_jpeg)
    
    # JPEG 2000
    encoded_jp2 = imagecodecs.jpeg2k_encode(image, level=90)
    encoded_jp2 = adjust_quality_for_size(encoded_jp2, target_size, imagecodecs.jpeg2k_encode, image)
    with open(f"{output_prefix}.jp2", "wb") as f:
        f.write(encoded_jp2)
    
    # JPEG XR
    encoded_jxr = imagecodecs.jpegxr_encode(image, level=90)
    encoded_jxr = adjust_quality_for_size(encoded_jxr, target_size, imagecodecs.jpegxr_encode, image)
    with open(f"{output_prefix}.jxr", "wb") as f:
        f.write(encoded_jxr)
    
    # JPEG XL
    encoded_jxl = imagecodecs.jpegxl_encode(image, effort=7)
    encoded_jxl = adjust_quality_for_size(encoded_jxl, target_size, imagecodecs.jpegxl_encode, image)
    with open(f"{output_prefix}.jxl", "wb") as f:
        f.write(encoded_jxl)

# Example usage
if __name__ == "__main__":
    compress_image("input.png", "output/comp", 4000)  # Target size in bytes
