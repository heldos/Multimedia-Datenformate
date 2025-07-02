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
    
    while len(encoded_data) > target_size and quality > 0.2:
        quality -= step
        encoded_data = encode_function(image, level=quality, **kwargs)

    print(f"Final quality level: {quality}, Size: {len(encoded_data)} bytes")
    
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
    decoded_jpeg = imagecodecs.jpeg_decode(encoded_jpeg)
    imagecodecs.imwrite(f"{output_prefix}-jpg.png", decoded_jpeg, codec='png')
    
    # JPEG 2000
    encoded_jp2 = imagecodecs.jpeg2k_encode(image, level=90)
    encoded_jp2 = adjust_quality_for_size(encoded_jp2, target_size, imagecodecs.jpeg2k_encode, image)
    decoded_jp2 = imagecodecs.jpeg2k_decode(encoded_jp2)
    imagecodecs.imwrite(f"{output_prefix}-jp2.png", decoded_jp2, codec='png')
    
    # JPEG XR
    encoded_jxr = imagecodecs.jpegxr_encode(image, level=90)
    encoded_jxr = adjust_quality_for_size(encoded_jxr, target_size, imagecodecs.jpegxr_encode, image)
    decoded_jxr = imagecodecs.jpegxr_decode(encoded_jxr)
    imagecodecs.imwrite(f"{output_prefix}-jpxr.png", decoded_jxr, codec='png')
    
    # JPEG XL
    encoded_jxl = imagecodecs.jpegxl_encode(image, effort=7)
    encoded_jxl = adjust_quality_for_size(encoded_jxl, target_size, imagecodecs.jpegxl_encode, image)
    decoded_jxl = imagecodecs.jpegxl_decode(encoded_jxl)
    imagecodecs.imwrite(f"{output_prefix}-jpxl.png", decoded_jxl, codec='png')

# Example usage
if __name__ == "__main__":
    compress_image("input.png", "output/comp", 4000)  # Target size in bytes
