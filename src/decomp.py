import imagecodecs
import numpy as np
import os

def decompress_and_save(input: str, output_dir: str):
    formats = [
        ("jpg", imagecodecs.jpeg_decode),
        ("jp2", imagecodecs.jpeg2k_decode),
        ("jxr", imagecodecs.jpegxr_decode),
        ("jxl", imagecodecs.jpegxl_decode),
    ]
    os.makedirs(output_dir, exist_ok=True)

    for ext, decode_func in formats:
        input_path = input
        output_path = os.path.join(output_dir, f"{input}.png")
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue
        with open(input_path, "rb") as f:
            encoded = f.read()
        try:
            image = decode_func(encoded)
            # Convert to uint8 if needed for PNG
            if image.dtype != np.uint8:
                image = (image / image.max() * 255).astype(np.uint8)
            imagecodecs.imwrite(output_path, image, codec='png')
        except Exception as e:
            print(f"Failed to decode {input_path}: {e}")

# Example usage
if __name__ == "__main__":
    decompress_and_save("output/comp", "output/decoded_pngs")