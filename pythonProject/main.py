import os
import imageio
from PIL import Image
import numpy as np
import pillow_heif
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import piq
import torch


def compress_image(input_image_path, output_image_base_path):
    """
    Komprimiert ein Bild in die Formate JPEG, JPEG2000, JPEG XR und JPEG XL.

    :param input_image_path: Pfad des Eingabebildes
    :param output_image_base_path: Basis-Pfad für Ausgabedateien
    """
    image = Image.open(input_image_path).convert("RGB")
    compressed_files = []

    # JPEG Komprimierung
    jpeg_output_path = f"{output_image_base_path}_JPEG.jpg"
    image.save(jpeg_output_path, "JPEG", quality=85)
    compressed_files.append(jpeg_output_path)
    print(f"Bild gespeichert als: {jpeg_output_path}")

    # JPEG2000 Komprimierung
    jpeg2000_output_path = f"{output_image_base_path}_JPEG2000.jp2"
    image.save(jpeg2000_output_path, "JPEG2000", quality=85)
    compressed_files.append(jpeg2000_output_path)
    print(f"Bild gespeichert als: {jpeg2000_output_path}")

    # JPEG XR (HEIF) Komprimierung
    jpeg_xr_output_path = f"{output_image_base_path}_JPEG_XR.heic"
    pillow_heif.register_heif_opener()  # Registrieren des HEIF-Openers
    image.save(jpeg_xr_output_path, "HEIF", quality=85)
    compressed_files.append(jpeg_xr_output_path)
    print(f"Bild gespeichert als: {jpeg_xr_output_path}")

    return compressed_files


def compute_quality_metrics(original_image_path, compressed_image_path):
    """
    Berechnet PSNR, SSIM (Full-Reference) und BRISQUE (No-Reference) für ein Bild.

    :param original_image_path: Pfad zum Originalbild
    :param compressed_image_path: Pfad zum komprimierten Bild
    """
    original = np.array(Image.open(original_image_path).convert("RGB"))
    compressed = np.array(Image.open(compressed_image_path).convert("RGB"))

    # Prüfen, ob das komprimierte Bild identisch mit dem Original ist
    if np.array_equal(original, compressed):
        print(f"Warnung: {compressed_image_path} ist identisch mit dem Original. PSNR ist unendlich.")
        psnr_value = float('inf')
        ssim_value = 1.0
    else:
        # Berechnung von PSNR und SSIM
        data_range = compressed.max() - compressed.min()
        if data_range == 0:
            print(f"Warnung: {compressed_image_path} hat keine Intensitätsvariation. PSNR nicht berechenbar.")
            psnr_value = float('inf')
            ssim_value = 1.0
        else:
            psnr_value = psnr(original, compressed, data_range=data_range)

            # Dynamische win_size für SSIM bestimmen
            min_dim = min(original.shape[0], original.shape[1])
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # Sicherstellen, dass win_size ungerade ist
            ssim_value = ssim(original, compressed, channel_axis=-1, data_range=data_range, win_size=win_size)

    # BRISQUE berechnen (No-Reference Metrik)
    compressed_torch = torch.tensor(compressed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    brisque_value = piq.brisque(compressed_torch).item()

    print(f"Qualitätsmetriken für {compressed_image_path}:")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"BRISQUE: {brisque_value:.2f} (niedriger ist besser)")
    return psnr_value, ssim_value, brisque_value


input_image = "img_1.png"
output_image_base_path = "output_compressed"

compressed_images = compress_image(input_image, output_image_base_path)

if compressed_images:
    for compressed_image in compressed_images:
        compute_quality_metrics(input_image, compressed_image)
else:
    print("Fehler: Keine komprimierten Bilder erzeugt.")
