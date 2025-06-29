# https://www.youtube.com/watch?v=wOdfNwD9cEA&t=13s
import time
import PIL.Image
import cv2
import PIL
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import bob.bio.vein.preprocessor as bvp
import bob.bio.vein.extractor as bve
import bob.bio.vein.algorithm as bva
import bob.io.image as bi

def vein_recognition(fi, fr):
    """
    Perform finger vein recognition using the Bob library with the repeated-line tracking pipeline.

    Parameters:
    input_image_path (str): Path to the input finger vein image.
    reference_image_path (str): Path to the reference finger vein image.

    Returns:
    bool: True if the images match, False otherwise.
    """

    mm = bva.MiuraMatch(ch=80, cw=90)
    fit = mm.create_templates(fi, enroll=True)
    fir = mm.create_templates(fr, enroll=False)
    score = mm.score(fit, fir)

    return score

def prepro(input_image):
    pp = bvp.Preprocessor(bvp.NoCrop(), bvp.FixedMask(top=0, bottom=0, left=0, right=0), bvp.HuangNormalization(padding_width=100, padding_constant=100), bvp.HistogramEqualization())
    return pp(input_image)

def extract(input_image):
    mc = bve.MaximumCurvature(sigma=2)
    return mc(input_image)


def vr_prep_input(input_image_path):
    return extract(prepro(bi.to_bob(imageio.imread(input_image_path, mode='F').astype(np.float32))))

if __name__ == "__main__":
    input_image_pathX = "input/genuine/1393-PLUS-FV3-Laser_PALMAR_060_01_07_01.png"
    input_image_path = "input/genuine/1407-PLUS-FV3-Laser_PALMAR_060_01_09_05.png"
    reference_image_path = "input/genuine/1406-PLUS-FV3-Laser_PALMAR_060_01_09_04.png"

    match_result = vein_recognition(input_image_path, reference_image_path)
    print(f"Match result: {match_result}")