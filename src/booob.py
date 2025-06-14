# https://www.youtube.com/watch?v=wOdfNwD9cEA&t=13s

import cv2
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import bob.bio.vein.config.repeated_line_tracking as rlt_config
import bob.bio.vein.preprocessor as bvp
import bob.bio.vein.extractor as bve
import bob.bio.vein.algorithm as bva
import bob.io.image as bi

def vein_recognition(input_image_path, reference_image_path, threshold=0.15):
    """
    Perform finger vein recognition using the Bob library with the repeated-line tracking pipeline.
    
    Parameters:
    input_image_path (str): Path to the input finger vein image.
    reference_image_path (str): Path to the reference finger vein image.
    
    Returns:
    bool: True if the images match, False otherwise.
    """
    # Load the images
    input_image = imageio.imread(input_image_path, mode='F').astype(np.float32)
    reference_image = imageio.imread(reference_image_path, mode='F').astype(np.float32)

    ii = bi.to_bob(input_image)
    rr = bi.to_bob(reference_image)

   
    pp = bvp.Preprocessor(bvp.NoCrop(), bvp.NoMask(), bvp.HuangNormalization(padding_width=5, padding_constant=51), bvp.HistogramEqualization())
    ii = pp(input_image)
    rr = pp(reference_image)
    

    mc = bve.MaximumCurvature(sigma=3)
    fi = mc(ii)
    fr = mc(rr)

    mm = bva.MiuraMatch(ch=80, cw=90)
    fit = mm.create_templates(fi, enroll=True)
    fir = mm.create_templates(fr, enroll=False)
    score = mm.score(fit, fir)

    #print(f"Score: {score}")

    return score >= threshold

if __name__ == "__main__":
    input_image_pathX = "input/genuine/1393-PLUS-FV3-Laser_PALMAR_060_01_07_01.png"
    input_image_path = "input/genuine/1407-PLUS-FV3-Laser_PALMAR_060_01_09_05.png"
    reference_image_path = "input/genuine/1406-PLUS-FV3-Laser_PALMAR_060_01_09_04.png"
    
    match_result = vein_recognition(input_image_path, reference_image_path)
    print(f"Match result: {match_result}")