import csv
import os
import time
import argparse
from comp import compress_image
from decomp import decompress_and_save
from qm import calc_qm
from qmanalytics import analyze_quality_metrics
from recog import vein_recog
from recoganalytics import calculate_analytics
from baseline import vein_recog_baseline  # Import the baseline function
from veinrecogutil import compare_finger_veins
from veinrecogutilv2 import find_vein_matches
from vein_mine import match_roi
from booob import vein_recognition, vr_prep_input
from fvmv3 import FingerVeinMatcher
import fvmv4
from performance_metrics import compute_metrics_from_csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def compress_file(input_path, output_path, size):
    fn = os.path.basename(input_path).split(".")[0]
    compress_image(input_path, os.path.join(output_path, fn), size)

def decompress_file(input_path, output_path, size):
    fn = os.path.basename(input_path).split(".")[0]
    decompress_and_save(input_path, os.path.join(output_path, fn))

def process_qm_file(input_dir, output_dir, size, filename):
    calc_qm(
        os.path.join(input_dir, filename),
        os.path.join(output_dir, "comp", str(size)),
        os.path.join(output_dir, "qm", str(size), f"{filename}.csv"),
        filename.split(".")[0]
    )

def process_qm_file_wrapper(args):
    input_dir, output_dir, size, filename = args
    process_qm_file(input_dir, output_dir, size, filename)

def main():
    parser = argparse.ArgumentParser(description="Process images.")
    parser.add_argument(
        "--ts",
        type=int,
        nargs="+",
        default=[2000, 4000, 8000, 16000, 32000],
        help="List of target sizes for compression."
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="Flag to perform image compression."
    )
    parser.add_argument(
        "--decomp",
        action="store_true",
        help="Flag to perform image decompression."
    )
    parser.add_argument(
        "--qm",
        action="store_true",
        help="Flag to perform QM calculation."
    )
    parser.add_argument(
        "--qma",
        action="store_true",
        help="Flag to perform QM analysation."
    )
    parser.add_argument(
        "--recog",
        action="store_true",
        help="Flag to perform bob vein recognition."
    )
    parser.add_argument(
        "--recogbaseline",
        action="store_true",
        help="Flag to perform baseline vein recognition."
    )
    parser.add_argument(
        "--performancebaseline",
        action="store_true",
        help="Flag to perform performance metric calculation on baseline."
    )
    parser.add_argument(
        "--recogjpeg",
        action="store_true",
        help="Flag to perform jpeg vein recognition."
    )
    parser.add_argument(
        "--performancejpeg",
        action="store_true",
        help="Flag to perform performance metric calculation on jpeg."
    )
    parser.add_argument(
        "--recogjpeg2000",
        action="store_true",
        help="Flag to perform jpeg2000 vein recognition."
    )
    parser.add_argument(
        "--performancejpeg2000",
        action="store_true",
        help="Flag to perform performance metric calculation on jpeg2000."
    )
    parser.add_argument(
        "--recogjpegxr",
        action="store_true",
        help="Flag to perform jpeg xr vein recognition."
    )
    parser.add_argument(
        "--performancejpegxr",
        action="store_true",
        help="Flag to perform performance metric calculation on jpeg xr."
    )
    parser.add_argument(
        "--recogjpegxl",
        action="store_true",
        help="Flag to perform jpeg xl vein recognition."
    )
    parser.add_argument(
        "--performancejpegxl",
        action="store_true",
        help="Flag to perform performance metric calculation on jpeg xl."
    )
    parser.add_argument(
        "--recoga",
        action="store_true",
        help="Flag to perform recog analysation."
    )
    args = parser.parse_args()

    input_dir = os.path.join("input", "genuine")
    output_dir = os.path.join("output")

    #img1 = os.path.join(input_dir, "1376-PLUS-FV3-Laser_PALMAR_059_01_09_04.png")
    #img2 = os.path.join(input_dir, "1377-PLUS-FV3-Laser_PALMAR_059_01_09_05.png")

    #result12 = compare_finger_veins(img1, img2)
    #print(f"Comparison result between {img1} and {img2}: {result12}")

    if args.comp:
        print("Compressing images...")
        for size in args.ts:
            os.makedirs(os.path.join(output_dir, "comp", str(size)), exist_ok=True)
            with ThreadPoolExecutor() as executor:
                input_files = [
                    os.path.join(input_dir, filename)
                    for filename in os.listdir(input_dir)
                    if filename.endswith(".jpg") or filename.endswith(".png")
                ]
                output_path = os.path.join(output_dir, "comp", str(size))
                executor.map(lambda f: compress_file(f, output_path, size), input_files)

    if args.decomp:
        print("Deompressing images...")
        for size in args.ts:
            os.makedirs(os.path.join(output_dir, "decomp", str(size)), exist_ok=True)
            with ThreadPoolExecutor() as executor:
                input_files = [
                    os.path.join(output_dir, "comp", str(size), filename)
                    for filename in os.listdir(os.path.join(output_dir, "comp", str(size)))
                    if filename.endswith(".jpg") or filename.endswith(".jp2") or filename.endswith(".jxr") or filename.endswith(".jxl")
                ]
                output_path = os.path.join(output_dir, "decomp", str(size))
                executor.map(lambda f: decompress_file(f, output_path, size), input_files)

    if args.qm:
        print("Running QM on images...")
        for size in args.ts:
            qm_output_dir = os.path.join(output_dir, "qm", str(size))
            os.makedirs(qm_output_dir, exist_ok=True)
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".png")
            ]

            with ProcessPoolExecutor() as executor:
                executor.map(
                    process_qm_file_wrapper,
                    [(input_dir, output_dir, size, filename) for filename in input_files]
                )

    if args.qma:
        print("Running QMA on images...")
        qma_output_dir = os.path.join(output_dir, "qma")
        os.makedirs(qma_output_dir, exist_ok=True)
        for size in args.ts:
            analyze_quality_metrics(
                os.path.join(output_dir, "qm", str(size)),
                os.path.join(qma_output_dir, f"{size}.csv")
            )

    if args.recog:
        print("Running recognition on images...")
        for size in args.ts:
            comp_dir = os.path.join(output_dir, "comp", str(size))
            output_path = os.path.join(output_dir, "recog", str(size))
            os.makedirs(output_path, exist_ok=True)
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".jpg") or filename.endswith(".png")
            ]
            for filename in input_files:
                vein_recog(
                    os.path.join(input_dir, filename),
                    comp_dir,
                    os.path.join(output_path, f"{filename.split('.')[0]}.csv"),
                    filename.split(".")[0]
                )

    if args.recogbaseline:
        print("Running baseline recognition on images...")
        count = 0
        input_files = [
            filename for filename in os.listdir(input_dir)
            if filename.endswith(".png")
        ]
        matcher = FingerVeinMatcher()

        for filename in input_files:
            start_time_file = time.time()
            first_number = str(filename.split('-')[0])
            ide = get_scores_exsist(first_number, os.path.join(output_dir, "baseline.csv"))
            if ide:
                print(f"ID {first_number} already exists in baseline.csv, skipping...")
                continue
            count += 1
            scores_genuine = []
            scores_fake = []
            user_id_1, finger_id_1 = filename.split("_")[3:5]
            for filename2 in input_files:
                if filename == filename2:
                    continue
                user_id_2, finger_id_2 = filename2.split("_")[3:5]

                is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                amount_of_images = 20
                if not is_match and len(scores_fake) == len(scores_genuine) and not len(scores_genuine) >= amount_of_images:
                    continue
                if len(scores_fake) == len(scores_genuine) and len(scores_genuine) >= amount_of_images:
                    print("-" * 50)
                    print(f"Time taken for file {count}: {time.time() - start_time_file:.2f} seconds")
                    if scores_genuine and scores_fake:
                        write_scores(first_number, scores_genuine, scores_fake, os.path.join(output_dir, "baseline.csv"))
                        print("-" * 50)
                        print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                        print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                        best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                        print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                    break
                score = fvmv4.match_finger_veins(os.path.join(input_dir, filename), os.path.join(input_dir, filename2), method='template')
                if is_match and score > 0.0:
                    scores_genuine.append(score)
                elif not is_match and score > 0.0:
                    scores_fake.append(score)

    if args.performancebaseline:
        compute_metrics_from_csv(os.path.join(output_dir, "baseline.csv"))


    if args.recogjpeg:
        print("Running jpeg recognition on images...")
        for size in args.ts:
            count = 0
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".png")
            ]
            comp_files = [
                filename2 for filename2 in os.listdir(os.path.join(output_dir, "comp", str(size)))
                if filename2.endswith("jpg.png")
            ]
            matcher = FingerVeinMatcher()

            for filename in input_files:
                start_time_file = time.time()
                first_number = str(filename.split('-')[0])
                ide = get_scores_exsist(first_number, os.path.join(output_dir, "recog", str(size), "jpeg.csv"))
                if ide:
                    print(f"ID {first_number} already exists in jpeg.csv, skipping...")
                    continue
                count += 1
                scores_genuine = []
                scores_fake = []
                user_id_1, finger_id_1 = filename.split("_")[3:5]
                for filename2 in comp_files:
                    #if filename == filename2:
                    #    continue
                    user_id_2, finger_id_2 = filename2.split("_")[3:5]

                    is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                    amount_of_images = 20
                    if not is_match and len(scores_fake) == len(scores_genuine) and not len(scores_genuine) >= amount_of_images:
                        continue
                    if len(scores_fake) == len(scores_genuine) and len(scores_genuine) >= amount_of_images:
                        print("-" * 50)
                        print(f"Time taken for file {count}: {time.time() - start_time_file:.2f} seconds")
                        if scores_genuine and scores_fake:
                            write_scores(first_number, scores_genuine, scores_fake, os.path.join(output_dir, "recog", str(size), "jpeg.csv"))
                            print("-" * 50)
                            print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                            print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                            best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                            print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                        break
                    score = fvmv4.match_finger_veins(os.path.join(input_dir, filename), os.path.join(output_dir, "comp", str(size), filename2), method='template')
                    if is_match and score > 0.0:
                        scores_genuine.append(score)
                    elif not is_match and score > 0.0:
                        scores_fake.append(score)

    if args.performancejpeg:
        for size in args.ts:
            print(f"Calculating performance metrics for jpeg at size {size}...")
            compute_metrics_from_csv(os.path.join(output_dir, "recog", str(size), "jpeg.csv"))


    if args.recogjpeg2000:
        print("Running jpeg 2000 recognition on images...")
        for size in args.ts:
            count = 0
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".png")
            ]
            comp_files = [
                filename2 for filename2 in os.listdir(os.path.join(output_dir, "comp", str(size)))
                if filename2.endswith("jp2.png")
            ]
            matcher = FingerVeinMatcher()

            for filename in input_files:
                start_time_file = time.time()
                first_number = str(filename.split('-')[0])
                ide = get_scores_exsist(first_number, os.path.join(output_dir, "recog", str(size), "jp2.csv"))
                if ide:
                    print(f"ID {first_number} already exists in jp2.csv, skipping...")
                    continue
                count += 1
                scores_genuine = []
                scores_fake = []
                user_id_1, finger_id_1 = filename.split("_")[3:5]
                for filename2 in comp_files:
                    #if filename == filename2:
                    #    continue
                    user_id_2, finger_id_2 = filename2.split("_")[3:5]

                    is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                    amount_of_images = 20
                    if not is_match and len(scores_fake) == len(scores_genuine) and not len(scores_genuine) >= amount_of_images:
                        continue
                    if len(scores_fake) == len(scores_genuine) and len(scores_genuine) >= amount_of_images:
                        print("-" * 50)
                        print(f"Time taken for file {count}: {time.time() - start_time_file:.2f} seconds")
                        if scores_genuine and scores_fake:
                            write_scores(first_number, scores_genuine, scores_fake, os.path.join(output_dir, "recog", str(size), "jp2.csv"))
                            print("-" * 50)
                            print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                            print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                            best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                            print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                        break
                    score = fvmv4.match_finger_veins(os.path.join(input_dir, filename), os.path.join(output_dir, "comp", str(size), filename2), method='template')
                    if is_match and score > 0.0:
                        scores_genuine.append(score)
                    elif not is_match and score > 0.0:
                        scores_fake.append(score)

    if args.performancejpeg2000:
        for size in args.ts:
            print(f"Calculating performance metrics for jpeg 2000 at size {size}...")
            compute_metrics_from_csv(os.path.join(output_dir, "recog", str(size), "jp2.csv"))

    if args.recogjpegxr:
        print("Running jpeg xr recognition on images...")
        for size in args.ts:
            count = 0
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".png")
            ]
            comp_files = [
                filename2 for filename2 in os.listdir(os.path.join(output_dir, "comp", str(size)))
                if filename2.endswith("jpxr.png")
            ]
            matcher = FingerVeinMatcher()

            for filename in input_files:
                start_time_file = time.time()
                first_number = str(filename.split('-')[0])
                ide = get_scores_exsist(first_number, os.path.join(output_dir, "recog", str(size), "jpxr.csv"))
                if ide:
                    print(f"ID {first_number} already exists in jpxr.csv, skipping...")
                    continue
                count += 1
                scores_genuine = []
                scores_fake = []
                user_id_1, finger_id_1 = filename.split("_")[3:5]
                for filename2 in comp_files:
                    #if filename == filename2:
                    #    continue
                    user_id_2, finger_id_2 = filename2.split("_")[3:5]

                    is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                    amount_of_images = 20
                    if not is_match and len(scores_fake) == len(scores_genuine) and not len(scores_genuine) >= amount_of_images:
                        continue
                    if len(scores_fake) == len(scores_genuine) and len(scores_genuine) >= amount_of_images:
                        print("-" * 50)
                        print(f"Time taken for file {count}: {time.time() - start_time_file:.2f} seconds")
                        if scores_genuine and scores_fake:
                            write_scores(first_number, scores_genuine, scores_fake, os.path.join(output_dir, "recog", str(size), "jpxr.csv"))
                            print("-" * 50)
                            print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                            print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                            best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                            print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                        break
                    score = fvmv4.match_finger_veins(os.path.join(input_dir, filename), os.path.join(output_dir, "comp", str(size), filename2), method='template')
                    if is_match and score > 0.0:
                        scores_genuine.append(score)
                    elif not is_match and score > 0.0:
                        scores_fake.append(score)

    if args.performancejpegxr:
        for size in args.ts:
            print(f"Calculating performance metrics for jpeg xr at size {size}...")
            compute_metrics_from_csv(os.path.join(output_dir, "recog", str(size), "jpxr.csv"))

    if args.recogjpegxl:
        print("Running jpeg xl recognition on images...")
        for size in args.ts:
            count = 0
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".png")
            ]
            comp_files = [
                filename2 for filename2 in os.listdir(os.path.join(output_dir, "comp", str(size)))
                if filename2.endswith("jpxl.png")
            ]
            matcher = FingerVeinMatcher()

            for filename in input_files:
                start_time_file = time.time()
                first_number = str(filename.split('-')[0])
                ide = get_scores_exsist(first_number, os.path.join(output_dir, "recog", str(size), "jpxl.csv"))
                if ide:
                    print(f"ID {first_number} already exists in jpxl.csv, skipping...")
                    continue
                count += 1
                scores_genuine = []
                scores_fake = []
                user_id_1, finger_id_1 = filename.split("_")[3:5]
                for filename2 in comp_files:
                    #if filename == filename2:
                    #    continue
                    user_id_2, finger_id_2 = filename2.split("_")[3:5]

                    is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                    amount_of_images = 20
                    if not is_match and len(scores_fake) == len(scores_genuine) and not len(scores_genuine) >= amount_of_images:
                        continue
                    if len(scores_fake) == len(scores_genuine) and len(scores_genuine) >= amount_of_images:
                        print("-" * 50)
                        print(f"Time taken for file {count}: {time.time() - start_time_file:.2f} seconds")
                        if scores_genuine and scores_fake:
                            write_scores(first_number, scores_genuine, scores_fake, os.path.join(output_dir, "recog", str(size), "jpxl.csv"))
                            print("-" * 50)
                            print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                            print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                            best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                            print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                        break
                    score = fvmv4.match_finger_veins(os.path.join(input_dir, filename), os.path.join(output_dir, "comp", str(size), filename2), method='template')
                    if is_match and score > 0.0:
                        scores_genuine.append(score)
                    elif not is_match and score > 0.0:
                        scores_fake.append(score)

    if args.performancejpegxr:
        for size in args.ts:
            print(f"Calculating performance metrics for jpeg xl at size {size}...")
            compute_metrics_from_csv(os.path.join(output_dir, "recog", str(size), "jpxl.csv"))

    if args.recoga:
        print("Running recognition analytics on images...")
        recog_output_dir = os.path.join(output_dir, "recog")
        recoga_output_dir = os.path.join(output_dir, "recoga")
        os.makedirs(recoga_output_dir, exist_ok=True)
        for size in args.ts:
            input_path = os.path.join(recog_output_dir, str(size))
            output_path = os.path.join(output_dir, "recoga", f"{size}.csv")
            calculate_analytics(input_path, output_path)

def find_best_threshold(genuine, fake):
    """
    Find the threshold that minimizes overlap between two score distributions.

    Args:
        genuine (list of float): List of scores for genuine samples.
        fake (list of float): List of scores for fake samples.

    Prints:
        The best threshold and the minimum overlap at that threshold.
    """
    # Combine all scores to search potential thresholds
    all_scores = sorted(set(genuine + fake))

    min_overlap = float('inf')
    best_threshold = None

    for threshold in all_scores:
        false_neg = sum(s < threshold for s in genuine)
        false_pos = sum(s >= threshold for s in fake)
        overlap = false_neg + false_pos

        if overlap < min_overlap:
            min_overlap = overlap
            best_threshold = threshold

    return best_threshold, min_overlap

def write_scores(ID, scores_genuine, scores_fake, output_path):
    """
    Write the scores to a CSV file.

    Args:
        ID (str): Identifier for the scores.
        scores_genuine (list of float): List of genuine scores.
        scores_fake (list of float): List of fake scores.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, mode='a', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow([ID, scores_genuine, scores_fake])

def get_scores_exsist(ID, csv):
    """
    Get the scores for a given ID from a CSV file.

    Args:
        ID (str): Identifier for the scores.
        csv (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing lists of genuine and fake scores.
    """
    id_exsists = False
    file_exists = os.path.isfile(csv)
    if not file_exists:
        print(f"CSV file {csv} does not exist.")
        return id_exsists
    with open(csv, mode='r') as file:
        for row in file:
            row_ID = row.strip().split(',')[0]
            if int(row_ID) == int(ID):
                id_exsists = True
                break
    return id_exsists



if __name__ == "__main__":
    main()