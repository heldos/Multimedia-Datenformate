import csv
import os
import time
import argparse
from comp import compress_image
from qm import calc_qm
from qmanalytics import analyze_quality_metrics
from recog import vein_recog
from recoganalytics import calculate_analytics
from baseline import vein_recog_baseline  # Import the baseline function
from veinrecogutil import compare_finger_veins
from veinrecogutilv2 import find_vein_matches
from vein_mine import match_roi
from booob import vein_recognition, vr_prep_input
from performance_metrics import compute_metrics_from_csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def compress_file(input_path, output_path, size):
    fn = os.path.basename(input_path).split(".")[0]
    compress_image(input_path, os.path.join(output_path, fn), size)

def process_qm_file(input_dir, output_dir, size, filename):
    calc_qm(
        os.path.join(input_dir, filename),
        os.path.join(output_dir, f"comp/{size}"),
        os.path.join(output_dir, f"qm/{size}", f"{filename.split('.')[0]}.csv"),
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
        default=[4000, 8000, 16000, 32000],
        help="List of target sizes for compression."
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="Flag to perform image compression."
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
        "--recoga",
        action="store_true",
        help="Flag to perform recog analysation."
    )
    args = parser.parse_args()

    input_dir = "input/genuine/"
    output_dir = "output/"

    #img1 = os.path.join(input_dir, "1376-PLUS-FV3-Laser_PALMAR_059_01_09_04.png")
    #img2 = os.path.join(input_dir, "1377-PLUS-FV3-Laser_PALMAR_059_01_09_05.png")

    #result12 = compare_finger_veins(img1, img2)
    #print(f"Comparison result between {img1} and {img2}: {result12}")

    if args.comp:
        print("Compressing images...")
        for size in args.ts:
            os.makedirs(os.path.join(output_dir, f"comp/{size}"), exist_ok=True)
            with ThreadPoolExecutor() as executor:
                input_files = [
                    os.path.join(input_dir, filename)
                    for filename in os.listdir(input_dir)
                    if filename.endswith(".jpg") or filename.endswith(".png")
                ]
                output_path = os.path.join(output_dir, f"comp/{size}")
                executor.map(lambda f: compress_file(f, output_path, size), input_files)

    if args.qm:
        print("Running QM on images...")
        for size in args.ts:
            qm_output_dir = os.path.join(output_dir, f"qm/{size}")
            os.makedirs(qm_output_dir, exist_ok=True)
            input_files = [
                filename for filename in os.listdir(input_dir)
                if filename.endswith(".jpg") or filename.endswith(".png")
            ]
            print(f"Input files: {input_files}")

            with ProcessPoolExecutor() as executor:
                executor.map(
                    process_qm_file_wrapper,
                    [(input_dir, output_dir, size, filename) for filename in input_files]
                )

    if args.qma:
        print("Running QMA on images...")
        qma_output_dir = os.path.join(output_dir, f"qma/")
        os.makedirs(qma_output_dir, exist_ok=True)
        for size in args.ts:
            analyze_quality_metrics(
                os.path.join(output_dir, f"qm/{size}"),
                os.path.join(qma_output_dir, f"{size}.csv")
            )

    if args.recog:
        print("Running recognition on images...")
        for size in args.ts:
            comp_dir = os.path.join(output_dir, f"comp/{size}")
            output_path = os.path.join(output_dir, f"recog/{size}")
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

        for filename in input_files:
            start_time_file = time.time()
            first_number = str(filename.split('-')[0])
            ide = get_scores_exsist(first_number, os.path.join(output_dir, "baseline.csv"))
            if ide:
                print(f"ID {first_number} already exists in baseline.csv, skipping...")
                continue
            count += 1
            count_genuine = 0
            count_fake = 0
            scores_genuine = []
            scores_fake = []
            user_id_1, finger_id_1 = filename.split("_")[3:5]
            fi = vr_prep_input(os.path.join(input_dir, filename))
            for filename2 in input_files:
                if filename == filename2:
                    continue
                user_id_2, finger_id_2 = filename2.split("_")[3:5]

                is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                amount_of_images = 10
                if is_match:
                    count_genuine += 1
                else:
                    if count_fake == count_genuine and not count_genuine >= amount_of_images:
                        continue
                    if count_genuine >= amount_of_images and count_fake == count_genuine:
                        print(f"Time taken for file {count}: {time.time() - start_time_file:.2f} seconds")
                        if scores_genuine and scores_fake:
                            write_scores(first_number, scores_genuine, scores_fake, os.path.join(output_dir, "baseline.csv"))
                            print("-" * 50)
                            print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                            #print(f"sorted_scores_genuine: {sorted(scores_genuine)}")
                            print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                            #print(f"sorted_scores_fake: {sorted(scores_fake)}")
                            best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                            print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                        break
                    count_fake += 1
                #start_time_vein_recog = time.time()
                fr = vr_prep_input(os.path.join(input_dir, filename2))
                score = vein_recognition(
                    fi,
                    fr,
                )
                if is_match:
                    scores_genuine.append(score)
                else:
                    scores_fake.append(score)
                #time_txt = f"{time.time() - start_time_vein_recog:.2f} seconds"
                #print(time_txt)

    if args.performancebaseline:
        compute_metrics_from_csv(os.path.join(output_dir, "baseline.csv"))

    if args.recoga:
        print("Running recognition analytics on images...")
        recog_output_dir = os.path.join(output_dir, "recog")
        recoga_output_dir = os.path.join(output_dir, "recoga")
        os.makedirs(recoga_output_dir, exist_ok=True)
        for size in args.ts:
            input_path = os.path.join(recog_output_dir, f"{size}")
            output_path = os.path.join(output_dir, f"recoga/{size}.csv")
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