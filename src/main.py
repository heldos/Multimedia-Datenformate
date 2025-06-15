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
from veinrecogutilv2fuck import find_vein_matches
from booob import vein_recognition, vr_prep_input
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
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        count = 0
        count_genuine = 0
        count_fake = 0
        input_files = [
            filename for filename in os.listdir(input_dir)
            if filename.endswith(".png")  # Baseline expects .png files
        ]

        for filename in input_files:
            start_time_file = time.time()
            first_number = int(filename.split('-')[0])
            threshold, exsists = get_best_threshold(os.path.join("output/best_threshold.csv"), first_number)
            if exsists:
                continue
            count += 1
            count_genuine = 0
            count_fake = 0
            scores_genuine = []
            scores_fake = []
            print("-" * 50)
            print("-" * 50)
            print("-" * 50)
            print(f"Processing file {count}/{len(input_files)}: {filename}")
            print("-" * 50)
            user_id_1, finger_id_1 = filename.split("_")[3:5]  # Extract user ID and finger ID
            fi = vr_prep_input(os.path.join(input_dir, filename))
            for filename2 in input_files:
                if filename == filename2:
                    continue
                user_id_2, finger_id_2 = filename2.split("_")[3:5]  # Extract user ID and finger ID

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
                            print("-" * 50)
                            print(f"min_score_genuine: {min(scores_genuine)} | max_score_genuine: {max(scores_genuine)} | avg_score_genuine: {sum(scores_genuine)/len(scores_genuine) if scores_genuine else 0}")
                            print(f"sorted_scores_genuine: {sorted(scores_genuine)}")
                            print(f"min_score_fake: {min(scores_fake)} | max_score_fake: {max(scores_fake)} | avg_score_fake: {sum(scores_fake)/len(scores_fake) if scores_fake else 0}")
                            print(f"sorted_scores_fake: {sorted(scores_fake)}")
                            best_threshold, min_overlap = find_best_threshold(scores_genuine, scores_fake)
                            if min_overlap >= 4:
                                print("❌❌❌")
                                print(f"Threshold {best_threshold} is too high, skipping...")
                                print("❌❌❌")
                                break
                            write_threshold(os.path.join("output/best_threshold.csv"), first_number, best_threshold)
                            print(f"Best threshold: {best_threshold} | Minimal overlap: {min_overlap}")
                            print("-" * 50)
                            print(f"True Positives: {tp} | False Positives: {fp}\nFalse Negatives: {fn} | True Negatives: {tn}")
                        break
                    count_fake += 1

                print(f"{count} {count_genuine} {count_fake} | Comparing {user_id_1} - {finger_id_1} with {user_id_2} - {finger_id_2} | Match: {is_match}")
                start_time_vein_recog = time.time()
                fr = vr_prep_input(os.path.join(input_dir, filename2))
                score = vein_recognition(
                    fi,
                    fr,
                )
                score = round(score, 4)
                
                print(f"Score: {score}")

                threshold, exsists = get_best_threshold(os.path.join("output/best_threshold.csv"), first_number)
                if score >= threshold:
                    result = True
                else:
                    result = False

                if is_match:
                    scores_genuine.append(score)
                else:
                    scores_fake.append(score)

                time_txt = f" | in {time.time() - start_time_vein_recog:.2f} seconds"
                if result and is_match:
                    tp += 1
                    print(f"✅{time_txt}")
                elif result and not is_match:
                    fp += 1
                    print(f"❌{time_txt}")
                elif not result and is_match:
                    fn += 1
                    print(f"❌{time_txt}")
                elif not result and not is_match:
                    tn += 1
                    print(f"✅{time_txt}")
                
                print("-" * 50)

        print(f"True Positives: {tp} | False Positives: {fp}\nFalse Negatives: {fn} | True Negatives: {tn}")
    
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

def write_threshold(filename, ID, new_threshold):
    """
    Write or update a CSV with ID and best_threshold.
    If the ID already exists, compute the average of the old and new threshold.
    Otherwise, add a new row with the new threshold.
    """
    data = {}
    # Check if the file exists and read its content first
    if os.path.exists(filename):
        with open(filename, newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if len(row) == 2:
                    try:
                        row_id = int(row[0])  # or str, depending on your IDs
                        row_threshold = float(row[1])  # or int
                        data[row_id] = row_threshold
                    except ValueError:
                        continue

    # Update or insert
    if ID in data:
        data[ID] = (data[ID] + new_threshold) / 2
    else:
        data[ID] = new_threshold

    # Write back to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for row_id, row_threshold in data.items():
            writer.writerow([row_id, row_threshold])

def get_best_threshold(filename, ID):
    """
    Retrieve the best threshold for a given ID from CSV.
    """
    if not os.path.exists(filename):
        return 0.15, False

    with open(filename, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) == 2:
                try:
                    row_id = int(row[0])  # or str
                    row_threshold = float(row[1])  # or int
                    if row_id == ID:
                        return row_threshold, True
                except ValueError:
                    continue
    return 0.15, False
                

if __name__ == "__main__":
    main()