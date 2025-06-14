import os
import argparse
from comp import compress_image
from qm import calc_qm
from qmanalytics import analyze_quality_metrics
from recog import vein_recog
from recoganalytics import calculate_analytics
from baseline import vein_recog_baseline  # Import the baseline function
from veinrecogutil import compare_finger_veins
from veinrecogutilv2fuck import find_vein_matches
from booob import vein_recognition
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
            count += 1
            count_genuine = 0
            count_fake = 0
            print("-" * 50)
            print(f"True Positives: {tp} | False Positives: {fp}\nFalse Negatives: {fn} | True Negatives: {tn}")
            print("-" * 50)
            print(f"Processing file {count}/{len(input_files)}: {filename}")
            print("-" * 50)
            user_id_1, finger_id_1 = filename.split("_")[3:5]  # Extract user ID and finger ID
            for filename2 in input_files:
                if filename == filename2:
                    continue
                user_id_2, finger_id_2 = filename2.split("_")[3:5]  # Extract user ID and finger ID

                is_match = user_id_1 == user_id_2 and finger_id_1 == finger_id_2
                if is_match:
                    count_genuine += 1
                else:
                    if count_fake == count_genuine and not count_genuine >= 20:
                        continue
                    if count_genuine >= 20 and count_fake == count_genuine:
                        break
                    count_fake += 1

                print(f"{count} {count_genuine} {count_fake} | Comparing {user_id_1} - {finger_id_1} with {user_id_2} - {finger_id_2} | Match: {is_match}")
                score = vein_recognition(
                    os.path.join(input_dir, filename),
                    os.path.join(input_dir, filename2),
                )
                print(f"Score: {score}")

                threshold = 0.072
                if score >= threshold:
                    result = True
                else:
                    result = False

                if result and is_match:
                    tp += 1
                    print("✅")
                elif result and not is_match:
                    fp += 1
                    print("❌")
                elif not result and is_match:
                    fn += 1
                    print("❌")
                elif not result and not is_match:
                    tn += 1
                    print("✅")
                
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
                

if __name__ == "__main__":
    main()