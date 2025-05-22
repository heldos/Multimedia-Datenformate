import os
import argparse
from comp import compress_image
from qm import calc_qm
from qmanalytics import analyze_quality_metrics
from recog import vein_recog
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
    args = parser.parse_args()

    input_dir = "input/genuine/"
    output_dir = "output/"

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
                

if __name__ == "__main__":
    main()