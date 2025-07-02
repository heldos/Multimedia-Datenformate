import os
import pandas as pd
from glob import glob

def analyze_quality_metrics(directory: str, output_csv: str):
    """
    Parse all CSV files in a directory and generate a summary CSV
    with compression quality metrics per format.
    """
    all_data = []

    # Collect all CSV files in the directory
    csv_files = glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    # Read and store each file's data
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # Combine all into one big DataFrame
    full_df = pd.concat(all_data, ignore_index=True)

    if "format" not in full_df.columns:
        print("Missing 'format' column in CSV files.")
        return

    # Group by format and summarize
    grouped = full_df.groupby("format")
    summary_list = []

    for fmt, group in grouped:
        summary = group.describe().loc[["mean", "min", "max"]].T
        summary = summary.reset_index().rename(columns={"index": "metric"})
        summary.insert(0, "format", fmt)
        summary_list.append(summary)

    # Concatenate all summaries
    final_summary = pd.concat(summary_list, ignore_index=True)

    # Write to CSV
    final_summary.to_csv(output_csv, index=False)
    print(f"Summary written to {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze image quality metrics from CSV files.")
    parser.add_argument("directory", help="Directory containing CSV metric files")
    parser.add_argument("output_csv", help="Output CSV file for summary")

    args = parser.parse_args()
    analyze_quality_metrics(args.directory, args.output_csv)
