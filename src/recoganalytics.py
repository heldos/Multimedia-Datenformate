import os
import pandas as pd

def calculate_analytics(input_dir, output_path):
    """
    Calculate mean, min, and max for numeric values in CSV files for each image format.

    Args:
        input_dir (str): Path to the directory containing input CSV files.
        output_path (str): Path to the output CSV file.
    """
    # Initialize an empty DataFrame to store aggregated data
    aggregated_data = pd.DataFrame()

    # Iterate through all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Append the data to the aggregated DataFrame
            aggregated_data = pd.concat([aggregated_data, df], ignore_index=True)

    # Group by 'format' and calculate mean, min, and max for numeric columns
    analytics = aggregated_data.groupby('format').agg(
        mean_num_keypoints_original=('num_keypoints_original', 'mean'),
        min_num_keypoints_original=('num_keypoints_original', 'min'),
        max_num_keypoints_original=('num_keypoints_original', 'max'),
        mean_num_keypoints_compressed=('num_keypoints_compressed', 'mean'),
        min_num_keypoints_compressed=('num_keypoints_compressed', 'min'),
        max_num_keypoints_compressed=('num_keypoints_compressed', 'max'),
        mean_num_matches=('num_matches', 'mean'),
        min_num_matches=('num_matches', 'min'),
        max_num_matches=('num_matches', 'max'),
        mean_match_score=('match_score', 'mean'),
        min_match_score=('match_score', 'min'),
        max_match_score=('match_score', 'max')
    ).reset_index()

    # Write the analytics to the output CSV file
    analytics.to_csv(output_path, index=False)
    print(f"Analytics saved to {output_path}")

# Example usage:
# calculate_analytics('/path/to/input/directory', '/path/to/output/analytics.csv')