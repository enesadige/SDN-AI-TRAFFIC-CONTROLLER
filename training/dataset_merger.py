import os
import glob
import pandas as pd

# ============================================================================
# SETTINGS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
OUTPUT_FILENAME = "dataset_all_merged.csv"

def merge_datasets():
    print(f" Working Directory: {DATA_DIR}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f" Error: Data directory not found at {DATA_DIR}")
        return

    # Find all CSV files in data/
    search_pattern = os.path.join(DATA_DIR, "*.csv")
    csv_files = sorted(glob.glob(search_pattern))
    
    # Exclude the output file itself if it exists, to avoid infinite loops or recursion
    csv_files = [f for f in csv_files if OUTPUT_FILENAME not in f]
    
    print(f" Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")

    if len(csv_files) == 0:
        print(" No CSV files found to merge!")
        return

    # Read and merge
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f" Failed to read {f}: {e}")

    if not dfs:
        print(" No valid dataframes to merge.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    
    # Save output
    output_path = os.path.join(DATA_DIR, OUTPUT_FILENAME)
    merged.to_csv(output_path, index=False)

    print(" Merge Completed!")
    print(f" Output saved to: {output_path}")
    print(f" Total rows: {len(merged)}")

if __name__ == "__main__":
    merge_datasets()
