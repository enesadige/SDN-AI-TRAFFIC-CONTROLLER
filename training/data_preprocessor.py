import pandas as pd
import re
import json
import os

# ============================================================================
# SETTINGS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

# Input file (Result of dataset_merger.py)
INPUT_FILE = os.path.join(DATA_DIR, "dataset_all_merged.csv")

# Output file (Cleaned data for model training)
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_cleaned_final.csv")

print(f" File reading: {INPUT_FILE}...")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f" File not found at {INPUT_FILE}!")
    print(" Please run 'dataset_merger.py' first.")
    exit()

original_count = len(df)
print(f" Initial Row Count: {original_count}")

# === 2. REMOVE DEAD FLOWS ===
# avg_path_load being 0 is NORMAL. We don't delete that.
# We only delete if flow_speed is suspiciously close to 0 (Ryu dead flows).
df = df[df["flow_speed_mbps"] > 0.0001]

deleted_count = original_count - len(df)
print(f" Deleted 'Dead Flow' Count: {deleted_count}")

# === 3. FIX PRECISION ===
float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    df[col] = df[col].round(6)

# === 4. FIX LIST STRINGS ===
list_cols = ["path_load_sorted", "path_capacity_sorted", "path_delay_sorted"]

def fix_list_precision(text):
    if isinstance(text, str):
        return re.sub(
            r"(\d+\.\d+)", 
            lambda m: f"{float(m.group(1)):.6f}", 
            text
        )
    return text

for col in list_cols:
    if col in df.columns:
        df[col] = df[col].apply(fix_list_precision)

# === 5. TARGET ANALYSIS ===
target_counts = df['target'].value_counts()

print("\n===  TARGET DISTRIBUTION (CLEANED) ===")
print(f" Normal (0)      : {target_counts.get(0, 0)} rows")
print(f" Congestion (1)  : {target_counts.get(1, 0)} rows")
print(f" Elephant (3)    : {target_counts.get(3, 0)} rows")
print(f"TOTAL             : {len(df)} rows")

# === 6. SAVE ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Clean CSV saved to: {OUTPUT_FILE}")
print(" Ready for training!")