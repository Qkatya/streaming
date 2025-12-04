#!/usr/bin/env python3
"""Diagnose why peaks weren't added."""

import pandas as pd
import pickle
from pathlib import Path

# Load manual blink labels
labels_df = pd.read_csv('manual_blink_labels.csv')
blink_labels = labels_df[labels_df['label'] == 'blink']

print(f"Total blink labels: {len(blink_labels)}")
print(f"\nFirst 5 blink labels:")
print(blink_labels[['run_path', 'tar_id', 'side', 'peak_frame_25fps']].head())

# Load the split to see what files were sampled
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'
NUM_RANDOM_SAMPLES = 1000
RANDOM_SEED = 42

with open(SPLIT_DF_PATH, 'rb') as f:
    split_df = pickle.load(f)

# Sample the same way as the script
sampled_df = split_df.sample(n=NUM_RANDOM_SAMPLES, random_state=RANDOM_SEED)

# Create set of sampled files
sampled_files = set()
for idx, row in sampled_df.iterrows():
    sampled_files.add((row['run_path'], row['tar_id'], row.get('side', None)))

print(f"\nTotal sampled files: {len(sampled_files)}")

# Check which blink labels are in the sampled files
matched = 0
unmatched = 0
unmatched_files = []

for idx, row in blink_labels.iterrows():
    key = (row['run_path'], row['tar_id'], row['side'])
    if key in sampled_files:
        matched += 1
    else:
        unmatched += 1
        unmatched_files.append(key)

print(f"\nBlink labels in sampled files: {matched}")
print(f"Blink labels NOT in sampled files: {unmatched}")

if unmatched > 0:
    print(f"\nFirst 5 unmatched files:")
    for i, file in enumerate(unmatched_files[:5]):
        print(f"  {i+1}. {file}")

# Load existing GT peaks
if Path('final_gt_peaks.pkl').exists():
    with open('final_gt_peaks.pkl', 'rb') as f:
        gt_peaks = pickle.load(f)
    print(f"\nExisting GT peaks: {len(gt_peaks)}")
else:
    print("\nNo existing GT peaks file found")


