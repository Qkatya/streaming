#!/usr/bin/env python3
"""Check FPS consistency between CSV and PKL files."""

import pandas as pd
import pickle
import numpy as np

# Load manual blink labels CSV
df = pd.read_csv('manual_blink_labels.csv')
print(f"Total labels in CSV: {len(df)}")
print(f"Blink labels: {len(df[df['label'] == 'blink'])}")

# Check the relationship between the two frame columns
print("\nFrame column relationship:")
print(f"Mean ratio (30fps/25fps): {(df['peak_frame_30fps'] / df['peak_frame_25fps']).mean():.4f}")
print(f"Expected ratio: {30/25:.4f}")

# Load final GT peaks
with open('final_gt_peaks.pkl', 'rb') as f:
    gt_peaks = pickle.load(f)

print(f"\nGT peaks in PKL: {len(gt_peaks)}")
print(f"First 5 peaks: {gt_peaks[:5]}")
print(f"Last 5 peaks: {gt_peaks[-5:]}")

# Check if peaks match what we'd expect from CSV
blink_labels = df[df['label'] == 'blink']
print(f"\nFirst 5 blink labels from CSV:")
print(blink_labels[['peak_frame_25fps', 'peak_frame_30fps', 'run_path']].head())

# The total concatenated frames should be 158239 at 25fps
total_frames_25fps = 158239
total_frames_30fps = int(total_frames_25fps * 1.2)

print(f"\nTotal frames at 25fps: {total_frames_25fps}")
print(f"Total frames at 30fps: {total_frames_30fps}")
print(f"Duration at 25fps: {total_frames_25fps / 25:.2f} seconds")
print(f"Duration at 30fps: {total_frames_30fps / 30:.2f} seconds")

# Check max peak value
print(f"\nMax GT peak: {gt_peaks.max()}")
print(f"If peaks are at 25fps, max should be < {total_frames_25fps}")
print(f"If peaks are at 30fps, max should be < {total_frames_30fps}")

if gt_peaks.max() < total_frames_25fps:
    print("✓ Peaks appear to be at 25fps")
elif gt_peaks.max() < total_frames_30fps:
    print("⚠ Peaks might be at 30fps")
else:
    print("✗ Peaks exceed expected range")


