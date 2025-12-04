#!/usr/bin/env python3
"""
Create manual GT peaks file from manual blink labels.

This script:
1. Loads manual labels from manual_blink_labels.csv
2. Loads the original unmatched peaks pickle
3. For each file, adjusts the GT peaks based on manual labels:
   - "blink": Add this frame as a GT peak
   - "no_blink": Keep GT as is (no change)
   - "dont_know": Remove the corresponding pred peak from analysis
4. Saves a manual GT peaks file compatible with blinking_split_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import zscore
from tqdm import tqdm
import pickle
from collections import defaultdict
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes

# Configuration
LABELS_CSV = "manual_blink_labels.csv"
UNMATCHED_PEAKS_PKL = "unmatched_pred_peaks_best_tpr_th0.0000.pkl"
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")

MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1

# Peak detection parameters (same as in manual_blink_labeling_dashboard.py)
HEIGHT_THRESHOLD = 0.0
GT_PROMINENCE = 1.0
DISTANCE = 5

print("="*80)
print("CREATING MANUAL GT PEAKS FROM LABELS")
print("="*80)

# Load manual labels
if not Path(LABELS_CSV).exists():
    print(f"Error: Labels file not found: {LABELS_CSV}")
    sys.exit(1)

labels_df = pd.read_csv(LABELS_CSV)
print(f"\nLoaded {len(labels_df)} manual labels")

# Count labels by type
label_counts = labels_df['label'].value_counts()
print(f"\nLabel distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

# Load unmatched peaks data
if not Path(UNMATCHED_PEAKS_PKL).exists():
    print(f"Error: Unmatched peaks file not found: {UNMATCHED_PEAKS_PKL}")
    sys.exit(1)

unmatched_df = pd.read_pickle(UNMATCHED_PEAKS_PKL)
print(f"\nLoaded {len(unmatched_df)} unmatched peaks")

# Group labels by file (run_path, tar_id, side)
labels_by_file = defaultdict(list)
for _, row in labels_df.iterrows():
    key = (row['run_path'], row['tar_id'], row['side'])
    labels_by_file[key].append({
        'peak_frame_25fps': row['peak_frame_25fps'],
        'label': row['label']
    })

print(f"\nNumber of unique files with labels: {len(labels_by_file)}")

# Now we need to process each file and create adjusted GT peaks
# The structure should be: for each file, detect GT peaks and add manual "blink" labels

print("\n" + "="*80)
print("PROCESSING FILES AND DETECTING GT PEAKS")
print("="*80)

# We'll create a dictionary mapping (run_path, tar_id, side) -> adjusted GT peaks
manual_gt_peaks_by_file = {}

# Get unique files from unmatched peaks
unique_files = unmatched_df[['run_path', 'tar_id', 'side']].drop_duplicates()
print(f"\nProcessing {len(unique_files)} unique files...")

for _, file_row in tqdm(unique_files.iterrows(), total=len(unique_files), desc="Processing files"):
    run_path = file_row['run_path']
    tar_id = file_row['tar_id']
    side = file_row['side']
    
    key = (run_path, tar_id, side)
    
    try:
        # Load GT blendshapes
        full_run_path = ALL_DATA_PATH / run_path
        gt_blendshapes = load_ground_truth_blendshapes(full_run_path, downsample=True)
        
        if gt_blendshapes is None:
            continue
        
        # Load prediction
        pred_file = INFERENCE_OUTPUTS_PATH / run_path / f"pred_{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}_{tar_id}.npy"
        
        if not pred_file.exists():
            continue
        
        pred_blendshapes = np.load(pred_file)
        
        # Align lengths
        gt_len = gt_blendshapes.shape[0]
        pred_len = pred_blendshapes.shape[0]
        
        if pred_len < gt_len:
            padding = np.zeros((gt_len - pred_len, pred_blendshapes.shape[1]))
            pred_blendshapes = np.concatenate([pred_blendshapes, padding], axis=0)
        elif pred_len > gt_len:
            pred_blendshapes = pred_blendshapes[:gt_len]
        
        # Extract blink channel based on side
        if side == 'right':
            gt_blink = gt_blendshapes[:, 10]  # eyeBlinkRight
        else:
            gt_blink = gt_blendshapes[:, 9]   # eyeBlinkLeft
        
        # Detect GT peaks (same method as in the dashboard)
        gt_smooth = savgol_filter(gt_blink, 9, 2, mode='interp')
        gt_zscore = zscore(gt_smooth)
        gt_peaks, _ = find_peaks(gt_zscore, height=HEIGHT_THRESHOLD, prominence=GT_PROMINENCE, distance=DISTANCE)
        
        # Convert to list for easier manipulation
        adjusted_peaks = list(gt_peaks)
        
        # Apply manual labels if they exist for this file
        if key in labels_by_file:
            for label_info in labels_by_file[key]:
                peak_frame = label_info['peak_frame_25fps']
                label = label_info['label']
                
                if label == 'blink':
                    # Add this frame as a GT peak if it's not already there
                    if peak_frame not in adjusted_peaks:
                        adjusted_peaks.append(peak_frame)
                elif label == 'no_blink':
                    # Do nothing - keep GT as is
                    pass
                elif label == 'dont_know':
                    # We don't modify GT peaks for "dont_know"
                    # These will be handled by removing pred peaks later
                    pass
            
            # Sort the peaks
            adjusted_peaks.sort()
        
        # Store the adjusted peaks
        manual_gt_peaks_by_file[key] = np.array(adjusted_peaks)
        
    except Exception as e:
        print(f"\n  ✗ Error processing {run_path}/{tar_id}/{side}: {e}")
        continue

print(f"\n" + "="*80)
print(f"SUMMARY")
print(f"="*80)
print(f"Processed {len(manual_gt_peaks_by_file)} files with adjusted GT peaks")

# Calculate statistics
total_original_peaks = 0
total_adjusted_peaks = 0
total_added_peaks = 0

for key in manual_gt_peaks_by_file:
    # Reload to get original peaks for comparison
    run_path, tar_id, side = key
    try:
        full_run_path = ALL_DATA_PATH / run_path
        gt_blendshapes = load_ground_truth_blendshapes(full_run_path, downsample=True)
        
        if side == 'right':
            gt_blink = gt_blendshapes[:, 10]
        else:
            gt_blink = gt_blendshapes[:, 9]
        
        gt_smooth = savgol_filter(gt_blink, 9, 2, mode='interp')
        gt_zscore = zscore(gt_smooth)
        original_peaks, _ = find_peaks(gt_zscore, height=HEIGHT_THRESHOLD, prominence=GT_PROMINENCE, distance=DISTANCE)
        
        total_original_peaks += len(original_peaks)
        total_adjusted_peaks += len(manual_gt_peaks_by_file[key])
        total_added_peaks += len(manual_gt_peaks_by_file[key]) - len(original_peaks)
    except:
        pass

print(f"\nPeak statistics:")
print(f"  Total original GT peaks: {total_original_peaks}")
print(f"  Total adjusted GT peaks: {total_adjusted_peaks}")
print(f"  Total peaks added: {total_added_peaks}")

# Save the manual GT peaks
output_file = "manual_gt_peaks_from_labels.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(manual_gt_peaks_by_file, f)

print(f"\nSaved manual GT peaks to: {output_file}")

# Also create a file for pred peaks to remove (for "dont_know" labels)
pred_peaks_to_remove = defaultdict(list)
for key, labels in labels_by_file.items():
    for label_info in labels:
        if label_info['label'] == 'dont_know':
            pred_peaks_to_remove[key].append(label_info['peak_frame_25fps'])

if pred_peaks_to_remove:
    remove_file = "pred_peaks_to_remove_from_labels.pkl"
    with open(remove_file, 'wb') as f:
        pickle.dump(dict(pred_peaks_to_remove), f)
    print(f"Saved pred peaks to remove to: {remove_file}")
    print(f"  Total pred peaks to remove: {sum(len(v) for v in pred_peaks_to_remove.values())}")

print(f"\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nNext steps:")
print(f"1. Use this file in blinking_split_analysis.py by setting:")
print(f"   USE_MANUAL_GT_PEAKS = True")
print(f"   MANUAL_GT_PEAKS_FILE = '{output_file}'")
print(f"2. Run the analysis to see improved ROC curve")


