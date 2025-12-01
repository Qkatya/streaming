#!/usr/bin/env python3
"""
Debug script to find which samples fail to load GT/prediction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Same configuration as blink_review_dashboard.py
PICKLE_FILE = "unmatched_pred_peaks_prom0.5000.pkl"
VIDEO_SNIPPETS_DIR = Path("video_snippets_marked")
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'

MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1

sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes

# Load data
df = pd.read_pickle(PICKLE_FILE)
if 'is_blink' not in df.columns:
    df['is_blink'] = None

df = df.sort_values('global_frame').reset_index(drop=True)
all_samples = df.copy()

split_df = pd.read_pickle(SPLIT_DF_PATH)

def load_gt_pred_data(row):
    """Load GT and prediction data for a given row."""
    run_path = row['run_path']
    
    if 'tar_id' in row and 'side' in row:
        tar_id = row['tar_id']
        side = row['side']
    else:
        matching_rows = split_df[split_df['run_path'] == run_path]
        if len(matching_rows) == 0:
            return None, None, None, f"run_path not found in split_df"
        split_row = matching_rows.iloc[0]
        tar_id = split_row['tar_id']
        side = split_row['side']
    
    # Load GT blendshapes
    full_run_path = ALL_DATA_PATH / run_path
    try:
        gt_blendshapes = load_ground_truth_blendshapes(full_run_path, downsample=True)
    except Exception as e:
        return None, None, None, f"GT load failed: {e}"
    
    # Load prediction
    pred_file = INFERENCE_OUTPUTS_PATH / run_path / f"pred_{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}_{tar_id}.npy"
    
    if not pred_file.exists():
        return None, None, None, f"Prediction file not found: {pred_file}"
    
    pred_blendshapes = np.load(pred_file)
    
    # Align lengths
    gt_len = gt_blendshapes.shape[0]
    pred_len = pred_blendshapes.shape[0]
    
    if pred_len < gt_len:
        padding = np.zeros((gt_len - pred_len, pred_blendshapes.shape[1]))
        pred_blendshapes = np.concatenate([pred_blendshapes, padding], axis=0)
    elif pred_len > gt_len:
        pred_blendshapes = pred_blendshapes[:gt_len]
    
    return gt_blendshapes, pred_blendshapes, row['local_frame'], None

print("="*80)
print("DEBUGGING MISSING SAMPLES")
print("="*80)

successful = []
failed = []

for idx, row in all_samples.iterrows():
    global_frame = row['global_frame']
    file_index = row['file_index']
    local_frame = row['local_frame']
    timestamp_seconds = row['timestamp_seconds']
    run_path_sanitized = row['run_path'].replace('/', '_').replace('\\', '_')
    
    filename = f"{global_frame}_{file_index}_{local_frame}_{timestamp_seconds:.2f}_{run_path_sanitized}.mp4"
    video_path = VIDEO_SNIPPETS_DIR / filename
    
    if not video_path.exists():
        failed.append({
            'index': idx,
            'global_frame': global_frame,
            'run_path': row['run_path'],
            'reason': 'Video file not found'
        })
        continue
    
    # Load GT and pred data
    gt_bs, pred_bs, peak_frame, error = load_gt_pred_data(row)
    
    if gt_bs is None:
        failed.append({
            'index': idx,
            'global_frame': global_frame,
            'run_path': row['run_path'],
            'reason': error
        })
    else:
        successful.append(idx)

print(f"\nTotal samples: {len(all_samples)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\n" + "="*80)
    print("FAILED SAMPLES:")
    print("="*80)
    failed_df = pd.DataFrame(failed)
    print(failed_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("UNIQUE FAILURE REASONS:")
    print("="*80)
    for reason in failed_df['reason'].unique():
        count = len(failed_df[failed_df['reason'] == reason])
        print(f"{reason}: {count} samples")

