#!/usr/bin/env python3
"""
Add manual blink labels to GT peaks.

This script:
1. Loads manual_blink_labels.csv
2. Replicates the file ordering from blinking_split_analysis.py
3. For entries labeled as "blink", calculates the global frame position
4. Adds new GT peaks and saves to final_gt_peaks.pkl
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
MANUAL_LABELS_CSV = "manual_blink_labels.csv"
FINAL_GT_PEAKS_PKL = "final_gt_peaks.pkl"
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")

# Random sampling configuration (must match blinking_split_analysis.py)
NUM_RANDOM_SAMPLES = 1000
RANDOM_SEED = 42


def load_split_and_sample(split_df_path, num_samples=None, random_seed=None):
    """Load split dataframe and optionally sample."""
    print(f"Loading split dataframe from: {split_df_path}")
    with open(split_df_path, 'rb') as f:
        split_df = pickle.load(f)
    
    print(f"Split dataframe shape: {split_df.shape}")
    
    # Random sampling if requested
    if num_samples is not None and num_samples < len(split_df):
        print(f"Randomly sampling {num_samples} rows (seed={random_seed})...")
        split_df = split_df.sample(n=num_samples, random_state=random_seed)
        print(f"Sampled {len(split_df)} rows")
    
    # Create list of (run_path, tar_id, side) tuples
    run_path_tar_id_side_tuples = []
    for idx, row in split_df.iterrows():
        run_path = row['run_path']
        tar_id = row['tar_id']
        side = row.get('side', None)
        run_path_tar_id_side_tuples.append((run_path, tar_id, side))
    
    print(f"Loaded {len(run_path_tar_id_side_tuples)} files from split")
    return run_path_tar_id_side_tuples


def load_gt_blendshapes_from_inference(run_path):
    """Load GT blendshapes from inference outputs (matches blinking_split_analysis.py)."""
    inference_file = INFERENCE_OUTPUTS_PATH / run_path / "blendshapes.pkl"
    
    if not inference_file.exists():
        return None
    
    try:
        with open(inference_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract GT blendshapes
        if isinstance(data, dict) and 'gt' in data:
            return data['gt']
        else:
            return None
    except Exception as e:
        print(f"Error loading {inference_file}: {e}")
        return None


def build_file_frame_mapping(run_path_tar_id_side_tuples):
    """
    Build a mapping from (run_path, tar_id, side) to global frame offset.
    This replicates the logic from blinking_split_analysis.py main() function.
    
    Returns:
        dict: {(run_path, tar_id, side): (start_frame, end_frame)}
        int: total_frames
    """
    print("\n" + "="*80)
    print("BUILDING FILE FRAME MAPPING")
    print("="*80)
    
    file_frame_mapping = {}
    current_frame = 0
    skipped = 0
    
    for run_path, tar_id, side in tqdm(run_path_tar_id_side_tuples, desc="Building frame mapping"):
        # Load GT blendshapes to get the number of frames
        gt_blendshapes = load_gt_blendshapes_from_inference(run_path)
        
        if gt_blendshapes is None:
            skipped += 1
            continue
        
        num_frames = gt_blendshapes.shape[0]
        start_frame = current_frame
        end_frame = current_frame + num_frames
        
        file_frame_mapping[(run_path, tar_id, side)] = (start_frame, end_frame)
        current_frame = end_frame
    
    print(f"\nBuilt frame mapping for {len(file_frame_mapping)} files")
    print(f"Skipped {skipped} files (GT not found)")
    print(f"Total frames: {current_frame}")
    
    return file_frame_mapping, current_frame


def main():
    print("="*80)
    print("ADDING MANUAL BLINK LABELS TO GT PEAKS")
    print("="*80)
    
    # Step 1: Load manual labels
    print(f"\nLoading manual labels from: {MANUAL_LABELS_CSV}")
    manual_labels_df = pd.read_csv(MANUAL_LABELS_CSV)
    print(f"Loaded {len(manual_labels_df)} manual labels")
    print(f"Columns: {manual_labels_df.columns.tolist()}")
    
    # Show label distribution
    label_counts = manual_labels_df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Filter for blink labels
    blink_labels = manual_labels_df[manual_labels_df['label'] == 'blink'].copy()
    print(f"\nFound {len(blink_labels)} blink labels to add")
    
    if len(blink_labels) == 0:
        print("No blink labels to add. Exiting.")
        return
    
    # Step 2: Load existing GT peaks
    print(f"\nLoading existing GT peaks from: {FINAL_GT_PEAKS_PKL}")
    with open(FINAL_GT_PEAKS_PKL, 'rb') as f:
        existing_gt_peaks = pickle.load(f)
    
    print(f"Existing GT peaks: {len(existing_gt_peaks)} peaks")
    print(f"Type: {type(existing_gt_peaks)}")
    print(f"Shape: {existing_gt_peaks.shape if hasattr(existing_gt_peaks, 'shape') else 'N/A'}")
    
    # Step 3: Load split and build file frame mapping
    run_path_tar_id_side_tuples = load_split_and_sample(
        SPLIT_DF_PATH,
        num_samples=NUM_RANDOM_SAMPLES,
        random_seed=RANDOM_SEED
    )
    
    file_frame_mapping, total_frames = build_file_frame_mapping(run_path_tar_id_side_tuples)
    print(f"Total frames in concatenated signal: {total_frames}")
    
    # Step 4: Process each blink label
    print("\n" + "="*80)
    print("PROCESSING BLINK LABELS")
    print("="*80)
    
    new_peaks = []
    skipped_count = 0
    
    for idx, row in tqdm(blink_labels.iterrows(), total=len(blink_labels), desc="Processing blink labels"):
        run_path = row['run_path']
        tar_id = row['tar_id']
        side = row['side']
        peak_frame_25fps = int(row['peak_frame_25fps'])
        
        # Look up the file in our mapping
        key = (run_path, tar_id, side)
        if key not in file_frame_mapping:
            print(f"Warning: File not found in mapping: {key}")
            skipped_count += 1
            continue
        
        start_frame, end_frame = file_frame_mapping[key]
        
        # Calculate global frame position
        global_frame = start_frame + peak_frame_25fps
        
        # Validate frame is within bounds
        if global_frame < start_frame or global_frame >= end_frame:
            print(f"Warning: Peak frame {peak_frame_25fps} out of bounds for file {run_path}")
            print(f"  File frames: [{start_frame}, {end_frame})")
            print(f"  Global frame: {global_frame}")
            skipped_count += 1
            continue
        
        new_peaks.append(global_frame)
    
    print(f"\nProcessed {len(new_peaks)} new peaks")
    print(f"Skipped {skipped_count} peaks")
    
    # Step 5: Combine with existing peaks
    if len(new_peaks) > 0:
        new_peaks_array = np.array(new_peaks, dtype=np.int64)
        
        # Combine and sort
        combined_peaks = np.concatenate([existing_gt_peaks, new_peaks_array])
        combined_peaks = np.unique(combined_peaks)  # Remove duplicates and sort
        
        print(f"\nExisting GT peaks: {len(existing_gt_peaks)}")
        print(f"New peaks: {len(new_peaks_array)}")
        print(f"Combined peaks (after deduplication): {len(combined_peaks)}")
        
        # Show some statistics
        print(f"\nFirst 10 new peaks (global frames): {sorted(new_peaks_array)[:10]}")
        print(f"Last 10 new peaks (global frames): {sorted(new_peaks_array)[-10:]}")
        
        # Step 6: Save updated GT peaks
        print(f"\nSaving updated GT peaks to: {FINAL_GT_PEAKS_PKL}")
        with open(FINAL_GT_PEAKS_PKL, 'wb') as f:
            pickle.dump(combined_peaks, f)
        
        print("✓ Successfully updated GT peaks!")
    else:
        print("\nNo new peaks to add.")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
