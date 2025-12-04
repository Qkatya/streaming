#!/usr/bin/env python3
"""
Apply manual blink labels to ground truth data.

This script:
1. Loads manual labels from manual_blink_labels.csv
2. For each label:
   - "blink": Adds a blink peak to the GT at the labeled frame
   - "no_blink": Does nothing (keeps GT as is)
   - "dont_know": Marks the prediction peak for removal in analysis
3. Saves the adjusted GT files and a mapping of adjustments
4. Re-runs the ROC curve analysis to show improvement
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
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
ADJUSTED_GT_DIR = Path("adjusted_gt_blendshapes")
ADJUSTMENTS_LOG = "gt_adjustments_log.pkl"
PEAKS_TO_REMOVE_LOG = "pred_peaks_to_remove.pkl"

# Peak injection parameters
BLINK_PEAK_HEIGHT = 0.8  # Height of injected blink in GT
BLINK_PEAK_WIDTH = 5  # Width of injected blink (frames on each side)

print("="*80)
print("APPLYING MANUAL LABELS TO GROUND TRUTH")
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

# Group labels by run_path, tar_id, side
grouped = labels_df.groupby(['run_path', 'tar_id', 'side'])
print(f"\nNumber of unique files to process: {len(grouped)}")

# Create output directory
ADJUSTED_GT_DIR.mkdir(exist_ok=True)

# Track adjustments
adjustments_log = []
peaks_to_remove = defaultdict(list)  # For "dont_know" labels

print("\n" + "="*80)
print("PROCESSING FILES")
print("="*80)

processed_files = 0
skipped_files = 0

for (run_path, tar_id, side), group in tqdm(grouped, desc="Processing files"):
    try:
        # Load original GT blendshapes
        full_run_path = ALL_DATA_PATH / run_path
        gt_file = full_run_path / "landmarks_and_blendshapes.npz"
        
        if not gt_file.exists():
            print(f"\n  ✗ GT file not found: {gt_file}")
            skipped_files += 1
            continue
        
        # Load GT data
        gt_data = np.load(gt_file)
        gt_blendshapes = gt_data['blendshapes'].copy()
        
        # Downsample from 200fps to 25fps (skip every 8th frame for 25fps)
        # Actually, the original code uses skip every 6th frame for ~33fps
        # Let's use the same downsampling as in the original code
        mask = np.ones(gt_blendshapes.shape[0], dtype=bool)
        mask[5::6] = False
        gt_blendshapes_downsampled = gt_blendshapes[mask]
        
        # Get the blink channel (eyeBlinkRight for right, eyeBlinkLeft for left)
        if side == 'right':
            blink_channel = 10  # eyeBlinkRight
        else:
            blink_channel = 9   # eyeBlinkLeft
        
        # Track if we made any changes
        made_changes = False
        file_adjustments = []
        
        # Process each label for this file
        for _, row in group.iterrows():
            label = row['label']
            peak_frame_25fps = row['peak_frame_25fps']
            peak_value = row['peak_value']
            
            if label == 'blink':
                # Add a blink peak to GT at this frame
                # Create a Gaussian-like peak centered at peak_frame_25fps
                center = peak_frame_25fps
                
                # Make sure we're within bounds
                if center < 0 or center >= len(gt_blendshapes_downsampled):
                    print(f"\n  ✗ Peak frame {center} out of bounds for {run_path}")
                    continue
                
                # Create a blink peak (Gaussian shape)
                for offset in range(-BLINK_PEAK_WIDTH, BLINK_PEAK_WIDTH + 1):
                    frame_idx = center + offset
                    if 0 <= frame_idx < len(gt_blendshapes_downsampled):
                        # Gaussian falloff
                        distance = abs(offset)
                        amplitude = BLINK_PEAK_HEIGHT * np.exp(-(distance**2) / (2 * (BLINK_PEAK_WIDTH/2)**2))
                        
                        # Add to existing value (don't replace, to preserve any existing signal)
                        current_value = gt_blendshapes_downsampled[frame_idx, blink_channel]
                        new_value = max(current_value, amplitude)  # Take max to avoid reducing existing peaks
                        gt_blendshapes_downsampled[frame_idx, blink_channel] = new_value
                
                made_changes = True
                file_adjustments.append({
                    'type': 'add_blink',
                    'frame': peak_frame_25fps,
                    'peak_value': peak_value
                })
                
            elif label == 'no_blink':
                # Do nothing - GT stays as is
                file_adjustments.append({
                    'type': 'no_change',
                    'frame': peak_frame_25fps,
                    'peak_value': peak_value
                })
                
            elif label == 'dont_know':
                # Mark this prediction peak for removal in analysis
                key = (run_path, tar_id, side)
                peaks_to_remove[key].append(peak_frame_25fps)
                
                file_adjustments.append({
                    'type': 'remove_pred',
                    'frame': peak_frame_25fps,
                    'peak_value': peak_value
                })
        
        # Save adjusted GT if we made changes
        if made_changes:
            # Upsample back to 200fps by repeating frames
            # This is a simple approach - we insert the same value 6 times
            gt_blendshapes_upsampled = np.zeros_like(gt_blendshapes)
            
            # Copy non-downsampled data back
            downsample_indices = np.where(mask)[0]
            for i, orig_idx in enumerate(downsample_indices):
                if i < len(gt_blendshapes_downsampled):
                    gt_blendshapes_upsampled[orig_idx] = gt_blendshapes_downsampled[i]
            
            # Interpolate the skipped frames
            skipped_indices = np.where(~mask)[0]
            for skip_idx in skipped_indices:
                # Find nearest neighbors
                prev_idx = skip_idx - 1
                next_idx = skip_idx + 1
                
                while prev_idx >= 0 and not mask[prev_idx]:
                    prev_idx -= 1
                while next_idx < len(mask) and not mask[next_idx]:
                    next_idx += 1
                
                if prev_idx >= 0 and next_idx < len(gt_blendshapes_upsampled):
                    # Linear interpolation
                    alpha = (skip_idx - prev_idx) / (next_idx - prev_idx)
                    gt_blendshapes_upsampled[skip_idx] = (
                        (1 - alpha) * gt_blendshapes_upsampled[prev_idx] +
                        alpha * gt_blendshapes_upsampled[next_idx]
                    )
                elif prev_idx >= 0:
                    gt_blendshapes_upsampled[skip_idx] = gt_blendshapes_upsampled[prev_idx]
                elif next_idx < len(gt_blendshapes_upsampled):
                    gt_blendshapes_upsampled[skip_idx] = gt_blendshapes_upsampled[next_idx]
            
            # Save adjusted GT
            output_file = ADJUSTED_GT_DIR / f"{run_path.replace('/', '_')}_{tar_id}_{side}_adjusted.npz"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                output_file,
                blendshapes=gt_blendshapes_upsampled,
                original_file=str(gt_file)
            )
        
        # Log adjustments
        adjustments_log.append({
            'run_path': run_path,
            'tar_id': tar_id,
            'side': side,
            'adjustments': file_adjustments,
            'file_modified': made_changes
        })
        
        processed_files += 1
        
    except Exception as e:
        print(f"\n  ✗ Error processing {run_path}/{tar_id}/{side}: {e}")
        skipped_files += 1
        continue

print(f"\n" + "="*80)
print(f"SUMMARY")
print(f"="*80)
print(f"Successfully processed: {processed_files} files")
print(f"Skipped: {skipped_files} files")

# Save logs
with open(ADJUSTMENTS_LOG, 'wb') as f:
    pickle.dump(adjustments_log, f)
print(f"\nSaved adjustments log to: {ADJUSTMENTS_LOG}")

with open(PEAKS_TO_REMOVE_LOG, 'wb') as f:
    pickle.dump(dict(peaks_to_remove), f)
print(f"Saved peaks to remove to: {PEAKS_TO_REMOVE_LOG}")

# Print summary statistics
total_blinks_added = sum(1 for adj in adjustments_log for a in adj['adjustments'] if a['type'] == 'add_blink')
total_no_change = sum(1 for adj in adjustments_log for a in adj['adjustments'] if a['type'] == 'no_change')
total_pred_removed = sum(1 for adj in adjustments_log for a in adj['adjustments'] if a['type'] == 'remove_pred')

print(f"\nAdjustment statistics:")
print(f"  Blinks added to GT: {total_blinks_added}")
print(f"  No changes (no_blink): {total_no_change}")
print(f"  Pred peaks marked for removal (dont_know): {total_pred_removed}")

print(f"\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nNext steps:")
print(f"1. Run the ROC analysis with adjusted GT files")
print(f"2. Compare ROC curves before and after adjustments")


