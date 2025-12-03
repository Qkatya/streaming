# Unmatched Predicted Peaks - Usage Guide

## Overview

The `blinking_split_analysis.py` script now automatically saves unmatched predicted peaks (false positives) from the threshold that gives the best True Positive Rate (TPR).

## What Gets Saved

After running the blink analysis and generating the ROC curve, the script:

1. **Identifies the best threshold**: Finds the prediction threshold that maximizes TPR
2. **Extracts unmatched peaks**: Gets all predicted peaks that don't match any ground truth peaks at that threshold
3. **Maps to original files**: Converts concatenated frame numbers back to individual file frame numbers
4. **Saves metadata**: Stores complete information about each unmatched peak

## Output File

The script creates a pickle file named:
```
unmatched_pred_peaks_best_tpr_th{threshold}.pkl
```

For example: `unmatched_pred_peaks_best_tpr_th0.0345.pkl`

## DataFrame Structure

The saved DataFrame contains the following columns for each unmatched predicted peak:

| Column | Type | Description |
|--------|------|-------------|
| `peak_frame_concat` | int | Frame number in the concatenated signal |
| `peak_frame_in_file` | int | Frame number within the original file (0-indexed) |
| `peak_value` | float | Blink intensity value at the peak |
| `run_path` | str | Path to the recording (e.g., `/mnt/A3000/Recordings/v2_data/run_123`) |
| `tar_id` | int | TAR file ID within the run |
| `side` | str | Side of the face ('left' or 'right') |
| `threshold` | float | The prediction threshold used (best TPR threshold) |

## Usage Example

```python
import pandas as pd

# Load the unmatched peaks
df = pd.read_pickle('unmatched_pred_peaks_best_tpr_th0.0345.pkl')

# View summary
print(f"Total unmatched peaks: {len(df)}")
print(f"Threshold used: {df['threshold'].iloc[0]}")

# Group by file to see which files have most false positives
false_positives_per_file = df.groupby(['run_path', 'tar_id', 'side']).size()
print("\nFalse positives per file:")
print(false_positives_per_file.sort_values(ascending=False).head())

# Access specific peak information
for idx, row in df.head().iterrows():
    print(f"\nPeak {idx}:")
    print(f"  File: {row['run_path']}")
    print(f"  TAR ID: {row['tar_id']}, Side: {row['side']}")
    print(f"  Frame in file: {row['peak_frame_in_file']}")
    print(f"  Peak value: {row['peak_value']:.4f}")
```

## Use Cases

### 1. Analyze False Positive Patterns
Identify which recordings or conditions lead to more false positives:
```python
# Files with most false positives
fp_counts = df.groupby('run_path').size().sort_values(ascending=False)
problematic_files = fp_counts.head(10)
```

### 2. Visual Inspection
Load the original signals and inspect false positive peaks:
```python
from blendshapes_data_utils import load_ground_truth_blendshapes
from blinking_split_analysis import load_prediction

# Get a specific false positive
row = df.iloc[0]
gt = load_ground_truth_blendshapes(row['run_path'])
pred = load_prediction(row['run_path'], row['tar_id'], row['side'], 
                       'model_name', history_size, lookahead_size)

# Extract blink signals and plot around the false positive
frame = row['peak_frame_in_file']
window = 100
# ... plot gt and pred around frame ± window
```

### 3. Refine Model
Use false positive examples to:
- Understand what patterns the model incorrectly identifies as blinks
- Create a dataset for model refinement
- Adjust post-processing or thresholds

### 4. Statistics
```python
# Peak value distribution
print(f"Mean peak value: {df['peak_value'].mean():.4f}")
print(f"Median peak value: {df['peak_value'].median():.4f}")
print(f"Peak value range: [{df['peak_value'].min():.4f}, {df['peak_value'].max():.4f}]")

# Distribution across sides
print("\nFalse positives by side:")
print(df['side'].value_counts())
```

## Notes

- The threshold is automatically selected to maximize TPR, which may result in higher false positive rates
- Frame numbers are 0-indexed
- The `peak_frame_in_file` allows you to directly index into the original file's data
- All peaks are from the prediction signal (not ground truth)
- The analysis uses the same preprocessing as the ROC curve generation (smoothing, z-score normalization, etc.)

## Integration with Existing Tools

You can use the saved peaks with existing visualization tools:

```python
# Load unmatched peaks
unmatched_df = pd.read_pickle('unmatched_pred_peaks_best_tpr_th0.0345.pkl')

# For each unique file, visualize the false positives
for (run_path, tar_id, side), group in unmatched_df.groupby(['run_path', 'tar_id', 'side']):
    peaks_in_file = group['peak_frame_in_file'].values
    # ... load and visualize with peaks marked
```

