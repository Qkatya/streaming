# Manual Blink Labels Integration

## Overview

The `blinking_split_analysis.py` script has been updated to automatically load and integrate manually labeled blinks from `manual_blink_labels.csv` into the GT peaks used for analysis.

## What Changed

### 1. New Configuration Parameter

Added a new configuration parameter at the top of the file:

```python
MANUAL_BLINK_LABELS_CSV = "manual_blink_labels.csv"  # Path to manual blink labels CSV
```

### 2. New Function: `load_manual_blink_labels_and_add_to_gt()`

This function:
- Loads the `manual_blink_labels.csv` file
- Filters for entries labeled as "blink"
- **Filters out peaks within 10 frames of the beginning or end of each recording** (to avoid edge artifacts)
- Converts local frame positions to global frame positions using the file frame mapping
- Returns an array of global frame indices for the manually labeled blinks

### 3. File Frame Mapping

The script now builds a `file_frame_mapping` dictionary that maps each `(run_path, tar_id, side)` tuple to its `(start_frame, end_frame)` range in the concatenated signal. This is used to convert local frame positions from the CSV to global frame positions.

### 4. Updated Manual GT Peaks Loading

The section that loads manual GT peaks now:
1. Loads existing manual GT peaks from `final_gt_peaks.pkl` (if it exists)
2. Loads manual blink labels from `manual_blink_labels.csv` (if it exists)
3. Combines both sources, removing duplicates
4. Uses the combined set of GT peaks for the blink analysis

## How It Works

### Step 1: Manual Labeling (Already Done)

You've already labeled blinks using the manual labeling dashboard, which saved results to `manual_blink_labels.csv`:

```csv
timestamp,run_path,tar_id,side,peak_frame_25fps,peak_frame_30fps,peak_value,label
2025-12-03T20:44:50.247766,2025/03/17/CelloGrid-113729/...,de52bbd7-...,left,109,130,0.312,blink
2025-12-03T20:44:56.798719,2025/04/07/BraisedPergola-191010/...,71944520-...,left,148,177,0.267,blink
...
```

### Step 2: Automatic Integration (Now Happens Automatically)

When you run `blinking_split_analysis.py`, it will:

1. **Load all files** and build the file frame mapping
2. **Load manual blink labels** from CSV
3. **Convert to global frames**: For each labeled blink, find its position in the concatenated signal
4. **Combine with existing GT peaks** (if `final_gt_peaks.pkl` exists)
5. **Use for analysis**: The combined GT peaks are used for ROC curve analysis

### Example Output

```
================================================================================
LOADING MANUALLY EDITED GT PEAKS AND MANUAL BLINK LABELS
================================================================================
✓ Loaded 1698 GT peaks from final_gt_peaks.pkl

Loading manual blink labels from CSV...
Loaded 217 manual labels from CSV

Label distribution:
  blink: 150
  no_blink: 45
  dont_know: 22

Found 150 blink labels to add to GT
Processed 145 new GT peaks from manual labels
Skipped 2 peaks (file not found or out of bounds)
Skipped 3 peaks (within 10 frames of recording edges)

✓ Loaded 148 new GT peaks from manual blink labels CSV

✓ Combined total: 1820 GT peaks
  - From pickle file: 1698
  - From CSV: 148
  - After deduplication: 1820

✓ Using 1820 manually edited GT peaks for analysis
```

## CSV Format

The `manual_blink_labels.csv` file should have these columns:

- `timestamp`: When the label was created
- `run_path`: Path to the recording
- `tar_id`: Unique identifier for the file
- `side`: 'left' or 'right'
- `peak_frame_25fps`: Frame position at 25fps (used for conversion)
- `peak_frame_30fps`: Frame position at 30fps (for reference)
- `peak_value`: Peak value (for reference)
- `label`: One of 'blink', 'no_blink', or 'dont_know'

## Label Meanings

- **`blink`**: This is a real blink that was missed by the GT. These peaks are **added to the GT**.
- **`no_blink`**: This is a false positive (not a real blink). These are **ignored** (not added to GT).
- **`dont_know`**: Uncertain cases. These are **ignored** (not added to GT).

## Benefits

1. **No manual script execution needed**: Previously, you had to run `add_manual_blink_labels_to_gt.py` separately. Now it's automatic.

2. **Combines multiple sources**: Automatically merges GT peaks from:
   - `final_gt_peaks.pkl` (from peak editor)
   - `manual_blink_labels.csv` (from manual labeling dashboard)

3. **Deduplication**: Automatically removes duplicate peaks if the same blink was labeled in both sources.

4. **Transparent**: Shows exactly how many peaks came from each source and how many were combined.

## Edge Filtering

The script automatically filters out peaks that are within **10 frames** of the beginning or end of each recording. This helps avoid:
- Edge artifacts from signal processing
- Incomplete blink patterns at recording boundaries
- False positives due to recording start/stop transitions

This filtering is applied to:
- Manual blink labels from CSV
- (Note: Automatic GT peak detection may have its own edge handling)

## Troubleshooting

### "File not found in mapping"

If you see warnings like:
```
Warning: File not found in mapping: (run_path, tar_id, side)
```

This means the CSV contains a label for a file that wasn't in the current analysis run. This can happen if:
- The CSV was created with a different random sample
- The file was skipped due to missing data

**Solution**: This is OK - the script will skip these labels and continue.

### "Peak frame out of bounds"

If you see:
```
Warning: Peak frame 250 out of bounds for file ...
```

This means the peak_frame_25fps value in the CSV is larger than the actual file length. This shouldn't happen if the CSV was created correctly.

**Solution**: Check the CSV file for errors, or remove the problematic entry.

### "Skipped N peaks (within 10 frames of recording edges)"

This is normal and expected. Peaks near the edges of recordings are automatically filtered out to avoid edge artifacts.

## Configuration

To disable automatic loading of manual blink labels, set:

```python
USE_MANUAL_GT_PEAKS = False
```

This will fall back to automatic GT peak detection.

## Files Modified

1. **`blinking_split_analysis.py`**:
   - Added `MANUAL_BLINK_LABELS_CSV` configuration
   - Added `load_manual_blink_labels_and_add_to_gt()` function
   - Added `file_frame_mapping` dictionary
   - Updated manual GT peaks loading logic to combine both sources

