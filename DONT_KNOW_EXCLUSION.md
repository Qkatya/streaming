# "Don't Know" Peak Exclusion from ROC Analysis

## Overview

Peaks labeled as **"dont_know"** in `manual_blink_labels.csv` are now automatically excluded from the prediction peaks during ROC curve calculation. This ensures that uncertain peaks don't affect the accuracy metrics.

## How It Works

### 1. Load "Don't Know" Peaks

The `load_dont_know_peaks_to_exclude()` function:
- Loads `manual_blink_labels.csv`
- Filters for entries with `label == 'dont_know'`
- Converts local frame positions to global frame positions
- Returns array of global frame indices to exclude

### 2. Filter Prediction Peaks

During peak matching in `_match_peaks()`:
- After detecting prediction peaks automatically
- Before matching with GT peaks
- Any prediction peak within `max_offset` frames of a "dont_know" peak is removed

### 3. Calculate ROC Metrics

The filtered prediction peaks are used for:
- Matching with GT peaks
- Calculating TP, FP, FN
- Computing TPR and FPR for the ROC curve

## Label Meanings

From `manual_blink_labels.csv`:

- **`blink`**: Real blink missed by GT → **Added to GT peaks**
- **`no_blink`**: False positive → **Ignored** (not added to GT, not excluded from pred)
- **`dont_know`**: Uncertain → **Excluded from prediction peaks** ← NEW

## Implementation Details

### Function: `load_dont_know_peaks_to_exclude()`

**Location**: `blinking_split_analysis.py`

```python
def load_dont_know_peaks_to_exclude(csv_filepath, file_frame_mapping):
    """Load peaks labeled as 'dont_know' from CSV to exclude from predictions."""
    # Load CSV
    manual_labels_df = pd.read_csv(csv_filepath)
    
    # Filter for dont_know labels only
    dont_know_labels = manual_labels_df[manual_labels_df['label'] == 'dont_know']
    
    # Convert to global frame positions
    # ... (similar to load_manual_blink_labels_and_add_to_gt)
    
    return exclude_peaks
```

### Filtering Logic in `_match_peaks()`

**Location**: `visualization.py`

```python
# Filter out manually excluded prediction peaks (e.g., 'dont_know' labels)
if exclude_pred_peaks is not None and len(exclude_pred_peaks) > 0:
    # For each excluded peak, find and remove nearby prediction peaks (within max_offset)
    exclude_mask = np.ones(len(pred_peaks), dtype=bool)
    for exclude_peak in exclude_pred_peaks:
        distances = np.abs(pred_peaks - exclude_peak)
        close_peaks = distances <= max_offset
        exclude_mask &= ~close_peaks
    pred_peaks = pred_peaks[exclude_mask]
```

**Key point**: Uses `max_offset` (default 10 frames) as the tolerance for matching. Any prediction peak within 10 frames of a "dont_know" label is excluded.

## Data Flow

```
1. Load manual_blink_labels.csv
   ↓
2. Filter for label == 'dont_know'
   ↓
3. Convert to global frame positions
   ↓
4. Pass to run_blink_analysis()
   ↓
5. Pass to BlinkAnalyzer.analyze_blinks()
   ↓
6. Pass to plot_blink_analysis()
   ↓
7. Pass to _match_peaks()
   ↓
8. Filter prediction peaks (remove peaks near 'dont_know' labels)
   ↓
9. Match remaining pred peaks with GT peaks
   ↓
10. Calculate ROC metrics (TP, FP, FN, TPR, FPR)
```

## Example Output

When running the analysis:

```
================================================================================
LOADING 'DONT_KNOW' PEAKS TO EXCLUDE FROM PREDICTIONS
================================================================================

Found 22 'dont_know' labels to exclude from predictions
Processed 20 'dont_know' peaks to exclude
Skipped 2 peaks (file not found or out of bounds)

✓ Will exclude 20 'dont_know' peaks from prediction analysis

Running blink analysis for causal_preprocessor_encoder_with_smile...
```

## Why This Matters

### Before:
- "Dont_know" peaks were treated as unmatched predictions (false positives)
- This artificially inflated the FPR
- ROC curve was pessimistic

### After:
- "Dont_know" peaks are excluded from analysis
- Only confident predictions are evaluated
- ROC curve reflects true model performance on clear cases

## Use Cases

Label peaks as "dont_know" when:
- The signal is ambiguous (could be blink or artifact)
- Recording quality is poor in that region
- You're uncertain if it's a real blink
- There's an edge case you want to exclude from metrics

## Files Modified

1. **`blinking_split_analysis.py`**:
   - Added `load_dont_know_peaks_to_exclude()` function
   - Updated `run_blink_analysis()` to accept `exclude_pred_peaks`
   - Load and pass "dont_know" peaks in main execution

2. **`blink_analyzer.py`**:
   - Updated `analyze_blinks()` to accept `exclude_pred_peaks`
   - Pass to `plot_blink_analysis()`

3. **`visualization.py`**:
   - Updated `plot_blink_analysis()` to accept `exclude_pred_peaks`
   - Updated `_match_peaks()` to accept `exclude_pred_peaks`
   - Added filtering logic to remove prediction peaks near "dont_know" labels

## Configuration

The matching tolerance is controlled by `max_offset` (default 10 frames):
- A prediction peak is excluded if it's within 10 frames of any "dont_know" label
- This ensures the same tolerance used for GT/pred matching is used for exclusion

## Testing

To verify the exclusion is working:
1. Check the output message showing how many "dont_know" peaks were loaded
2. Compare ROC curves with/without "dont_know" exclusion
3. Verify that FPR decreases when "dont_know" peaks are excluded
4. Check that excluded peaks don't appear in unmatched_pred list


