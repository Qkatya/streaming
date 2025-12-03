# Using Manually Edited GT Peaks in Analysis

## Overview

The analysis has been updated to use your manually edited GT peaks from the peak editor instead of automatically detecting them.

## Configuration

In `blinking_split_analysis.py`, there are two new configuration parameters (lines 42-44):

```python
# Manual GT peaks configuration
USE_MANUAL_GT_PEAKS = True  # Set to True to use manually edited GT peaks
MANUAL_GT_PEAKS_FILE = "final_gt_peaks.pkl"  # Path to manually edited GT peaks
```

## How It Works

### 1. **Automatic Mode** (USE_MANUAL_GT_PEAKS = False)
- GT peaks are detected automatically using z-score peak detection
- Uses thresholds: `gt_th=0.06`, `prominence=1.0`

### 2. **Manual Mode** (USE_MANUAL_GT_PEAKS = True) ✅ **CURRENT**
- Loads GT peaks from `final_gt_peaks.pkl` (your manually edited peaks)
- Uses these exact peaks for matching with prediction peaks
- No automatic detection for GT peaks

## Workflow

### Step 1: Edit GT Peaks (Already Done!)
```bash
python launch_peak_editor_from_analysis.py concatenated_signals_20251203_151149.pkl
# Edit peaks, click Save Changes
# Creates: final_gt_peaks.pkl
```

### Step 2: Run Analysis with Manual GT Peaks
```bash
python blinking_split_analysis.py
```

You'll see:
```
================================================================================
LOADING MANUALLY EDITED GT PEAKS
================================================================================
✓ Loaded 1698 manually edited GT peaks from final_gt_peaks.pkl
✓ Using manually edited GT peaks for analysis

Running blink analysis for causal_preprocessor_encoder_with_smile...
```

## What Changed

### Files Modified:

1. **`blinking_split_analysis.py`**
   - Added `USE_MANUAL_GT_PEAKS` and `MANUAL_GT_PEAKS_FILE` config
   - Added `load_manual_gt_peaks()` function
   - Updated `run_blink_analysis()` to accept `manual_gt_peaks` parameter
   - Loads manual peaks before running analysis

2. **`blink_analyzer.py`**
   - Updated `analyze_blinks()` to accept `manual_gt_peaks` parameter
   - Passes manual peaks to `plot_blink_analysis()`

3. **`visualization.py`**
   - Updated `_match_peaks()` to accept `manual_gt_peaks` parameter
   - If manual peaks provided, uses them instead of detecting
   - Updated `plot_blink_analysis()` to accept and pass manual peaks

## Results

### Metrics Calculated with Manual GT Peaks:
- **TPR (True Positive Rate)**: Based on your manually verified GT peaks
- **FPR (False Positive Rate)**: Predictions not matching your manual GT
- **FNR (False Negative Rate)**: Your manual GT peaks not detected by model
- **ROC Curve**: Generated using your manual ground truth

### Benefits:
✅ More accurate metrics (based on human-verified ground truth)
✅ Removes false positives from automatic detection
✅ Adds missing peaks that automatic detection missed
✅ Consistent ground truth across multiple analyses

## Switching Between Modes

### Use Manual GT Peaks (Recommended after editing):
```python
USE_MANUAL_GT_PEAKS = True
```

### Use Automatic Detection:
```python
USE_MANUAL_GT_PEAKS = False
```

## File Locations

```
/home/katya.ivantsiv/streaming/
├── final_gt_peaks.pkl          # Your manually edited GT peaks (MAIN FILE)
├── removed_peaks.pkl            # Peaks you removed
├── added_peaks.pkl              # Peaks you added
└── concatenated_signals_*.pkl   # Original concatenated data
```

## Verification

To verify your manual peaks are being used, check the terminal output:

```
✓ Loaded 1698 manually edited GT peaks from final_gt_peaks.pkl
✓ Using manually edited GT peaks for analysis
```

If you see this, your manual peaks are being used! 🎉

## Example: Comparing Results

### Before (Automatic Detection):
```
GT peaks: 1775 (automatically detected)
Matched: 0
Unmatched GT: 1775
```

### After (Manual Editing):
```
✓ Loaded 1698 manually edited GT peaks
GT peaks: 1698 (manually verified)
Matched: [your actual matches]
Unmatched GT: [real false negatives]
```

## Notes

- The manual GT peaks are frame indices in the **concatenated signal**
- They apply to the entire dataset (all 1000 recordings combined)
- You can re-edit peaks anytime and re-run the analysis
- The original automatic detection is still available by setting `USE_MANUAL_GT_PEAKS = False`

## Summary

✅ Your manually edited GT peaks are now used in the analysis
✅ More accurate metrics based on human verification
✅ Easy to switch between manual and automatic modes
✅ All changes are backward compatible

