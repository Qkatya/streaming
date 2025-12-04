# Manual Peaks Workflow - Updated

## Overview
This document explains how to use manual blink labels to correct GT peaks and visualize the results.

## Files Involved

1. **manual_blink_labels.csv** - Contains your manual tags from the dashboard (144 blinks, 36 dont_know, 35 no_blink)
2. **manual_gt_peaks_from_labels.pkl** - Dictionary mapping (run_path, tar_id, side) → peak indices per file
3. **concatenated_signals_TIMESTAMP.pkl** - Concatenated GT/Pred signals WITH file order information
4. **plot_manual_corrections_comparison.py** - Visualization script

## Workflow

### Step 1: Re-run Analysis to Generate File Order

The concatenated signals file needs to include file order and offsets for proper peak conversion.

```bash
python blinking_split_analysis.py
```

This will create a new `concatenated_signals_TIMESTAMP.pkl` file that includes:
- `gt_concat` - Concatenated GT signal
- `pred_concat` - Concatenated pred signal  
- `file_order` - List of (run_path, tar_id, side) tuples in concatenation order
- `file_offsets` - List of frame offsets for each file
- `matches` - Peak matching information

### Step 2: Update Plot Script with New File

Edit `plot_manual_corrections_comparison.py` line 29 to use the new file:

```python
CONCATENATED_SIGNALS_FILE = "concatenated_signals_20251203_XXXXXX.pkl"  # Use new timestamp
```

### Step 3: Run Visualization

```bash
python plot_manual_corrections_comparison.py
```

This will:
1. Load the concatenated signals
2. Detect original automatic GT peaks
3. Load manual GT peaks dictionary
4. Convert manual peaks to concatenated format using file order
5. Show two subplots:
   - Top: Original automatic GT detection
   - Bottom: After manual corrections
6. Highlight added/removed peaks

## Color Legend

- 🟢 **GREEN circles** = Matched GT peaks (correctly detected by model)
- 🔵 **BLUE diamonds** = Unmatched GT peaks (missed by model)
- 🟨 **YELLOW squares** = Matched Pred peaks (correct predictions)
- 🔴 **RED X's** = Unmatched Pred peaks (false positives)
- 💜 **MAGENTA stars** = ADDED peaks (from manual labeling)
- 🩵 **CYAN X's** = REMOVED peaks (from manual labeling)

## What Changed

### Modified Files:

1. **save_concatenated_for_editing.py**
   - Added `file_order` and `file_offsets` parameters
   - Now saves file order information to pkl file

2. **blinking_split_analysis.py**
   - Calculates file offsets during concatenation
   - Passes file_order and file_offsets to save function

3. **plot_manual_corrections_comparison.py**
   - Loads file order from concatenated signals
   - Converts manual peaks dict to concatenated format
   - Shows two subplots with linked x-axis
   - Uses clear colors (red, green, yellow, blue)

## Troubleshooting

**Problem**: "No file order in concatenated file"
**Solution**: Re-run `blinking_split_analysis.py` to generate a new concatenated_signals file

**Problem**: "Manual corrections added 0 GT peaks"
**Solution**: The old concatenated_signals file doesn't have file order. Generate a new one.

**Problem**: Graphs look identical
**Solution**: Need to re-run analysis with updated code to get file order information

## Next Steps

After visualization:
1. Verify that added/removed peaks are correct
2. Use the manual GT peaks for final ROC analysis
3. Compare TPR/FPR before and after corrections


