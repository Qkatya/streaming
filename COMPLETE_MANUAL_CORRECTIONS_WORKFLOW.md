# Complete Manual Corrections Workflow

## Overview

The system now integrates **TWO sources** of manual GT corrections:

1. **Peak Editor App** (`peak_editor_app.py`) - For editing GT peaks in the concatenated signal
2. **Manual Blink Labeling Dashboard** (`manual_blink_labeling_dashboard.py`) - For labeling prediction peaks

Both are now properly integrated into the analysis pipeline!

## Two Sources of Manual Corrections

### Source 1: Peak Editor App (GT Peak Editing)
**File**: `peak_editor_app.py`  
**Output Files**:
- `removed_peaks.pkl` - Peaks removed from GT (196 peaks)
- `added_peaks.pkl` - Peaks added to GT (84 peaks)
- `final_gt_peaks.pkl` - Final GT peaks after edits (THIS IS USED!)

**What it does**:
- Allows you to click on the concatenated signal to add/remove GT peaks
- Works on the CONCATENATED signal (all files merged together)
- Saves peaks as frame indices in the concatenated space

**When to use**:
- When you want to fix GT peaks that were incorrectly detected
- When you see missing blinks in the GT
- When you see false positives in the GT

### Source 2: Manual Blink Labeling Dashboard (Prediction Labeling)
**File**: `manual_blink_labeling_dashboard.py`  
**Output File**:
- `manual_blink_labels.csv` - Labels for prediction peaks (215 labels total)

**Label Types**:
1. **`blink`** (144 labels) - Prediction found a real blink that GT missed → **ADD to GT**
2. **`no_blink`** (35 labels) - Prediction is a false positive → **REMOVE from Pred**
3. **`dont_know`** (36 labels) - Uncertain → **REMOVE from Pred** (exclude from analysis)

**What it does**:
- Shows you unmatched prediction peaks with video snippets
- You label each peak as blink/no_blink/dont_know
- Saves labels per file with frame indices (25fps)

**When to use**:
- After running analysis with unmatched pred peaks
- When you want to verify if pred peaks are real blinks or false positives
- When you want to improve GT by adding missed blinks

## How They Work Together

### Step 1: Initial Analysis
```bash
python blinking_split_analysis.py
```
- Detects GT peaks automatically
- Detects pred peaks at various thresholds
- Saves unmatched pred peaks for review

### Step 2: Edit GT Peaks (Optional)
```bash
python launch_peak_editor_from_analysis.py concatenated_signals_TIMESTAMP.pkl
```
- Opens interactive dashboard
- Click to add/remove GT peaks
- Click "Save Changes" → creates `final_gt_peaks.pkl`

### Step 3: Label Prediction Peaks (Optional)
```bash
python manual_blink_labeling_dashboard.py
```
- Reviews unmatched pred peaks
- Labels each as blink/no_blink/dont_know
- Saves to `manual_blink_labels.csv`

### Step 4: Re-run Analysis with Corrections
```bash
python blinking_split_analysis.py
```
The system now:
1. **Loads `final_gt_peaks.pkl`** (from peak editor)
   - Converts concatenated peaks to per-file format
2. **Loads `manual_blink_labels.csv`** (from labeling dashboard)
   - Extracts GT corrections (blink labels → add to GT)
   - Extracts pred removals (no_blink/dont_know → remove from pred)
3. **Applies BOTH corrections**:
   - Uses manually edited GT peaks as base
   - Adds "blink" labels to GT
   - Removes "no_blink"/"dont_know" peaks from pred
4. **Runs analysis** with fully corrected GT and pred peaks

## Technical Details

### File Format Conversion

**Peak Editor Format** (concatenated):
```python
final_gt_peaks.pkl = np.array([265, 1500, 2614, ...])  # Concatenated frame indices
```

**Analysis Format** (per-file):
```python
peaks_dict = {
    (run_path, tar_id, side): np.array([10, 25, 42, ...]),  # Per-file frame indices
    ...
}
```

The system automatically converts between these formats!

### Correction Application Order

For each file:
1. **Load base GT peaks**:
   - If file is in `final_gt_peaks.pkl` → use those peaks
   - Otherwise → detect automatically
2. **Apply CSV corrections**:
   - Add peaks for "blink" labels
   - (No changes for "no_blink"/"dont_know" - those affect pred)
3. **Convert to concatenated space**
4. **Pass to BlinkAnalyzer** which handles pred peak removal

### Code Changes

**`blinking_split_analysis.py`**:
- Added `convert_concatenated_peaks_to_per_file()` - Converts peak editor format to per-file format
- Updated `load_manual_corrections_from_labels()` - Returns BOTH gt_corrections and pred_removals
- Updated `main()` - Loads and applies both sources of corrections

**`plot_corrected_tags_comparison.py`**:
- Already had correct logic for removing pred peaks
- Shows visual comparison of original vs corrected peaks

## Current Status

✅ **Peak Editor Integration**: `final_gt_peaks.pkl` (196 removed, 84 added)  
✅ **Label CSV Integration**: `manual_blink_labels.csv` (144 blink, 35 no_blink, 36 dont_know)  
✅ **Format Conversion**: Concatenated ↔ Per-file  
✅ **Correction Application**: Both sources applied correctly  

## Verification

To verify the integration is working:

```bash
python blinking_split_analysis.py
```

Look for these messages:
```
================================================================================
LOADING MANUAL CORRECTIONS FROM LABELS CSV
================================================================================
✓ Loaded 215 manual labels from manual_blink_labels.csv
  Found GT corrections for 163 files
  Found pred removals for 163 files
  Label counts: blink=144 (add to GT), no_blink=35 (remove from pred), dont_know=36 (remove from pred)

================================================================================
LOADING MANUALLY EDITED GT PEAKS
================================================================================
✓ Loaded 1698 manually edited GT peaks (concatenated format) from final_gt_peaks.pkl
  Note: These are concatenated peaks and will be converted to per-file format
  Converting concatenated peaks to per-file format...
  Converted to 569 files with manual peaks
  Files with manual peaks (from pkl): 569
  Files with automatic peaks: 431
  Files with manual corrections (from CSV): 163
✓ Using manually edited GT peaks with corrections for analysis
  Total manual GT peaks in concatenated signal: 2854
```

This confirms:
- Both correction sources are loaded
- Format conversion is working
- Corrections are being applied

## Next Steps

1. **Run analysis** to see the impact of corrections on ROC curves
2. **Review results** to see if TPR/FPR improved
3. **Iterate** if needed:
   - Add more labels in the dashboard
   - Edit more GT peaks in the editor
   - Re-run analysis

The system is now fully integrated and ready to use! 🎉

