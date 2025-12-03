# Manual Blink Labeling Workflow

## Overview
This workflow allows you to manually review and correct blink detection results, then use those corrections to improve the ground truth (GT) peaks used in analysis.

## Files Involved

### 1. `manual_blink_labeling_dashboard.py`
**Purpose**: Interactive dashboard for manually labeling unmatched prediction peaks

**Features**:
- Loads video snippets of unmatched prediction peaks
- Shows video with prediction signal and all peaks (matched/unmatched)
- Three labeling options:
  - **BLINK**: This is a real blink that was missed by GT → adds peak to GT
  - **NO BLINK**: This is a false positive prediction → GT is correct, do nothing
  - **DON'T KNOW**: Uncertain → removes peak from pred in analysis
- Saves labels to `manual_blink_labels.csv`
- Keyboard shortcuts: B=Blink, N=No Blink, U=Don't Know, ←/→=Navigate

**Usage**:
```bash
python manual_blink_labeling_dashboard.py
# Open browser to http://localhost:8054
```

### 2. `manual_blink_labels.csv`
**Purpose**: Persistent storage of manual labels

**Format**:
```csv
timestamp,run_path,tar_id,side,peak_frame_25fps,peak_frame_30fps,peak_value,label
2025-12-03T20:44:41.445004,2025/04/09/...,uuid,left,28,33,0.259,no_blink
```

**Labels**:
- `blink`: Real blink missed by GT → add to GT
- `no_blink`: False positive in pred → keep GT as is
- `dont_know`: Uncertain → remove from pred

### 3. `plot_corrected_tags_comparison.py`
**Purpose**: Visualize original vs corrected GT tags

**Features**:
- Plots concatenated signals in same order as `blinking_split_analysis.py`
- Top subplot: Original GT peaks (automatic detection)
- Bottom subplot: Corrected GT peaks (after applying manual labels)
- Shows effect of manual corrections

**Usage**:
```bash
python plot_corrected_tags_comparison.py
```

**Configuration**:
- `NUM_EXAMPLES = 20`: Number of files to plot (set to None for all)
- Uses same random sampling as `blinking_split_analysis.py`

### 4. `blinking_split_analysis.py` (UPDATED)
**Purpose**: Main analysis script, now uses corrected GT peaks

**Key Changes**:
- Loads manual corrections from `manual_blink_labels.csv`
- Applies corrections to GT peaks:
  - **"blink"** → adds peak at frame (25fps)
  - **"no_blink"** → keeps GT as is
  - **"dont_know"** → keeps GT as is (only affects pred)
- Uses corrected GT for ROC curve analysis

**New Functions**:
- `load_manual_corrections_from_labels()`: Loads corrections from CSV
- `apply_manual_corrections_to_peaks()`: Applies corrections to peak array
- `convert_manual_peaks_dict_to_concat()`: Updated to accept corrections

## Workflow Steps

### Step 1: Generate Unmatched Peaks
Run analysis to find unmatched prediction peaks:
```bash
python blinking_split_analysis.py
```
This creates `unmatched_pred_peaks_*.pkl` with peaks to review.

### Step 2: Download Video Snippets
Download videos for the unmatched peaks:
```bash
python download_unmatched_peak_videos.py
```
This creates videos in `video_snippets/` directory.

### Step 3: Manual Labeling
Review and label the video snippets:
```bash
python manual_blink_labeling_dashboard.py
```
- Watch each video
- Label as Blink / No Blink / Don't Know
- Labels saved automatically to `manual_blink_labels.csv`

### Step 4: Visualize Corrections
See the effect of your corrections:
```bash
python plot_corrected_tags_comparison.py
```
This shows before/after comparison of GT peaks.

### Step 5: Re-run Analysis with Corrections
Run analysis again with corrected GT:
```bash
python blinking_split_analysis.py
```
Now uses corrected GT peaks from your manual labels!

## Label Interpretation

### "blink" Label
- **Meaning**: The prediction found a real blink that GT missed
- **Action**: Add a peak to GT at `peak_frame_25fps`
- **Effect**: Increases GT peak count, improves TPR

### "no_blink" Label
- **Meaning**: The prediction is a false positive
- **Action**: Do nothing to GT (GT is correct)
- **Effect**: GT stays the same, this peak remains as FP

### "dont_know" Label
- **Meaning**: Uncertain if it's a blink
- **Action**: Remove peak from pred analysis
- **Effect**: Reduces pred peak count, may affect FPR

## Statistics Tracking

The workflow tracks:
- Total manual labels created
- Files with corrections
- Label distribution (blink/no_blink/dont_know)
- Original vs corrected peak counts
- TPR/FPR changes

## File Ordering

All scripts use the **same file ordering** from `blinking_split_analysis.py`:
- Same split file: `LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl`
- Same random sampling: `NUM_RANDOM_SAMPLES = 1000`, `RANDOM_SEED = 42`
- Ensures consistency across all visualizations and analyses

## Notes

- Labels are **append-only** in CSV (never overwritten)
- Latest label for each peak is used if multiple exist
- Frame numbers: 25fps for blendshapes, 30fps for video
- Corrections are applied on top of automatic detection or manual pkl peaks

