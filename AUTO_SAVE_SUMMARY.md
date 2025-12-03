# Automatic Save Feature - Summary

## ✅ What Changed

I've modified your `blinking_split_analysis.py` to **automatically save** the concatenated GT and prediction signals after running the blink analysis.

## 🎯 How It Works

### 1. Run Your Analysis (As Normal)
```bash
python blinking_split_analysis.py
```

### 2. Automatic Save Happens
At the end of the blink analysis, the script now automatically:
- Saves `gt_concat` (concatenated GT signal)
- Saves `pred_concat` (concatenated prediction signal)  
- Saves `matches` (detected peaks and matching info)
- Creates a **timestamped filename** to avoid overwriting

### 3. You'll See This Output
```
================================================================================
SAVING CONCATENATED DATA FOR PEAK EDITOR
================================================================================
Including detected peaks:
  GT peaks: 245
  Pred peaks: 267
  Matched GT: 198
  Unmatched GT: 47
  Unmatched Pred: 69

✓ Saved concatenated signals to: concatenated_signals_20231203_143052.pkl
  Frames: 45000
  Duration: 1500.0 seconds
  Timestamp: 2023-12-03T14:30:52.123456

Now run:
  python launch_peak_editor_from_analysis.py concatenated_signals_20231203_143052.pkl
```

### 4. Edit Your Peaks
```bash
python launch_peak_editor_from_analysis.py concatenated_signals_20231203_143052.pkl
```

## 🔒 No Overwriting!

**Each run creates a unique file with timestamp:**
- `concatenated_signals_20231203_143052.pkl`
- `concatenated_signals_20231203_150234.pkl`
- `concatenated_signals_20231203_163015.pkl`

Your previous analyses are **never overwritten**!

## 📝 Code Changes Made

### 1. Modified `run_blink_analysis()` function
- Now returns the last `blink_analysis_model` for saving
- Returns: `TPR_lst, FNR_lst, FPR_lst, blink_analysis_model`

### 2. Added auto-save in `main()` function
- After running blink analysis
- Calls `save_for_peak_editor()` automatically
- Uses timestamped filenames

### 3. Updated `save_concatenated_for_editing.py`
- Added `use_timestamp` parameter (default: True)
- Generates unique filenames with datetime
- Returns the saved filename

## 🎁 Benefits

✅ **Automatic** - No manual saving needed
✅ **Safe** - Never overwrites previous runs
✅ **Traceable** - Timestamp shows when analysis was run
✅ **Convenient** - Just run your analysis as normal
✅ **Flexible** - Can still manually save if needed

## 📂 File Naming Pattern

```
concatenated_signals_YYYYMMDD_HHMMSS.pkl

Examples:
concatenated_signals_20231203_143052.pkl  (Dec 3, 2023 at 14:30:52)
concatenated_signals_20231203_150234.pkl  (Dec 3, 2023 at 15:02:34)
```

## 🔍 Finding Your Files

List all saved concatenated files:
```bash
ls -lt concatenated_signals_*.pkl
```

The most recent file is at the top!

## 💾 What's Saved

Each `.pkl` file contains:
```python
{
    'gt_concat': np.ndarray,        # Concatenated GT signal
    'pred_concat': np.ndarray,      # Concatenated prediction signal
    'matches': dict,                # Peak detection and matching info
    'num_frames': int,              # Total frames
    'duration_seconds': float,      # Duration in seconds
    'sample_rate': float,           # 30.0 Hz
    'timestamp': str                # ISO format timestamp
}
```

## 🚀 Quick Start

```bash
# 1. Run analysis (auto-saves concatenated data)
python blinking_split_analysis.py

# 2. Look for this in the output:
#    "python launch_peak_editor_from_analysis.py concatenated_signals_XXXXXX.pkl"

# 3. Copy and run that command
python launch_peak_editor_from_analysis.py concatenated_signals_20231203_143052.pkl

# 4. Edit peaks in browser, save changes

# 5. Use edited peaks
python
>>> import pickle
>>> with open('final_gt_peaks.pkl', 'rb') as f:
...     peaks = pickle.load(f)
```

## ✨ That's It!

Your workflow is now:
1. Run `python blinking_split_analysis.py` (auto-saves)
2. Run the command it prints out
3. Edit peaks
4. Use `final_gt_peaks.pkl`

No manual saving needed! 🎉


