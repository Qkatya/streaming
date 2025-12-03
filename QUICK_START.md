# Quick Start Guide - Peak Editor

## Installation

```bash
pip install dash plotly pandas numpy scipy
```

## Test with Synthetic Data (Recommended First)

```bash
python test_peak_editor.py
```

This will:
- Generate synthetic blink signals
- Launch the app at http://127.0.0.1:8050
- Let you practice using the interface

## Use with Real Data

### Option 1: Command Line (Easiest)

```bash
python launch_peak_editor.py 3663
```

Replace `3663` with your desired row index.

### Option 2: Interactive Selection

```bash
python launch_peak_editor.py
```

The script will show available recordings and prompt you to select one.

### Option 3: Python Script

```python
from launch_peak_editor import load_data_for_editor
from peak_editor_app import create_app

# Load data
gt_signal, pred_signal, metadata = load_data_for_editor(row_idx=3663)

# Launch app
app = create_app(gt_signal, pred_signal)
app.run_server(debug=True, port=8050)
```

## How to Edit Peaks

1. **Navigate**: Use Previous/Next buttons to move between 50-second windows
2. **Remove Peak**: Click on a green or orange circle (existing GT peak)
3. **Add Peak**: Click anywhere else on the graph
4. **Save**: Click "💾 Save Changes" button

## Output Files

After editing and saving, you'll have:
- `removed_peaks.pkl` - Peaks you removed
- `added_peaks.pkl` - Peaks you added  
- `final_gt_peaks.pkl` - Final GT peaks (use this in your analysis)

## Loading Results

```python
import pickle

# Load final peaks
with open('final_gt_peaks.pkl', 'rb') as f:
    final_gt_peaks = pickle.load(f)

print(f"Total GT peaks: {len(final_gt_peaks)}")
```

## Tips

- Green circles = Matched GT peaks (good!)
- Orange circles = Unmatched GT peaks (might want to review)
- Red X's = False positive predictions
- The z-score view makes peaks easier to identify
- Changes are saved incrementally - you can stop and resume anytime

## Troubleshooting

**Port already in use?**
```python
app.run_server(debug=True, port=8051)  # Use different port
```

**Can't find data?**
Check that your paths in `blinking_split_analysis.py` are correct:
- `ALL_DATA_PATH`
- `INFERENCE_OUTPUTS_PATH`
- `SPLIT_DF_PATH`

**No peaks showing?**
The default thresholds might be too strict. Adjust in `peak_editor_app.py`:
```python
self.original_gt_peaks = self._detect_initial_peaks(
    self.gt_filtered, 
    height_threshold=0.3,  # Lower = more peaks
    min_prominence=0.5     # Lower = more peaks
)
```


