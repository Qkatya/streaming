# Integration Guide - Edit Concatenated GT Peaks

## What You Want

You want to edit the **concatenated** GT peaks from your entire analysis (all examples combined), not individual recordings.

## Quick Solution

### Option 1: Add to Your Existing Analysis Script

Add this code to your `blinking_split_analysis.py` after you run the blink analysis:

```python
# After your blink analysis (around line 344 or wherever you have results)
from save_concatenated_for_editing import save_for_peak_editor

# Your existing code that creates blink_analysis_model
blink_analysis_model = analyzer.analyze_blinks(
    gt_th=gt_th,
    model_th=model_th,
    blendshapes_list=blendshapes_list,
    pred_blends_list=pred_blends_list,
    max_offset=quantizer,
    significant_gt_movenents=significant_gt_movenents
)

# ADD THIS: Save concatenated data for editing
save_for_peak_editor(
    gt_concat=blink_analysis_model['gt_concat'],
    pred_concat=blink_analysis_model['pred_concat'],
    matches=blink_analysis_model['matches'],
    output_file='concatenated_signals.pkl'
)
```

Then run:
```bash
python launch_peak_editor_from_analysis.py concatenated_signals.pkl
```

### Option 2: Save Data Manually in Python

If you already have the concatenated data in memory:

```python
import pickle

# Your concatenated arrays (from blink_analysis_model)
gt_concat = blink_analysis_model['gt_concat']
pred_concat = blink_analysis_model['pred_concat']
matches = blink_analysis_model['matches']

# Save it
data = {
    'gt_concat': gt_concat,
    'pred_concat': pred_concat,
    'matches': matches,
    'num_frames': len(gt_concat),
    'duration_seconds': len(gt_concat) / 30.0
}

with open('concatenated_signals.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Saved! Now run: python launch_peak_editor_from_analysis.py concatenated_signals.pkl")
```

### Option 3: Direct Launch from Python

If you have the arrays in memory and want to launch immediately:

```python
from peak_editor_app import create_app

# Your concatenated arrays
gt_concat = blink_analysis_model['gt_concat']
pred_concat = blink_analysis_model['pred_concat']

# Launch editor directly
app = create_app(gt_concat, pred_concat, sample_rate=30.0)
app.run_server(debug=True, port=8050)
```

## Complete Workflow

### Step 1: Run Your Analysis
```bash
python blinking_split_analysis.py
```

### Step 2: Save Concatenated Data

Add this to your script or run in Python console:
```python
from save_concatenated_for_editing import save_for_peak_editor

# Assuming you have blink_analysis_model from your analysis
save_for_peak_editor(
    blink_analysis_model['gt_concat'],
    blink_analysis_model['pred_concat'],
    blink_analysis_model['matches']
)
```

### Step 3: Launch Peak Editor
```bash
python launch_peak_editor_from_analysis.py concatenated_signals.pkl
```

### Step 4: Edit Peaks
- Browser opens at http://127.0.0.1:8050
- Navigate through 50-second windows
- Click on peaks to remove them
- Click elsewhere to add peaks
- Save changes

### Step 5: Use Edited Peaks

The edited GT peaks are saved to `final_gt_peaks.pkl`:

```python
import pickle

# Load edited peaks
with open('final_gt_peaks.pkl', 'rb') as f:
    edited_gt_peaks = pickle.load(f)

# These are frame indices in the concatenated signal
print(f"Total edited GT peaks: {len(edited_gt_peaks)}")

# Use them to recalculate your metrics
# ... your analysis code ...
```

## What Gets Saved

After editing, you'll have:

1. **`removed_peaks.pkl`** - Set of frame indices you removed from original GT peaks
2. **`added_peaks.pkl`** - Set of frame indices you added as new GT peaks
3. **`final_gt_peaks.pkl`** - Final array of GT peak frame indices (USE THIS!)

All frame indices are relative to the **concatenated signal**, not individual recordings.

## Example: Complete Integration

Here's a complete example you can add to your `blinking_split_analysis.py`:

```python
# Near the end of your main() function, after blink analysis:

def save_and_launch_peak_editor():
    """Save concatenated data and optionally launch peak editor."""
    from save_concatenated_for_editing import save_for_peak_editor
    
    # Save concatenated signals
    save_for_peak_editor(
        gt_concat=blink_analysis_model['gt_concat'],
        pred_concat=blink_analysis_model['pred_concat'],
        matches=blink_analysis_model['matches'],
        output_file='concatenated_signals.pkl'
    )
    
    # Ask user if they want to launch editor
    response = input("\nLaunch peak editor now? (y/n): ")
    if response.lower() == 'y':
        from peak_editor_app import create_app
        app = create_app(
            blink_analysis_model['gt_concat'],
            blink_analysis_model['pred_concat'],
            sample_rate=30.0
        )
        print("\nLaunching peak editor at http://127.0.0.1:8050")
        print("Press Ctrl+C to stop")
        app.run_server(debug=True, port=8050)

# Call it after your analysis
if CALCULATE_BLINK_METRICS:
    save_and_launch_peak_editor()
```

## Summary

**To edit concatenated GT peaks:**

1. Run your analysis to get `gt_concat` and `pred_concat`
2. Save them: `save_for_peak_editor(gt_concat, pred_concat, matches)`
3. Launch editor: `python launch_peak_editor_from_analysis.py concatenated_signals.pkl`
4. Edit in browser
5. Use `final_gt_peaks.pkl` in your analysis

That's it! 🎯


