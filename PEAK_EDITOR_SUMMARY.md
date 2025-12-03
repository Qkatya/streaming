# Peak Editor Application - Complete Summary

## Overview

I've created a complete interactive Dash application for editing ground truth (GT) peaks in blink detection analysis. The application allows you to:

1. **View** z-score comparisons between GT and predicted signals with peak matching
2. **Navigate** through 50-second windows of data
3. **Edit** GT peaks by clicking on the graph (add or remove)
4. **Save** modifications to pickle files for use in analysis
5. **Analyze** your modifications with visualization tools

## Files Created

### Core Application Files

1. **`peak_editor_app.py`** (Main application)
   - `PeakEditor` class: Manages GT peaks and modifications
   - Dash app with interactive visualization
   - Automatic saving/loading of modifications
   - Real-time peak matching and statistics

2. **`launch_peak_editor.py`** (Data loader)
   - Integrates with your existing analysis pipeline
   - Loads data from split dataframe
   - Extracts blink signals from blendshapes
   - Launches the app with real data

3. **`test_peak_editor.py`** (Testing tool)
   - Generates synthetic blink signals
   - Allows you to test the interface before using real data
   - Useful for training and debugging

4. **`analyze_modifications.py`** (Analysis tool)
   - Loads and analyzes saved modifications
   - Generates statistics and visualizations
   - Exports data to CSV format

### Documentation Files

5. **`PEAK_EDITOR_README.md`** (Comprehensive documentation)
   - Detailed feature descriptions
   - Complete usage instructions
   - Configuration options
   - Troubleshooting guide

6. **`QUICK_START.md`** (Quick reference)
   - Installation instructions
   - Quick start examples
   - Common tasks
   - Tips and tricks

7. **`requirements_dash.txt`** (Dependencies)
   - All required Python packages
   - Version specifications

## Data Flow

```
Input Data:
  ├─ Ground truth blendshapes (from landmarks_and_blendshapes.npz)
  └─ Predicted blendshapes (from inference outputs)
       ↓
  Extract blink signals (average of left/right eye blinks)
       ↓
  Launch Peak Editor App
       ↓
  Interactive editing (add/remove peaks)
       ↓
Output Files:
  ├─ removed_peaks.pkl (set of removed peak indices)
  ├─ added_peaks.pkl (set of added peak indices)
  └─ final_gt_peaks.pkl (final array of GT peak indices)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install dash plotly pandas numpy scipy
```

### 2. Test with Synthetic Data (Recommended)
```bash
python test_peak_editor.py
```

### 3. Use with Real Data
```bash
python launch_peak_editor.py 3663
```
(Replace 3663 with your desired row index)

### 4. Edit Peaks
- Navigate with Previous/Next buttons
- Click on peaks to remove them
- Click elsewhere to add new peaks
- Save changes with the Save button

### 5. Analyze Results
```bash
python analyze_modifications.py
```

## Key Features

### Interactive Visualization
- **Z-score plots**: Makes peaks easier to identify
- **Color-coded markers**:
  - Green circles = Matched GT peaks
  - Orange circles = Unmatched GT peaks
  - Green X's = Matched predictions
  - Red X's = False positive predictions
- **Real-time updates**: Changes reflected immediately

### Window-Based Navigation
- 50-second windows (1500 frames at 30 Hz)
- Previous/Next buttons for easy navigation
- Window counter shows progress
- Maintains modifications across windows

### Persistent Storage
- Modifications saved to pickle files
- Automatic loading of previous edits
- Can stop and resume anytime
- Three output files for different use cases

### Statistics Panel
Shows real-time information:
- Total GT peaks (after modifications)
- Number of removed peaks
- Number of added peaks
- Current window frame range
- Peaks in current window
- Matching statistics

## Peak Detection Algorithm

The app uses z-score based peak detection:

1. **Filter** GT signal with moving average (window=3)
2. **Calculate** z-scores for both GT and prediction signals
3. **Detect** peaks using scipy.signal.find_peaks:
   - Height threshold: 0.5 z-score units
   - Prominence threshold: 1.0 z-score units
4. **Match** GT and prediction peaks:
   - Maximum offset: 10 frames (0.33 seconds)
   - Closest prediction peak selected for each GT peak

## Output File Formats

### removed_peaks.pkl
```python
# Python set of frame indices
{150, 342, 567, 891, ...}
```

### added_peaks.pkl
```python
# Python set of frame indices
{234, 456, 789, ...}
```

### final_gt_peaks.pkl
```python
# Numpy array of frame indices (sorted)
array([  45,  123,  234,  456,  567, ...])
```

## Using Results in Analysis

```python
import pickle
import numpy as np

# Load final GT peaks
with open('final_gt_peaks.pkl', 'rb') as f:
    final_gt_peaks = pickle.load(f)

# Use in your analysis
print(f"Number of GT peaks: {len(final_gt_peaks)}")

# Convert to time
peak_times = final_gt_peaks / 30.0  # Assuming 30 Hz

# Calculate metrics
from blinking_split_analysis import calculate_metrics
# ... your analysis code ...
```

## Integration with Existing Code

The peak editor integrates seamlessly with your existing analysis:

### From blinking_split_analysis.py
- Uses same data paths and configuration
- Loads from same split dataframe
- Extracts blink signals using same method
- Compatible with BlinkAnalyzer class

### From blink_analyzer.py
- Uses same blink extraction logic
- Compatible with analyze_blinks() method
- Can replace detected peaks with edited peaks

### From visualization.py
- Uses same peak detection function (_detect_blink_peaks)
- Uses same peak matching function (_match_peaks)
- Compatible with z-score visualization

## Workflow Example

### Complete workflow from start to finish:

```bash
# 1. Test the interface (optional but recommended)
python test_peak_editor.py

# 2. Edit real data
python launch_peak_editor.py 3663

# 3. In the browser:
#    - Navigate through windows
#    - Remove false positive GT peaks (orange circles)
#    - Add missing GT peaks
#    - Click Save Changes

# 4. Analyze your modifications
python analyze_modifications.py

# 5. Use in your analysis
python
>>> import pickle
>>> with open('final_gt_peaks.pkl', 'rb') as f:
...     peaks = pickle.load(f)
>>> print(f"Edited {len(peaks)} GT peaks")
```

## Configuration Options

### In peak_editor_app.py:
```python
WINDOW_SIZE_SECONDS = 50  # Window size
SAMPLE_RATE = 30.0  # Sampling rate
REMOVED_PEAKS_FILE = "removed_peaks.pkl"
ADDED_PEAKS_FILE = "added_peaks.pkl"
FINAL_GT_PEAKS_FILE = "final_gt_peaks.pkl"
```

### Peak detection thresholds:
```python
height_threshold = 0.5  # Z-score height
min_prominence = 1.0    # Z-score prominence
max_offset = 10         # Frame offset for matching
```

## Troubleshooting

### Common Issues

**Port 8050 already in use:**
```python
app.run_server(debug=True, port=8051)
```

**No peaks visible:**
- Lower the height_threshold (e.g., 0.3)
- Lower the min_prominence (e.g., 0.5)

**Data not loading:**
- Check paths in blinking_split_analysis.py
- Verify split dataframe exists
- Ensure GT and prediction files exist

**Changes not saving:**
- Check write permissions
- Click the Save button
- Check console for errors

## Advanced Usage

### Custom Data Loading
```python
from peak_editor_app import create_app
import numpy as np

# Load your custom data
gt_signal = np.load('my_gt_signal.npy')
pred_signal = np.load('my_pred_signal.npy')

# Launch app
app = create_app(gt_signal, pred_signal, sample_rate=30.0)
app.run_server(debug=True, port=8050)
```

### Batch Processing
```python
from launch_peak_editor import load_data_for_editor

# Process multiple recordings
row_indices = [3663, 3953, 3642]

for row_idx in row_indices:
    gt, pred, meta = load_data_for_editor(row_idx)
    # Save with unique names
    # ... launch app or process automatically ...
```

### Programmatic Peak Editing
```python
from peak_editor_app import PeakEditor

# Create editor
editor = PeakEditor(gt_signal, pred_signal)

# Add/remove peaks programmatically
editor.toggle_peak(150)  # Remove peak at frame 150
editor.toggle_peak(200)  # Add peak at frame 200

# Get current peaks
current_peaks = editor.get_current_gt_peaks()

# Save
editor.save_modifications()
```

## Future Enhancements

Possible improvements:
- Keyboard shortcuts (arrow keys for navigation)
- Zoom and pan controls
- Batch editing mode (edit multiple recordings)
- Undo/redo functionality
- Export to different formats (JSON, CSV)
- Comparison view (before/after)
- Automatic peak suggestion based on ML
- Multi-user collaboration features

## Support and Feedback

For issues or questions:
1. Check the README files
2. Review the Quick Start guide
3. Test with synthetic data first
4. Check console output for errors
5. Verify data paths and file permissions

## Summary

You now have a complete, production-ready application for interactive GT peak editing. The application:

✓ Integrates with your existing analysis pipeline
✓ Provides intuitive click-to-edit interface
✓ Saves modifications persistently
✓ Includes comprehensive documentation
✓ Provides analysis and visualization tools
✓ Supports both testing and production use
✓ Exports data in multiple formats

Start with the test script, then move to real data, and use the analysis tools to verify your edits!


