# Ground Truth Peak Editor - Interactive Dash Application

An interactive web application for editing ground truth (GT) peaks in blink detection analysis. The app displays z-score comparisons with peak matching and allows you to add or remove GT peaks by clicking on the graph.

## Features

- **Interactive Visualization**: View z-score comparison between GT and predicted signals with peak matching
- **Window-based Navigation**: Browse through 50-second windows of data
- **Click-to-Edit**: Add or remove GT peaks by clicking on the graph
- **Persistent Storage**: All modifications are saved to pickle files
- **Real-time Statistics**: See counts of matched/unmatched peaks and modifications

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_dash.txt
```

Or install individually:
```bash
pip install dash plotly pandas numpy scipy
```

## Usage

### Quick Start

Launch the app with a specific recording:

```bash
python launch_peak_editor.py [row_index]
```

For example:
```bash
python launch_peak_editor.py 3663
```

If you don't provide a row index, the script will show you available recordings and prompt you to select one.

### Using in Python

You can also use the app programmatically:

```python
from peak_editor_app import create_app
import numpy as np

# Load your data
gt_signal = ...  # Your GT blink signal (1D numpy array)
pred_signal = ...  # Your predicted blink signal (1D numpy array)

# Create and launch app
app = create_app(gt_signal, pred_signal, sample_rate=30.0)
app.run_server(debug=True, port=8050)
```

## How to Use the App

### Navigation

1. **Window Navigation**: Use the "Previous Window" and "Next Window" buttons to move between 50-second windows
2. **Window Display**: Shows current window number and total windows

### Editing Peaks

1. **Remove a Peak**: Click on or near an existing GT peak (green or orange circle marker)
   - The peak will be removed if you click within 3 frames of it
   
2. **Add a Peak**: Click anywhere else on the graph where you want to add a new GT peak
   - A new peak will be added at the clicked location

3. **Visual Feedback**: 
   - Green circles = Matched GT peaks (found in predictions)
   - Orange circles = Unmatched GT peaks (not found in predictions)
   - Green X's = Matched prediction peaks
   - Red X's = Unmatched prediction peaks

### Saving Changes

Click the "💾 Save Changes" button to save your modifications. This will create/update three pickle files:

1. **`removed_peaks.pkl`**: Set of frame indices for peaks you removed
2. **`added_peaks.pkl`**: Set of frame indices for peaks you added
3. **`final_gt_peaks.pkl`**: Array of all GT peak indices after modifications

## Data Files

### Input Files
The app loads data from your existing analysis pipeline:
- Ground truth blendshapes from `landmarks_and_blendshapes.npz`
- Predictions from inference outputs

### Output Files
All output files are saved in the current directory:

- `removed_peaks.pkl`: Peaks removed from original GT
- `added_peaks.pkl`: Peaks added to GT
- `final_gt_peaks.pkl`: Final set of GT peaks

### File Format

All pickle files use Python sets (for removed/added) or numpy arrays (for final):

```python
import pickle
import numpy as np

# Load removed peaks
with open('removed_peaks.pkl', 'rb') as f:
    removed_peaks = pickle.load(f)  # Set of frame indices

# Load added peaks
with open('added_peaks.pkl', 'rb') as f:
    added_peaks = pickle.load(f)  # Set of frame indices

# Load final GT peaks
with open('final_gt_peaks.pkl', 'rb') as f:
    final_gt_peaks = pickle.load(f)  # Numpy array of frame indices
```

## Information Panel

The info panel shows:
- **Total GT Peaks**: Current number of GT peaks after modifications
- **Removed Peaks**: Number of peaks you've removed
- **Added Peaks**: Number of peaks you've added
- **Window**: Current frame range being displayed
- **GT Peaks in Window**: Number of GT peaks in current window
- **Matched Peaks**: Number of GT peaks matched with predictions
- **Unmatched GT Peaks**: Number of GT peaks not matched with predictions
- **Unmatched Pred Peaks**: Number of prediction peaks not matched with GT

## Configuration

You can modify these parameters in `peak_editor_app.py`:

```python
WINDOW_SIZE_SECONDS = 50  # Size of each window in seconds
SAMPLE_RATE = 30.0  # Hz
REMOVED_PEAKS_FILE = "removed_peaks.pkl"
ADDED_PEAKS_FILE = "added_peaks.pkl"
FINAL_GT_PEAKS_FILE = "final_gt_peaks.pkl"
```

## Peak Detection Parameters

The app uses z-score based peak detection with these default parameters:
- **Height threshold**: 0.5 (z-score units)
- **Prominence threshold**: 1.0 (z-score units)
- **Max offset for matching**: 10 frames

These can be adjusted in the `PeakEditor` class initialization.

## Workflow Integration

### Loading Data from Analysis Pipeline

The `launch_peak_editor.py` script integrates with your existing `blinking_split_analysis.py` workflow:

1. Loads split dataframe
2. Extracts GT and predicted blendshapes for a specific recording
3. Extracts blink signals (average of left and right eye blinks)
4. Launches the interactive editor

### Using Modified Peaks in Analysis

After editing, you can load the final peaks in your analysis:

```python
import pickle
import numpy as np

# Load final GT peaks
with open('final_gt_peaks.pkl', 'rb') as f:
    final_gt_peaks = pickle.load(f)

# Use in your analysis
print(f"Number of GT peaks: {len(final_gt_peaks)}")
print(f"Peak locations (frames): {final_gt_peaks}")
```

## Troubleshooting

### App won't start
- Check that all dependencies are installed: `pip install -r requirements_dash.txt`
- Ensure port 8050 is not already in use
- Check that data files exist and are accessible

### Can't see peaks
- Adjust the z-score thresholds in the code
- Check that your signals have sufficient variation
- Verify that data is loaded correctly

### Changes not saving
- Check file permissions in the current directory
- Ensure you clicked the "Save Changes" button
- Check console for error messages

## Technical Details

### Peak Detection Algorithm

The app uses z-score based peak detection:
1. Calculate z-score of the blink signal
2. Find peaks using `scipy.signal.find_peaks` with height and prominence thresholds
3. Match GT and prediction peaks within a maximum time offset

### Peak Matching

Peaks are matched between GT and predictions if:
- They are within `max_offset` frames of each other (default: 10 frames = 0.33 seconds)
- The closest prediction peak is selected for each GT peak

### Data Processing

1. GT signal is filtered with a simple moving average (window size 3)
2. Z-scores are calculated for both GT and prediction signals
3. Peaks are detected on the z-score signals
4. Window data is extracted for efficient visualization

## Future Enhancements

Possible improvements:
- Keyboard shortcuts for navigation
- Zoom and pan controls
- Batch editing mode
- Undo/redo functionality
- Export to different formats
- Comparison view with original peaks

## Support

For issues or questions, refer to the main analysis scripts:
- `blinking_split_analysis.py`
- `blink_analyzer.py`
- `visualization.py`


