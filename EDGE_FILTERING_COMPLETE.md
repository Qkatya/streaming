# Complete Edge Filtering Implementation

## Summary

Edge filtering has been implemented comprehensively to filter peaks within **10 frames** of:
1. Beginning/end of each individual recording (for manual CSV labels)
2. Beginning/end of the concatenated signal (for automatic detection)
3. **All file boundaries** in the concatenated signal (between recordings)

## Implementation Details

### 1. Manual CSV Labels (`load_manual_blink_labels_and_add_to_gt`)

**Location**: `blinking_split_analysis.py`

Filters peaks within individual recordings before converting to global frames:
```python
# Filter out peaks within edge_margin frames from beginning or end of recording
if peak_frame_25fps < edge_margin or peak_frame_25fps >= (file_length - edge_margin):
    skipped_edge_count += 1
    continue
```

### 2. Automatic Peak Detection (`_detect_blink_peaks`)

**Location**: `visualization.py`

Filters peaks at the edges of the concatenated signal:
```python
# Filter out peaks within edge_margin frames from beginning or end
signal_length = len(signal)
valid_peaks = peaks[(peaks >= edge_margin) & (peaks < signal_length - edge_margin)]
```

### 3. File Boundary Filtering (`_filter_peaks_at_file_boundaries`)

**Location**: `visualization.py` (NEW FUNCTION)

Filters peaks near all file boundaries in the concatenated signal:
```python
def _filter_peaks_at_file_boundaries(peaks, file_boundaries, edge_margin=10):
    """Filter out peaks within edge_margin frames of any file boundary."""
    valid_mask = np.ones(len(peaks), dtype=bool)
    
    for i, peak in enumerate(peaks):
        for boundary in file_boundaries:
            if abs(peak - boundary) < edge_margin:
                valid_mask[i] = False
                break
    
    return peaks[valid_mask]
```

This function is applied to:
- **Manual GT peaks** (after loading from pickle/CSV)
- **Automatic GT peaks** (after detection)
- **Prediction peaks** (after detection)

## Data Flow

### Manual GT Peaks:
1. Load from `final_gt_peaks.pkl` or `manual_blink_labels.csv`
2. Filter at concatenated signal edges (start/end)
3. **Filter at all file boundaries** ← NEW
4. Use for analysis

### Automatic GT Peaks:
1. Detect peaks in concatenated signal
2. Filter at concatenated signal edges (start/end) - built into `_detect_blink_peaks`
3. **Filter at all file boundaries** ← NEW
4. Use for analysis

### Prediction Peaks:
1. Detect peaks in concatenated signal
2. Filter at concatenated signal edges (start/end) - built into `_detect_blink_peaks`
3. **Filter at all file boundaries** ← NEW
4. Use for matching with GT

## Configuration

The edge margin is configurable via:
```python
EDGE_MARGIN = 10  # Number of frames from boundaries to exclude
```

This value is passed through the entire pipeline:
- `blinking_split_analysis.py` → `run_blink_analysis()`
- `blink_analyzer.py` → `analyze_blinks()`
- `visualization.py` → `plot_blink_analysis()` → `_match_peaks()`
- Applied in `_detect_blink_peaks()` and `_filter_peaks_at_file_boundaries()`

## Files Modified

1. **`visualization.py`**:
   - Added `_filter_peaks_at_file_boundaries()` function
   - Updated `_detect_blink_peaks()` to add `edge_margin` parameter
   - Updated `_match_peaks()` to add `edge_margin` and `file_boundaries` parameters
   - Updated `plot_blink_analysis()` to add `edge_margin` and `file_boundaries` parameters
   - Applied boundary filtering to all peak types

2. **`blink_analyzer.py`**:
   - Updated `analyze_blinks()` to add `edge_margin` and `file_boundaries` parameters
   - Passes these to `plot_blink_analysis()`

3. **`blinking_split_analysis.py`**:
   - Added `EDGE_MARGIN` configuration parameter
   - Updated `load_manual_blink_labels_and_add_to_gt()` to filter within recordings
   - Updated `run_blink_analysis()` to add `edge_margin` and `file_boundaries` parameters
   - Passes `file_boundaries` list to the analysis pipeline

## Example Output

When running the analysis, you'll see:
```
Found 150 blink labels to add to GT
Processed 145 new GT peaks from manual labels
Skipped 2 peaks (file not found or out of bounds)
Skipped 3 peaks (within 10 frames of recording edges)
```

And during peak detection:
- Automatic GT/pred peaks are filtered at concatenated signal edges
- All peaks (manual and automatic) are filtered at file boundaries
- This ensures no peaks are detected near any edge or boundary

## Benefits

✅ **Comprehensive**: Filters at all possible edge locations  
✅ **Consistent**: Same filtering applied to GT and predictions  
✅ **Configurable**: Single `EDGE_MARGIN` parameter controls all filtering  
✅ **Transparent**: Clear separation between different filtering stages  
✅ **Robust**: Handles both manual and automatic peak detection  

## Testing

To verify the filtering is working:
1. Check that no peaks appear within 10 frames of file boundaries in plots
2. Verify the "Skipped N peaks (within 10 frames of recording edges)" message
3. Confirm peak counts are reduced after filtering
4. Look at the z-scored signal plots to see boundary lines and verify no peaks near them


