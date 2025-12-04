# Summary of Changes - Manual Blink Labels Integration

## What Was Done

### Problem
The manual blink labels saved in `manual_blink_labels.csv` were not being automatically loaded and added to the GT peaks during blink analysis. You had to manually run a separate script (`add_manual_blink_labels_to_gt.py`) to integrate them.

### Solution
Updated `blinking_split_analysis.py` to automatically load and integrate manual blink labels from the CSV file during analysis.

## Key Changes

### 1. Added Configuration
```python
MANUAL_BLINK_LABELS_CSV = "manual_blink_labels.csv"
```

### 2. Added File Frame Mapping
The script now tracks which frames belong to which files:
```python
file_frame_mapping = {}  # Maps (run_path, tar_id, side) -> (start_frame, end_frame)
```

This is populated as files are loaded and used to convert local frame positions from the CSV to global frame positions in the concatenated signal.

### 3. New Function: `load_manual_blink_labels_and_add_to_gt()`
This function:
- Loads `manual_blink_labels.csv`
- Filters for entries labeled as "blink" (ignores "no_blink" and "dont_know")
- **Filters out peaks within 10 frames of recording edges** (to avoid edge artifacts)
- Converts local frame positions to global positions using the file frame mapping
- Returns array of global frame indices

### 4. Updated Manual GT Peaks Loading
The blink analysis section now:
1. Loads existing manual GT peaks from `final_gt_peaks.pkl` (if exists)
2. Loads manual blink labels from `manual_blink_labels.csv` (if exists)
3. Combines both sources, removing duplicates
4. Uses the combined set for analysis

## Benefits

✅ **Automatic**: No need to run separate scripts - just run the analysis  
✅ **Combines Sources**: Merges peaks from pickle file and CSV automatically  
✅ **Edge Filtering**: Removes peaks within 10 frames of recording boundaries  
✅ **Deduplication**: Automatically removes duplicate peaks  
✅ **Transparent**: Shows exactly how many peaks came from each source  
✅ **Robust**: Handles missing files and out-of-bounds peaks gracefully  

## Example Output

```
================================================================================
LOADING MANUALLY EDITED GT PEAKS AND MANUAL BLINK LABELS
================================================================================
✓ Loaded 1698 GT peaks from final_gt_peaks.pkl

Loading manual blink labels from CSV...
Loaded 217 manual labels from CSV

Label distribution:
  blink: 150
  no_blink: 45
  dont_know: 22

Found 150 blink labels to add to GT
Processed 145 new GT peaks from manual labels
Skipped 2 peaks (file not found or out of bounds)
Skipped 3 peaks (within 10 frames of recording edges)

✓ Loaded 145 new GT peaks from manual blink labels CSV

✓ Combined total: 1820 GT peaks
  - From pickle file: 1698
  - From CSV: 145
  - After deduplication: 1820

✓ Using 1820 manually edited GT peaks for analysis
```

## Edge Filtering Details

Peaks are filtered out if they are within **10 frames** of:
1. **Beginning/end of each individual recording** (in manual CSV labels)
2. **Beginning/end of the concatenated signal** (for automatic detection)
3. **All file boundaries in the concatenated signal** (between recordings)

### For Manual CSV Labels:
- `peak_frame_25fps < 10` → Filtered (too close to start of recording)
- `peak_frame_25fps >= (file_length - 10)` → Filtered (too close to end of recording)

### For Automatic GT/Pred Peak Detection:
- Peaks within 10 frames of the concatenated signal start/end → Filtered
- Peaks within 10 frames of any file boundary → Filtered

This comprehensive filtering helps avoid:
- Edge artifacts from signal processing
- Incomplete blink patterns at recording boundaries
- False positives from recording start/stop transitions
- Artifacts at concatenation points between files

## Files Modified

1. **`blinking_split_analysis.py`**:
   - Added `MANUAL_BLINK_LABELS_CSV` config
   - Added `file_frame_mapping` dictionary
   - Added `load_manual_blink_labels_and_add_to_gt()` function with edge filtering
   - Updated manual GT peaks loading to combine both sources
   - Tracks `successfully_loaded_tuples` for accurate file mapping

2. **`MANUAL_BLINK_LABELS_INTEGRATION.md`** (new):
   - Comprehensive documentation of the integration
   - Usage examples
   - Troubleshooting guide

## Testing

To test the changes:

1. Make sure you have `manual_blink_labels.csv` in the working directory
2. Run the analysis:
   ```bash
   python blinking_split_analysis.py
   ```
3. Check the output for the "LOADING MANUALLY EDITED GT PEAKS AND MANUAL BLINK LABELS" section
4. Verify that peaks from the CSV are being loaded and combined

## Configuration

To disable automatic loading of manual blink labels:
```python
USE_MANUAL_GT_PEAKS = False
```

To change the edge margin (default is 10 frames):
```python
# In the function call, change edge_margin parameter
manual_blink_peaks = load_manual_blink_labels_and_add_to_gt(
    MANUAL_BLINK_LABELS_CSV, 
    file_frame_mapping,
    edge_margin=10  # Change this value
)
```

