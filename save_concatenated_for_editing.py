"""
Helper script to save concatenated GT and prediction signals for editing.
Run this after your blinking_split_analysis.py to prepare data for the peak editor.
"""
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

def save_for_peak_editor(gt_concat, pred_concat, matches=None, 
                         output_file=None, use_timestamp=True,
                         file_order=None, file_offsets=None):
    """
    Save concatenated signals and optionally detected peaks for the peak editor.
    
    Parameters:
    -----------
    gt_concat : np.ndarray
        Concatenated ground truth blink signal
    pred_concat : np.ndarray
        Concatenated predicted blink signal
    matches : dict, optional
        Dictionary with detected peaks and matches (from visualization.py)
    output_file : str, optional
        Output file path. If None, generates timestamped filename.
    use_timestamp : bool
        If True, adds timestamp to filename to avoid overwriting (default: True)
    file_order : list, optional
        List of (run_path, tar_id, side) tuples in the order files were concatenated
    file_offsets : list, optional
        List of frame offsets for each file in the concatenated signal
    
    Example:
    --------
    # After running your blink analysis:
    from save_concatenated_for_editing import save_for_peak_editor
    
    # From your analysis results:
    blink_analysis = analyzer.analyze_blinks(...)
    gt_concat = blink_analysis['gt_concat']
    pred_concat = blink_analysis['pred_concat']
    matches = blink_analysis['matches']
    
    # Save for editing (with timestamp to avoid overwriting):
    save_for_peak_editor(gt_concat, pred_concat, matches)
    """
    # Generate filename with timestamp if needed
    if output_file is None or use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file is None:
            output_file = f"concatenated_signals_{timestamp}.pkl"
        else:
            # Insert timestamp before .pkl extension
            base = Path(output_file).stem
            ext = Path(output_file).suffix
            output_file = f"{base}_{timestamp}{ext}"
    
    data = {
        'gt_concat': gt_concat,
        'pred_concat': pred_concat,
        'num_frames': len(gt_concat),
        'duration_seconds': len(gt_concat) / 30.0,
        'sample_rate': 30.0,
        'timestamp': datetime.now().isoformat()
    }
    
    if matches is not None:
        data['matches'] = matches
        print(f"Including detected peaks:")
        print(f"  GT peaks: {len(matches.get('gt_peaks', []))}")
        print(f"  Pred peaks: {len(matches.get('pred_peaks', []))}")
        print(f"  Matched GT: {len(matches.get('matched_gt', []))}")
        print(f"  Unmatched GT: {len(matches.get('unmatched_gt', []))}")
        print(f"  Unmatched Pred: {len(matches.get('unmatched_pred', []))}")
    
    if file_order is not None and file_offsets is not None:
        data['file_order'] = file_order
        data['file_offsets'] = file_offsets
        print(f"Including file order information:")
        print(f"  Number of files: {len(file_order)}")
        print(f"  Total frames: {len(gt_concat)}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Saved concatenated signals to: {output_file}")
    print(f"  Frames: {data['num_frames']}")
    print(f"  Duration: {data['duration_seconds']:.1f} seconds")
    print(f"  Timestamp: {data['timestamp']}")
    print(f"\nNow run:")
    print(f"  python launch_peak_editor_from_analysis.py {output_file}")
    
    return output_file

def example_usage():
    """
    Example showing how to integrate this into your analysis workflow.
    """
    print("""
# Add this to your blinking_split_analysis.py or run it separately:

from blink_analyzer import BlinkAnalyzer
from save_concatenated_for_editing import save_for_peak_editor

# Your existing analysis code...
analyzer = BlinkAnalyzer()
blink_analysis = analyzer.analyze_blinks(
    blendshapes_list=blendshapes_list,
    pred_blends_list=pred_blends_list,
    gt_th=0.5,
    model_th=0.5,
    max_offset=10
)

# Extract concatenated signals
gt_concat = blink_analysis['gt_concat']
pred_concat = blink_analysis['pred_concat']
matches = blink_analysis['matches']

# Save for peak editor
save_for_peak_editor(gt_concat, pred_concat, matches)

# Now you can run:
# python launch_peak_editor_from_analysis.py concatenated_signals.pkl
""")

if __name__ == '__main__':
    example_usage()

