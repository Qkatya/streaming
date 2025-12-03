"""
Helper script to launch the peak editor app with data from blinking_split_analysis.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import from existing modules
from blinking_split_analysis import (
    ALL_DATA_PATH, INFERENCE_OUTPUTS_PATH, SPLIT_DF_PATH,
    MODEL_NAME, HISTORY_SIZE, LOOKAHEAD_SIZE,
    load_predictions, load_gt_blendshapes
)
from blink_analyzer import BlinkAnalyzer
from peak_editor_app import create_app

def extract_blinks(data: np.ndarray) -> np.ndarray:
    """Extract blink signal from blendshape data."""
    if data.shape[1] == 51:
        return (data[:, 8] + data[:, 9]) / 2  # Combine left and right blink
    return (data[:, 9] + data[:, 10]) / 2  # Combine left and right blink

def load_data_for_editor(row_idx=None, split_df_path=None):
    """
    Load GT and prediction data for the peak editor.
    
    Parameters:
    -----------
    row_idx : int, optional
        Row index from the split dataframe. If None, will prompt user.
    split_df_path : str, optional
        Path to split dataframe. If None, uses default from config.
    
    Returns:
    --------
    gt_signal : np.ndarray
        Ground truth blink signal
    pred_signal : np.ndarray
        Predicted blink signal
    metadata : dict
        Additional metadata about the recording
    """
    # Load split dataframe
    if split_df_path is None:
        split_df_path = SPLIT_DF_PATH
    
    print(f"Loading split dataframe from: {split_df_path}")
    split_df = pd.read_pickle(split_df_path)
    print(f"Split dataframe shape: {split_df.shape}")
    
    # Select row
    if row_idx is None:
        print("\nAvailable recordings:")
        print(split_df[['run_path', 'tar_id']].head(20))
        row_idx = int(input("\nEnter row index to edit: "))
    
    row = split_df.iloc[row_idx]
    run_path = row['run_path']
    tar_id = row['tar_id']
    
    print(f"\nLoading data for:")
    print(f"  Row index: {row_idx}")
    print(f"  Run path: {run_path}")
    print(f"  Target ID: {tar_id}")
    
    # Load ground truth
    print("\nLoading ground truth blendshapes...")
    gt_blendshapes = load_gt_blendshapes(run_path)
    if gt_blendshapes is None:
        raise ValueError("Failed to load ground truth blendshapes")
    
    # Load predictions
    print("Loading predictions...")
    pred_blendshapes = load_predictions(run_path, tar_id, MODEL_NAME, HISTORY_SIZE, LOOKAHEAD_SIZE)
    if pred_blendshapes is None:
        raise ValueError("Failed to load predictions")
    
    # Extract blink signals
    print("Extracting blink signals...")
    gt_blinks = extract_blinks(gt_blendshapes)
    pred_blinks = extract_blinks(pred_blendshapes)
    
    # Align lengths
    min_len = min(len(gt_blinks), len(pred_blinks))
    gt_blinks = gt_blinks[:min_len]
    pred_blinks = pred_blinks[:min_len]
    
    print(f"\nData loaded successfully!")
    print(f"  GT signal length: {len(gt_blinks)} frames ({len(gt_blinks)/30:.1f} seconds)")
    print(f"  Pred signal length: {len(pred_blinks)} frames ({len(pred_blinks)/30:.1f} seconds)")
    
    metadata = {
        'row_idx': row_idx,
        'run_path': run_path,
        'tar_id': tar_id,
        'duration_seconds': len(gt_blinks) / 30.0,
        'num_frames': len(gt_blinks)
    }
    
    return gt_blinks, pred_blinks, metadata

def main():
    """Main entry point."""
    print("=" * 80)
    print("Ground Truth Peak Editor - Data Loader")
    print("=" * 80)
    
    # Parse command line arguments
    row_idx = None
    if len(sys.argv) > 1:
        try:
            row_idx = int(sys.argv[1])
            print(f"\nUsing row index from command line: {row_idx}")
        except ValueError:
            print(f"\nInvalid row index: {sys.argv[1]}")
            sys.exit(1)
    
    try:
        # Load data
        gt_signal, pred_signal, metadata = load_data_for_editor(row_idx)
        
        # Print metadata
        print("\n" + "=" * 80)
        print("Recording Metadata:")
        print("=" * 80)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Create and launch app
        print("\n" + "=" * 80)
        print("Launching Peak Editor App...")
        print("=" * 80)
        print("\nThe app will open in your browser at: http://127.0.0.1:8050")
        print("\nInstructions:")
        print("  - Use the Previous/Next buttons to navigate between 50-second windows")
        print("  - Click on the graph to add or remove GT peaks")
        print("  - Click near an existing peak (green/orange circles) to remove it")
        print("  - Click elsewhere to add a new peak")
        print("  - Click 'Save Changes' to save your modifications")
        print("\nModifications are saved to:")
        print(f"  - removed_peaks.pkl (peaks you removed)")
        print(f"  - added_peaks.pkl (peaks you added)")
        print(f"  - final_gt_peaks.pkl (final GT peaks after modifications)")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 80)
        
        # Create and run app
        app = create_app(gt_signal, pred_signal, sample_rate=30.0)
        app.run_server(debug=True, port=8050)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()


