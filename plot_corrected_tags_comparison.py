#!/usr/bin/env python3
"""
Plot comparison of original GT tags vs corrected tags based on manual labels.

Shows concatenated examples in the same order as blinking_split_analysis.py with:
- Top subplot: Original GT peaks
- Bottom subplot: Corrected GT peaks based on manual labels
  - blink: add peak to GT at frame (25fps)
  - no_blink: do nothing
  - dont_know: remove peak from pred
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import zscore
import sys
import plotly.io as pio
from tqdm import tqdm
pio.renderers.default = "browser"

# Configuration
LABELS_CSV = "manual_blink_labels.csv"
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")

# Split dataframe configuration (same as blinking_split_analysis.py)
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'
NUM_RANDOM_SAMPLES = 1000
RANDOM_SEED = 42

MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1

# Peak detection parameters
PROMINENCE_PRED = 0.5
HEIGHT_THRESHOLD = 0.0
DISTANCE = 5
MAX_OFFSET = 10

# Import data loading functions
sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes

# Number of examples to plot (set to None to plot all files)
NUM_EXAMPLES = 20


def load_pred_and_gt_data(run_path, tar_id):
    """Load prediction and GT blendshapes for a given sample."""
    # Load GT blendshapes
    full_run_path = ALL_DATA_PATH / run_path
    try:
        gt_blendshapes = load_ground_truth_blendshapes(full_run_path, downsample=True)
    except Exception as e:
        print(f"Error loading GT for {run_path}: {e}")
        return None, None
    
    # Load prediction
    pred_file = INFERENCE_OUTPUTS_PATH / run_path / f"pred_{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}_{tar_id}.npy"
    
    if not pred_file.exists():
        print(f"Prediction file not found: {pred_file}")
        return None, None
    
    pred_blendshapes = np.load(pred_file)
    
    # Align lengths
    gt_len = gt_blendshapes.shape[0]
    pred_len = pred_blendshapes.shape[0]
    
    if pred_len < gt_len:
        padding = np.zeros((gt_len - pred_len, pred_blendshapes.shape[1]))
        pred_blendshapes = np.concatenate([pred_blendshapes, padding], axis=0)
    elif pred_len > gt_len:
        pred_blendshapes = pred_blendshapes[:gt_len]
    
    return gt_blendshapes, pred_blendshapes


def detect_all_peaks(gt_bs, pred_bs):
    """Detect all peaks in GT and prediction signals."""
    gt_blink = gt_bs[:, 10]  # eyeBlinkRight
    pred_blink = pred_bs[:, 10]
    
    # Apply smoothing and z-score
    gt_smooth = savgol_filter(gt_blink, 9, 2, mode='interp')
    pred_smooth = savgol_filter(pred_blink, 9, 2, mode='interp')
    
    gt_zscore = zscore(gt_smooth)
    pred_zscore = zscore(pred_smooth)
    
    # Detect peaks
    gt_peaks, _ = find_peaks(gt_zscore, height=HEIGHT_THRESHOLD, prominence=1.0, distance=DISTANCE)
    pred_peaks, _ = find_peaks(pred_zscore, height=HEIGHT_THRESHOLD, prominence=PROMINENCE_PRED, distance=DISTANCE)
    
    # Match peaks
    matched_gt = []
    matched_pred = []
    unmatched_gt = []
    unmatched_pred = list(pred_peaks)
    
    for gt_peak in gt_peaks:
        distances = np.abs(pred_peaks - gt_peak)
        close_peaks = np.where(distances <= MAX_OFFSET)[0]
        
        if len(close_peaks) > 0:
            closest_idx = close_peaks[np.argmin(distances[close_peaks])]
            closest_pred_peak = pred_peaks[closest_idx]
            
            matched_gt.append(gt_peak)
            matched_pred.append(closest_pred_peak)
            
            if closest_pred_peak in unmatched_pred:
                unmatched_pred.remove(closest_pred_peak)
        else:
            unmatched_gt.append(gt_peak)
    
    return {
        'gt_smooth': gt_smooth,
        'pred_smooth': pred_smooth,
        'gt_zscore': gt_zscore,
        'pred_zscore': pred_zscore,
        'gt_peaks': np.array(gt_peaks),
        'pred_peaks': np.array(pred_peaks),
        'matched_gt': np.array(matched_gt),
        'matched_pred': np.array(matched_pred),
        'unmatched_gt': np.array(unmatched_gt),
        'unmatched_pred': np.array(unmatched_pred)
    }


def apply_corrections(peak_data, corrections, gt_smooth):
    """Apply manual corrections to create corrected GT and pred peaks.
    
    Args:
        peak_data: Original peak detection results
        corrections: List of (frame_25fps, label) tuples
        gt_smooth: Original GT smooth signal for adding new peaks
    
    Returns:
        dict with corrected_gt_peaks and corrected_pred_peaks
    """
    # Start with original peaks
    corrected_gt_peaks = list(peak_data['gt_peaks'])
    corrected_pred_peaks = list(peak_data['pred_peaks'])
    
    # Find pred peaks within ±2 frames of each correction
    def find_nearby_pred_peak(frame, pred_peaks, tolerance=2):
        """Find pred peak within tolerance of frame."""
        for p in pred_peaks:
            if abs(p - frame) <= tolerance:
                return p
        return None
    
    for frame_25fps, label in corrections:
        if label == 'blink':
            # Add peak to GT at this frame (25fps) - model found a real blink GT missed
            if frame_25fps not in corrected_gt_peaks:
                corrected_gt_peaks.append(frame_25fps)
        elif label == 'no_blink':
            # Remove pred peak - this is a false positive
            nearby_peak = find_nearby_pred_peak(frame_25fps, corrected_pred_peaks)
            if nearby_peak is not None and nearby_peak in corrected_pred_peaks:
                corrected_pred_peaks.remove(nearby_peak)
        elif label == 'dont_know':
            # Remove peak from pred - uncertain, exclude from analysis
            nearby_peak = find_nearby_pred_peak(frame_25fps, corrected_pred_peaks)
            if nearby_peak is not None and nearby_peak in corrected_pred_peaks:
                corrected_pred_peaks.remove(nearby_peak)
    
    # Sort and convert to arrays
    corrected_gt_peaks = np.array(sorted(corrected_gt_peaks))
    corrected_pred_peaks = np.array(sorted(corrected_pred_peaks))
    
    # Re-match peaks with corrected GT
    matched_gt = []
    matched_pred = []
    unmatched_gt = []
    unmatched_pred = list(corrected_pred_peaks)
    
    for gt_peak in corrected_gt_peaks:
        distances = np.abs(corrected_pred_peaks - gt_peak)
        close_peaks = np.where(distances <= MAX_OFFSET)[0]
        
        if len(close_peaks) > 0:
            closest_idx = close_peaks[np.argmin(distances[close_peaks])]
            closest_pred_peak = corrected_pred_peaks[closest_idx]
            
            matched_gt.append(gt_peak)
            matched_pred.append(closest_pred_peak)
            
            if closest_pred_peak in unmatched_pred:
                unmatched_pred.remove(closest_pred_peak)
        else:
            unmatched_gt.append(gt_peak)
    
    return {
        'corrected_gt_peaks': corrected_gt_peaks,
        'corrected_pred_peaks': corrected_pred_peaks,
        'corrected_matched_gt': np.array(matched_gt),
        'corrected_matched_pred': np.array(matched_pred),
        'corrected_unmatched_gt': np.array(unmatched_gt),
        'corrected_unmatched_pred': np.array(unmatched_pred)
    }


def get_run_paths_from_split(df_path: str, num_random_samples: int = None, random_seed: int = None):
    """Load run_paths, tar_ids, and side from split dataframe (same as blinking_split_analysis.py)."""
    print(f"Loading dataframe from {df_path}")
    df = pd.read_pickle(df_path)
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Random sampling
    if num_random_samples is not None and num_random_samples < len(df):
        if random_seed is not None:
            np.random.seed(random_seed)
        random_indices = np.random.choice(len(df), size=num_random_samples, replace=False)
        rows_to_process = df.iloc[random_indices]
        print(f"Randomly sampled {num_random_samples} rows (seed={random_seed})")
    else:
        rows_to_process = df
        print(f"Processing all {len(rows_to_process)} rows")
    
    # Extract run_path, tar_id, and side tuples
    run_path_tar_id_side_tuples = []
    for idx, row in rows_to_process.iterrows():
        run_path_tar_id_side_tuples.append((row.run_path, row.tar_id, row.side))
    
    return run_path_tar_id_side_tuples


def main():
    import pickle
    
    # Load labels
    if not Path(LABELS_CSV).exists():
        print(f"Labels file not found: {LABELS_CSV}")
        return
    
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(labels_df)} manual labels")
    
    # Load manually edited GT peaks from peak editor
    print("\n" + "="*80)
    print("LOADING MANUALLY EDITED GT PEAKS FROM PEAK EDITOR")
    print("="*80)
    manual_gt_peaks_dict = {}
    
    # Try to load from manual_gt_peaks_from_labels.pkl (per-file format from blinking_split_analysis.py)
    if Path("manual_gt_peaks_from_labels.pkl").exists():
        with open("manual_gt_peaks_from_labels.pkl", 'rb') as f:
            manual_gt_peaks_dict = pickle.load(f)
        print(f"✓ Loaded {len(manual_gt_peaks_dict)} files with manually edited GT peaks from manual_gt_peaks_from_labels.pkl")
    elif Path("final_gt_peaks.pkl").exists():
        with open("final_gt_peaks.pkl", 'rb') as f:
            manual_peaks_data = pickle.load(f)
        
        # Check if it's concatenated format (array) or per-file format (dict)
        if isinstance(manual_peaks_data, dict):
            manual_gt_peaks_dict = manual_peaks_data
            print(f"✓ Loaded {len(manual_gt_peaks_dict)} files with manually edited GT peaks from final_gt_peaks.pkl")
        else:
            print(f"✓ Loaded {len(manual_peaks_data)} manually edited GT peaks (concatenated format) from final_gt_peaks.pkl")
            print(f"  ⚠ Cannot use concatenated format without file offsets - will use automatic GT peak detection")
    else:
        print("⚠ No manual GT peaks files found - will use automatic GT peak detection")
    
    # Get files in the same order as blinking_split_analysis.py
    print("\n" + "="*80)
    print("LOADING FILES IN SAME ORDER AS BLINKING_SPLIT_ANALYSIS.PY")
    print("="*80)
    run_path_tar_id_side_tuples = get_run_paths_from_split(
        SPLIT_DF_PATH,
        num_random_samples=NUM_RANDOM_SAMPLES,
        random_seed=RANDOM_SEED
    )
    
    # Limit to NUM_EXAMPLES if specified
    if NUM_EXAMPLES is not None:
        selected_files = run_path_tar_id_side_tuples[:NUM_EXAMPLES]
        print(f"Plotting first {NUM_EXAMPLES} files from the split")
    else:
        selected_files = run_path_tar_id_side_tuples
        print(f"Plotting all {len(selected_files)} files from the split")
    
    print(f"Total files to plot: {len(selected_files)}")
    
    # Collect data for all examples
    all_data = []
    
    for run_path, tar_id, side in tqdm(selected_files, desc="Loading files"):
        # Get all labels for this file
        file_labels = labels_df[
            (labels_df['run_path'] == run_path) &
            (labels_df['tar_id'] == tar_id) &
            (labels_df['side'] == side)
        ]
        
        # Load data
        gt_bs, pred_bs = load_pred_and_gt_data(run_path, tar_id)
        
        if gt_bs is None:
            print(f"Skipping {run_path}/{tar_id}/{side} - could not load data")
            continue
        
        # Detect original peaks (automatic detection)
        peak_data = detect_all_peaks(gt_bs, pred_bs)
        
        # Check if we have manually edited GT peaks for this file
        file_key = (run_path, tar_id, side)
        if file_key in manual_gt_peaks_dict:
            # Use manually edited GT peaks instead of automatic detection
            manual_gt_peaks = manual_gt_peaks_dict[file_key]
            peak_data['gt_peaks'] = manual_gt_peaks
            
            # Re-match with manual GT peaks
            matched_gt = []
            matched_pred = []
            unmatched_gt = []
            unmatched_pred = list(peak_data['pred_peaks'])
            
            for gt_peak in manual_gt_peaks:
                distances = np.abs(peak_data['pred_peaks'] - gt_peak)
                close_peaks = np.where(distances <= MAX_OFFSET)[0]
                
                if len(close_peaks) > 0:
                    closest_idx = close_peaks[np.argmin(distances[close_peaks])]
                    closest_pred_peak = peak_data['pred_peaks'][closest_idx]
                    
                    matched_gt.append(gt_peak)
                    matched_pred.append(closest_pred_peak)
                    
                    if closest_pred_peak in unmatched_pred:
                        unmatched_pred.remove(closest_pred_peak)
                else:
                    unmatched_gt.append(gt_peak)
            
            peak_data['matched_gt'] = np.array(matched_gt)
            peak_data['matched_pred'] = np.array(matched_pred)
            peak_data['unmatched_gt'] = np.array(unmatched_gt)
            peak_data['unmatched_pred'] = np.array(unmatched_pred)
        
        # Collect corrections for this file
        corrections = []
        for _, row in file_labels.iterrows():
            corrections.append((row['peak_frame_25fps'], row['label']))
        
        # Apply corrections from labels CSV on top of manual peaks
        corrected_data = apply_corrections(peak_data, corrections, peak_data['gt_smooth'])
        
        all_data.append({
            'run_path': run_path,
            'tar_id': tar_id,
            'side': side,
            'peak_data': peak_data,
            'corrected_data': corrected_data,
            'corrections': corrections,
            'num_labels': len(corrections)
        })
    
    if len(all_data) == 0:
        print("No data to plot!")
        return
    
    print(f"Successfully loaded {len(all_data)} examples")
    
    # Create concatenated plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Original GT Tags",
            "Corrected GT Tags (based on manual labels)"
        ),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5]
    )
    
    # Concatenate signals
    current_offset = 0
    file_boundaries = [0]
    
    for i, data in enumerate(all_data):
        peak_data = data['peak_data']
        corrected_data = data['corrected_data']
        
        signal_length = len(peak_data['pred_zscore'])
        frames = np.arange(signal_length) + current_offset
        
        # Plot 1: Original tags
        # Prediction signal
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=peak_data['pred_zscore'],
                mode='lines',
                name='Prediction' if i == 0 else None,
                line=dict(color='red', width=1),
                showlegend=(i == 0),
                legendgroup='pred'
            ),
            row=1, col=1
        )
        
        # Original GT peaks
        if len(peak_data['gt_peaks']) > 0:
            gt_peak_frames = peak_data['gt_peaks'] + current_offset
            gt_peak_values = peak_data['pred_zscore'][peak_data['gt_peaks']]
            fig.add_trace(
                go.Scatter(
                    x=gt_peak_frames,
                    y=gt_peak_values,
                    mode='markers',
                    name='Original GT peaks' if i == 0 else None,
                    marker=dict(color='blue', size=8, symbol='diamond'),
                    showlegend=(i == 0),
                    legendgroup='orig_gt'
                ),
                row=1, col=1
            )
        
        # Original matched pred peaks
        if len(peak_data['matched_pred']) > 0:
            matched_frames = peak_data['matched_pred'] + current_offset
            matched_values = peak_data['pred_zscore'][peak_data['matched_pred']]
            fig.add_trace(
                go.Scatter(
                    x=matched_frames,
                    y=matched_values,
                    mode='markers',
                    name='Matched pred' if i == 0 else None,
                    marker=dict(color='green', size=8, symbol='circle'),
                    showlegend=(i == 0),
                    legendgroup='matched'
                ),
                row=1, col=1
            )
        
        # Original unmatched pred peaks
        if len(peak_data['unmatched_pred']) > 0:
            unmatched_frames = peak_data['unmatched_pred'] + current_offset
            unmatched_values = peak_data['pred_zscore'][peak_data['unmatched_pred']]
            fig.add_trace(
                go.Scatter(
                    x=unmatched_frames,
                    y=unmatched_values,
                    mode='markers',
                    name='Unmatched pred' if i == 0 else None,
                    marker=dict(color='orange', size=8, symbol='circle'),
                    showlegend=(i == 0),
                    legendgroup='unmatched'
                ),
                row=1, col=1
            )
        
        # Plot 2: Corrected tags
        # Prediction signal
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=peak_data['pred_zscore'],
                mode='lines',
                name='Prediction' if i == 0 else None,
                line=dict(color='red', width=1),
                showlegend=False,
                legendgroup='pred2'
            ),
            row=2, col=1
        )
        
        # Corrected GT peaks
        if len(corrected_data['corrected_gt_peaks']) > 0:
            corrected_gt_frames = corrected_data['corrected_gt_peaks'] + current_offset
            corrected_gt_values = peak_data['pred_zscore'][corrected_data['corrected_gt_peaks']]
            fig.add_trace(
                go.Scatter(
                    x=corrected_gt_frames,
                    y=corrected_gt_values,
                    mode='markers',
                    name='Corrected GT peaks' if i == 0 else None,
                    marker=dict(color='purple', size=8, symbol='diamond'),
                    showlegend=(i == 0),
                    legendgroup='corr_gt'
                ),
                row=2, col=1
            )
        
        # Corrected matched pred peaks
        if len(corrected_data['corrected_matched_pred']) > 0:
            corr_matched_frames = corrected_data['corrected_matched_pred'] + current_offset
            corr_matched_values = peak_data['pred_zscore'][corrected_data['corrected_matched_pred']]
            fig.add_trace(
                go.Scatter(
                    x=corr_matched_frames,
                    y=corr_matched_values,
                    mode='markers',
                    name='Corrected matched' if i == 0 else None,
                    marker=dict(color='green', size=8, symbol='circle'),
                    showlegend=(i == 0),
                    legendgroup='corr_matched'
                ),
                row=2, col=1
            )
        
        # Corrected unmatched pred peaks
        if len(corrected_data['corrected_unmatched_pred']) > 0:
            corr_unmatched_frames = corrected_data['corrected_unmatched_pred'] + current_offset
            corr_unmatched_values = peak_data['pred_zscore'][corrected_data['corrected_unmatched_pred']]
            fig.add_trace(
                go.Scatter(
                    x=corr_unmatched_frames,
                    y=corr_unmatched_values,
                    mode='markers',
                    name='Corrected unmatched' if i == 0 else None,
                    marker=dict(color='orange', size=8, symbol='circle'),
                    showlegend=(i == 0),
                    legendgroup='corr_unmatched'
                ),
                row=2, col=1
            )
        
        # Highlight manual corrections
        for frame_25fps, label in data['corrections']:
            correction_frame = frame_25fps + current_offset
            
            # Add vertical line for each correction
            color_map = {
                'blink': 'lightgreen',
                'no_blink': 'lightcoral',
                'dont_know': 'lightyellow'
            }
            
            fig.add_vline(
                x=correction_frame,
                line_dash="dot",
                line_color=color_map.get(label, 'gray'),
                line_width=1,
                opacity=0.5,
                row=2, col=1
            )
        
        # Add file boundary
        current_offset += signal_length
        file_boundaries.append(current_offset)
        
        # Add vertical separator between files
        if i < len(all_data) - 1:
            fig.add_vline(
                x=current_offset,
                line_dash="solid",
                line_color="black",
                line_width=2,
                row=1, col=1
            )
            fig.add_vline(
                x=current_offset,
                line_dash="solid",
                line_color="black",
                line_width=2,
                row=2, col=1
            )
    
    # Update layout
    fig.update_xaxes(title_text="Frame (25 fps)", row=2, col=1)
    fig.update_yaxes(title_text="Z-Score", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
    # Calculate statistics
    total_corrections = sum(d['num_labels'] for d in all_data)
    blink_count = sum(len([c for c in d['corrections'] if c[1] == 'blink']) for d in all_data)
    no_blink_count = sum(len([c for c in d['corrections'] if c[1] == 'no_blink']) for d in all_data)
    dont_know_count = sum(len([c for c in d['corrections'] if c[1] == 'dont_know']) for d in all_data)
    
    fig.update_layout(
        title=f"Original vs Corrected GT Tags - {len(all_data)} Files Concatenated<br>" +
              f"<sub>Total corrections: {total_corrections} (blink: {blink_count}, no_blink: {no_blink_count}, dont_know: {dont_know_count})</sub>",
        hovermode='x unified',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Show plot in browser
    fig.show()
    
    # Also save to file
    output_file = "corrected_tags_comparison.html"
    fig.write_html(output_file)
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for i, data in enumerate(all_data):
        print(f"\nFile {i+1}: {data['run_path']}/{data['tar_id']}/{data['side']}")
        print(f"  Manual corrections: {data['num_labels']}")
        print(f"  Original: {len(data['peak_data']['gt_peaks'])} GT peaks, "
              f"{len(data['peak_data']['matched_pred'])} matched, "
              f"{len(data['peak_data']['unmatched_pred'])} unmatched")
        print(f"  Corrected: {len(data['corrected_data']['corrected_gt_peaks'])} GT peaks, "
              f"{len(data['corrected_data']['corrected_matched_pred'])} matched, "
              f"{len(data['corrected_data']['corrected_unmatched_pred'])} unmatched")
    
    print("\n" + "="*80)
    print(f"Total files: {len(all_data)}")
    print(f"Total corrections: {total_corrections}")
    print(f"  - blink (add GT peak): {blink_count}")
    print(f"  - no_blink (keep as is): {no_blink_count}")
    print(f"  - dont_know (remove pred peak): {dont_know_count}")
    print("="*80)


if __name__ == '__main__':
    main()

