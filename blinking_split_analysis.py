import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"
from blink_analyzer import BlinkAnalyzer
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from scipy.stats import zscore

# Import lightweight data utilities (no ML dependencies - fast import!)
from blendshapes_data_utils import load_ground_truth_blendshapes, BLENDSHAPES_ORDERED

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Paths
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")

# Split dataframe configuration
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'
# SPLIT_DF_PATH = '/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250402_split_1/LOUD_GIP_general_clean_250415_v2.pkl'

# Row indices to process from the split (same as in streaming_split_inference.py)
ROW_INDICES = None
# ROW_INDICES = [3663, 3953]  # Will be randomly sampled from the dataframe in main()
# ROW_INDICES = [3663]  # Will be randomly sampled from the dataframe in main()
# ROW_INDICES = [3663, 3953, 3642, 317, 6117, 1710, 3252, 1820, 3847]
# ROW_INDICES = list(range(0, 4))#[3663, 3953, 3642, 317, 6117, 1710, 3252, 1820, 3847]

# Random sampling configuration
NUM_RANDOM_SAMPLES = 1000# None  # Set to None to process all rows, or a number to randomly sample
RANDOM_SEED = 42  # Set seed for reproducibility (None for different samples each run)
THRESHOLD_FACTOR = 0.25

# Manual GT peaks configuration
USE_MANUAL_GT_PEAKS = True  # Set to True to use manually edited GT peaks from final_gt_peaks.pkl
MANUAL_GT_PEAKS_FILE = "final_gt_peaks.pkl"  # Path to manually edited GT peaks
MANUAL_BLINK_LABELS_CSV = "manual_blink_labels.csv"  # Path to manual blink labels CSV
EDGE_MARGIN = 10  # Number of frames from beginning/end of recordings to exclude from peak detection
# Model configuration - which prediction file to load
MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
MODEL2_NAME = 'causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side'
# MODEL2_NAME = 'new21_baseline_blendshapes_normalized'

MODEL_NAME_TO_PLOT = 'nemo_with_smile'
MODEL2_NAME_TO_PLOT = 'old_fairseq'
# MODEL_NAME = 'causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1

BLENDSHAPES_ORDERED = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
                       'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                       'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
                       'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
                       'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
PROMINANT_MOUTH_BLENDSHAPES = ['mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthSmileRight', 'mouthFrownRight']
PROMINANT_EYE_BLENDSHAPES = ['eyeBlinkRight', 'eyeLookInRight', 'eyeLookOutRight']
PROMINANT_JAW_AND_CHEEKS_BLENDSHAPES = ['jawRight', 'jawOpen', 'cheekPuff']
# CHEEKS_BLENDSHAPES = ['cheekPuff', 'cheekSquintLeft' ,'cheekSquintRight']
BROW_BLENDSHAPES = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']

# Blendshapes to plot
BLENDSHAPES_TO_PLOT = BLENDSHAPES_ORDERED
# BLENDSHAPES_TO_PLOT = ['eyeBlinkRight', 'jawOpen', 'mouthFunnel', 'cheekPuff', 'mouthSmileRight']

# Analysis flags - control which metrics to calculate
CALCULATE_BLINK_METRICS = True  # ROC curve analysis for blink detection
CALCULATE_CORRELATION_METRICS = False #True  # PCC, L1, L2 on raw predictions
CALCULATE_DIFF_METRICS = False #True  # PCC, L1, L2 on frame-to-frame differences

# Plotting mode flags
PLOT_BY_CATEGORIES = True#False  # If True, plot by predefined categories; If False, plot all blendshapes above threshold
VELOCITY_AGREEMENT_THRESHOLD = 80.0  # Minimum velocity agreement % to include in the plot (when PLOT_BY_CATEGORIES=False)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_blendshapes_category_name(blendshapes_list=None):
    """Get a human-readable name for the given blendshapes category.
    
    Args:
        blendshapes_list: List of blendshapes to get name for. If None, uses BLENDSHAPES_TO_PLOT.
    """
    if blendshapes_list is None:
        blendshapes_list = BLENDSHAPES_TO_PLOT
    
    if blendshapes_list == PROMINANT_MOUTH_BLENDSHAPES:
        return "Prominent Mouth Blendshapes"
    elif blendshapes_list == PROMINANT_EYE_BLENDSHAPES:
        return "Prominent Eye Blendshapes"
    elif blendshapes_list == PROMINANT_JAW_AND_CHEEKS_BLENDSHAPES:
        return "Prominent Jaw and Cheeks Blendshapes"
    elif blendshapes_list == BROW_BLENDSHAPES:
        return "Brow Blendshapes"
    else:
        return "Custom Blendshapes"

def plot_each_bs(concatenated_gt, concatenated_pred, ):

    fig = make_subplots(
        rows=len(BLENDSHAPES_TO_PLOT), 
        cols=1, 
        shared_xaxes=True,
        subplot_titles=tuple(f"<span style='font-size:16px'>{bs}</span>" for bs in BLENDSHAPES_TO_PLOT),
        vertical_spacing=0.008
    )

    # Plot each blendshape
    for plot_idx, bs_name in enumerate(BLENDSHAPES_TO_PLOT, 1):
        bs_idx = BLENDSHAPES_ORDERED.index(bs_name)
        
        # Get data for this blendshape
        gt_data = concatenated_gt[:, bs_idx]
        pred_data = concatenated_pred[:, bs_idx]
        
        # Add GT trace
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(gt_data)),
                y=gt_data,
                name='GT',
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=(plot_idx == 1),
                legendgroup='GT'
            ),
            row=plot_idx, col=1
        )
        
        # Add prediction trace
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(pred_data)),
                y=pred_data,
                name='Pred',
                mode='lines',
                line=dict(color='#d62728', width=2),  # Red
                showlegend=(plot_idx == 1),
                legendgroup='Pred'
            ),
            row=plot_idx, col=1
        )

    # Update layout
    category_name = get_blendshapes_category_name()
    fig.update_layout(
        title_text=f"Blendshapes Comparison: GT vs Prediction (Right Side Only)<br>"
                   f"<sub>{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}</sub><br>",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=20,
        height=300 * len(BLENDSHAPES_TO_PLOT),
        hovermode='x unified',
        legend=dict(
            font=dict(size=9),
            yanchor="top",
            y=1,
            xanchor="right",
            x=-0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=100)
    )

    fig.update_xaxes(title_text="Frame", row=len(BLENDSHAPES_TO_PLOT), col=1)
    fig.show()

def plot_one_bs(gt, pred, ):

    fig = make_subplots(
        rows=1, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.008
    )

    # Plot each blendshape
        
    # Get data for this blendshape
    gt_data = gt
    pred_data = pred
    
    # Add GT trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(gt_data)),
            y=gt_data,
            name='GT',
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=True,
            legendgroup='GT'
        ),
        row=1, col=1
    )
    
    # Add prediction trace
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(pred_data)),
            y=pred_data,
            name='Pred',
            mode='lines',
            line=dict(color='#d62728', width=2),  # Red
            showlegend=True,
            legendgroup='Pred'
        ),
        row=1, col=1
    )

    # Update layout
    # category_name = get_blendshapes_category_name()
    fig.update_layout(
        # title_text=f"Blendshapes Comparison: GT vs Prediction (Right Side Only)<br>"
        #            f"<sub>{category_name}</sub><br>"
        #            f"<sub>{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}</sub><br>",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=20,
        height=300,
        hovermode='x unified',
        legend=dict(
            font=dict(size=9),
            yanchor="top",
            y=1,
            xanchor="right",
            x=-0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=100)
    )

    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.show()
    
def get_run_paths_from_split(df_path: str, row_indices: list = None, side_filter: str = None, 
                            num_random_samples: int = None, random_seed: int = None):
    """Load run_paths, tar_ids, and side from split dataframe.
    
    Args:
        df_path: Path to the pickle file containing the split dataframe
        row_indices: List of row indices to process. If None, process all rows (or sample randomly if num_random_samples is set).
        side_filter: Filter by side ('left', 'right', or None for all)
        num_random_samples: Number of random samples to select. If None, use all rows (or row_indices if provided).
        random_seed: Random seed for reproducibility. If None, sampling will be different each run.
        
    Returns:
        List of tuples (run_path, tar_id, side)
    """
    print(f"Loading dataframe from {df_path}")
    df = pd.read_pickle(df_path)
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Filter by side first if specified
    if side_filter is not None:
        df = df[df['side'] == side_filter]
        print(f"Filtered to {len(df)} rows with side='{side_filter}'")
    
    # Select rows to process
    if row_indices is None:
        if num_random_samples is not None and num_random_samples < len(df):
            # Random sampling
            if random_seed is not None:
                np.random.seed(random_seed)
            random_indices = np.random.choice(len(df), size=num_random_samples, replace=False)
            rows_to_process = df.iloc[random_indices]
            print(f"Randomly sampled {num_random_samples} rows (seed={random_seed})")
        else:
            rows_to_process = df
            print(f"Processing all {len(rows_to_process)} rows")
    else:
        rows_to_process = df.iloc[row_indices]
        print(f"Processing {len(rows_to_process)} selected rows: {row_indices}")
    
    
    # Extract run_path, tar_id, and side tuples
    run_path_tar_id_side_tuples = []
    for idx, row in rows_to_process.iterrows():
        run_path_tar_id_side_tuples.append((row.run_path, row.tar_id, row.side))
    
    return run_path_tar_id_side_tuples

def load_prediction(run_path: str, tar_id: str, side: str, model_name: str, history: int, lookahead: int):
    """Load prediction file for a given run_path, tar_id, side and model configuration.
    
    Args:
        run_path: Path to the run directory
        tar_id: The tar_id for this specific file
        side: The side ('left' or 'right')
        model_name: Name of the model
        history: History size used
        lookahead: Lookahead size used
        
    Returns:
        Prediction array or None if file not found
    """
    pred_file = INFERENCE_OUTPUTS_PATH / run_path / f"pred_{model_name}_H{history}_LA{lookahead}_{tar_id}.npy"
    
    if not pred_file.exists():
        print(f"Warning: Prediction file not found: {pred_file}")
        return None
    
    return np.load(pred_file)

def load_gt_blendshapes(run_path: str):
    """Load ground truth blendshapes from landmarks_and_blendshapes.npz."""
    full_run_path = ALL_DATA_PATH / run_path
    
    try:
        return load_ground_truth_blendshapes(full_run_path, downsample=True)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None

def calculate_metrics(matched_gt, matched_pred, unmatched_gt, unmatched_pred, pred_concat, quantizer):
    unmatched_gt -=45
    P = matched_gt + unmatched_gt
    N = len(pred_concat) / quantizer - P    
    TP = matched_pred
    FP = unmatched_pred
    
    TPR = (TP / P)*100
    FNR = (100 - TPR)
    FPR = (FP/(pred_concat.shape[0]/25))*60#(FP / N)*100
            
    return TPR, FNR, FPR

def load_dont_know_peaks_to_exclude(csv_filepath, file_frame_mapping):
    """Load peaks labeled as 'dont_know' from CSV to exclude from predictions.
    
    Args:
        csv_filepath: Path to manual_blink_labels.csv
        file_frame_mapping: Dict mapping (run_path, tar_id, side) to (start_frame, end_frame)
    
    Returns:
        Array of global frame positions for 'dont_know' peaks to exclude, or None if not found
    """
    if not Path(csv_filepath).exists():
        print(f"Manual blink labels CSV not found: {csv_filepath}")
        return None
    
    try:
        # Load CSV
        manual_labels_df = pd.read_csv(csv_filepath)
        
        # Filter for dont_know labels only
        dont_know_labels = manual_labels_df[manual_labels_df['label'] == 'dont_know'].copy()
        
        if len(dont_know_labels) == 0:
            return None
        
        print(f"\nFound {len(dont_know_labels)} 'dont_know' labels to exclude from predictions")
        
        # Convert to global frame positions
        exclude_peaks = []
        skipped_count = 0
        
        for idx, row in dont_know_labels.iterrows():
            run_path = row['run_path']
            tar_id = row['tar_id']
            side = row['side']
            peak_frame_25fps = int(row['peak_frame_25fps'])
            
            # Look up the file in our mapping
            key = (run_path, tar_id, side)
            if key not in file_frame_mapping:
                skipped_count += 1
                continue
            
            start_frame, end_frame = file_frame_mapping[key]
            
            # Calculate global frame position
            global_frame = start_frame + peak_frame_25fps
            
            # Validate frame is within bounds
            if global_frame < start_frame or global_frame >= end_frame:
                skipped_count += 1
                continue
            
            exclude_peaks.append(global_frame)
        
        print(f"Processed {len(exclude_peaks)} 'dont_know' peaks to exclude")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} peaks (file not found or out of bounds)")
        
        if len(exclude_peaks) > 0:
            return np.array(exclude_peaks, dtype=np.int64)
        else:
            return None
            
    except Exception as e:
        print(f"Error loading dont_know peaks: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_manual_blink_labels_and_add_to_gt(csv_filepath, file_frame_mapping, edge_margin=10):
    """Load manual blink labels from CSV and convert to global frame positions.
    
    Args:
        csv_filepath: Path to manual_blink_labels.csv
        file_frame_mapping: Dict mapping (run_path, tar_id, side) to (start_frame, end_frame)
        edge_margin: Number of frames from beginning/end of each recording to exclude (default: 10)
    
    Returns:
        Array of global frame positions for manually labeled blinks, or None if file not found
    """
    if not Path(csv_filepath).exists():
        print(f"Manual blink labels CSV not found: {csv_filepath}")
        return None
    
    try:
        # Load CSV
        manual_labels_df = pd.read_csv(csv_filepath)
        print(f"Loaded {len(manual_labels_df)} manual labels from CSV")
        
        # Show label distribution
        label_counts = manual_labels_df['label'].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        # Filter for blink labels only
        blink_labels = manual_labels_df[manual_labels_df['label'] == 'blink'].copy()
        print(f"\nFound {len(blink_labels)} blink labels to add to GT")
        
        if len(blink_labels) == 0:
            return None
        
        # Convert to global frame positions
        new_gt_peaks = []
        skipped_count = 0
        skipped_edge_count = 0
        
        for idx, row in blink_labels.iterrows():
            run_path = row['run_path']
            tar_id = row['tar_id']
            side = row['side']
            peak_frame_25fps = int(row['peak_frame_25fps'])
            
            # Look up the file in our mapping
            key = (run_path, tar_id, side)
            if key not in file_frame_mapping:
                print(f"Warning: File not found in mapping: {key}")
                skipped_count += 1
                continue
            
            start_frame, end_frame = file_frame_mapping[key]
            file_length = end_frame - start_frame
            
            # Filter out peaks within edge_margin frames from beginning or end of recording
            if peak_frame_25fps < edge_margin or peak_frame_25fps >= (file_length - edge_margin):
                skipped_edge_count += 1
                continue
            
            # Calculate global frame position
            global_frame = start_frame + peak_frame_25fps
            
            # Validate frame is within bounds
            if global_frame < start_frame or global_frame >= end_frame:
                print(f"Warning: Peak frame {peak_frame_25fps} out of bounds for file {run_path}")
                print(f"  File frames: [{start_frame}, {end_frame})")
                print(f"  Global frame: {global_frame}")
                skipped_count += 1
                continue
            
            new_gt_peaks.append(global_frame)
        
        print(f"Processed {len(new_gt_peaks)} new GT peaks from manual labels")
        print(f"Skipped {skipped_count} peaks (file not found or out of bounds)")
        print(f"Skipped {skipped_edge_count} peaks (within {edge_margin} frames of recording edges)")
        
        if len(new_gt_peaks) > 0:
            return np.array(new_gt_peaks, dtype=np.int64)
        else:
            return None
            
    except Exception as e:
        print(f"Error loading manual blink labels: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_manual_gt_peaks(filepath="final_gt_peaks.pkl", convert_from_30fps_to_25fps=False):
    """Load manually edited GT peaks from pickle file.
    
    Args:
        filepath: Path to manual GT peaks file
        convert_from_30fps_to_25fps: If True, converts peak indices from 30fps to 25fps (divide by 1.2)
    
    Returns array of peak indices (in concatenated format).
    """
    import pickle
    from pathlib import Path
    
    if not Path(filepath).exists():
        print(f"Warning: Manual GT peaks file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            manual_peaks = pickle.load(f)
        
        # Convert from 30fps to 25fps if needed (peak editor saves at 30fps but blendshapes are 25fps)
        if convert_from_30fps_to_25fps:
            manual_peaks = (manual_peaks / 1.2).astype(np.int64)
            print(f"✓ Loaded {len(manual_peaks)} manually edited GT peaks from {filepath} (converted from 30fps to 25fps)")
        else:
            print(f"✓ Loaded {len(manual_peaks)} manually edited GT peaks from {filepath}")
        
        return manual_peaks
    except Exception as e:
        print(f"Error loading manual GT peaks: {e}")
        return None

def load_removed_peaks(filepath="removed_peaks.pkl", convert_from_30fps_to_25fps=True):
    """Load peaks to be removed from GT.
    
    Args:
        filepath: Path to removed peaks file
        convert_from_30fps_to_25fps: If True, converts peak indices from 30fps to 25fps (divide by 1.2)
    
    Returns array of peak indices (in concatenated format) to remove.
    """
    import pickle
    from pathlib import Path
    
    if not Path(filepath).exists():
        print(f"Warning: Removed peaks file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            removed_peaks = pickle.load(f)
        
        if isinstance(removed_peaks, np.ndarray):
            removed_peaks_array = removed_peaks
        elif isinstance(removed_peaks, (list, tuple, set)):
            removed_peaks_array = np.array(list(removed_peaks))
        else:
            print(f"Warning: Unexpected format for removed peaks: {type(removed_peaks)}")
            return None
        
        # Convert from 30fps to 25fps if needed (peak editor saves at 30fps but blendshapes are 25fps)
        if convert_from_30fps_to_25fps:
            removed_peaks_array = (removed_peaks_array / 1.2).astype(np.int64)
            print(f"✓ Loaded {len(removed_peaks_array)} peaks to remove from {filepath} (converted from 30fps to 25fps)")
        else:
            print(f"✓ Loaded {len(removed_peaks_array)} peaks to remove from {filepath}")
        
        return removed_peaks_array
    except Exception as e:
        print(f"Error loading removed peaks: {e}")
        return None

def run_blink_analysis(th_list, blendshapes_list, pred_blends_list, quantizer, gt_th, significant_gt_movenents, manual_gt_peaks=None, edge_margin=10, file_boundaries=None, exclude_pred_peaks=None):
    TPR_lst = []
    FNR_lst = []
    FPR_lst = []
    saved_blink_analysis = None  # Store the last analysis for saving
    
    for model_th in tqdm(th_list):
        analyzer = BlinkAnalyzer()
        # gt_th = 0
        # model_th = 0.01
        blink_analysis_model = analyzer.analyze_blinks(gt_th=gt_th,model_th=model_th, blendshapes_list=blendshapes_list, pred_blends_list=pred_blends_list, max_offset=quantizer, significant_gt_movenents=significant_gt_movenents, manual_gt_peaks=manual_gt_peaks, edge_margin=edge_margin, file_boundaries=file_boundaries, exclude_pred_peaks=exclude_pred_peaks)
        # _, fig = list(blink_analysis_model['plots'].items())[1] 
        # fig.show()
        matched_gt = len(blink_analysis_model['matches']['matched_gt'])
        matched_pred = len(blink_analysis_model['matches']['matched_pred'])
        unmatched_gt = len(blink_analysis_model['matches']['unmatched_gt'])
        unmatched_pred = len(blink_analysis_model['matches']['unmatched_pred'])
        pred_concat = blink_analysis_model['pred_concat']
        TPR, FNR, FPR = calculate_metrics(matched_gt, matched_pred, unmatched_gt, unmatched_pred, pred_concat, quantizer)
        TPR_lst.append(TPR)
        FNR_lst.append(FNR)
        FPR_lst.append(FPR)
        
        # Save the last analysis for later use
        saved_blink_analysis = blink_analysis_model
        
    return TPR_lst, FNR_lst, FPR_lst, saved_blink_analysis

def velocity_agreement(gt_bs, pred_bs, already_diff=False):
    if already_diff:
        # Data is already differenced, just use it directly
        da = gt_bs
        db = pred_bs
    else:
        # Apply Savitzky-Golay filter and then compute differences
        gt_smooth = savgol_filter(gt_bs, 9, 2, mode='interp')
        pred_smooth = savgol_filter(pred_bs, 9, 2, mode='interp')
        da = np.diff(gt_smooth)
        db = np.diff(pred_smooth)
    
    sign_match = np.mean(np.sign(da) == np.sign(db))
    r = np.corrcoef(da, db)[0, 1]
    
    return sign_match, r

def plot_gt_pred_time(gt, pred, blendshapes, num_frames=1000):
    import matplotlib.pyplot as plt
    n = gt.shape[1]
    frames = np.arange(min(gt.shape[0], num_frames))
    plt.figure(figsize=(14, n * 2.5))
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.plot(frames, gt[:num_frames, i], label='GT', color='black', linewidth=1.2)
        plt.plot(frames, pred[:num_frames, i], label='Pred', alpha=0.7)
        plt.title(f'Blendshape: {blendshapes[i]}')
        plt.legend()
    plt.tight_layout()
    plt.show()
            
def calculate_correlation_and_distance_metrics(gt, pred, significant_gt_movenents=None, significant_pred_movenents=None, blendshape_names=None, use_diff=False, blendshapes_to_analyze=None):
    assert gt.shape == pred.shape, f"GT and Pred shapes must match: {gt.shape} vs {pred.shape}"
    
    # Determine which blendshapes to analyze
    if blendshapes_to_analyze is not None and blendshape_names is not None:
        # Get indices of blendshapes to analyze
        blendshape_indices = [blendshape_names.index(bs) for bs in blendshapes_to_analyze if bs in blendshape_names]
        analyzed_names = [blendshape_names[i] for i in blendshape_indices]
        
        # Filter GT and Pred to only include selected blendshapes
        gt = gt[:, blendshape_indices]
        pred = pred[:, blendshape_indices]
        if significant_gt_movenents is not None:
            significant_gt_movenents = significant_gt_movenents[:, blendshape_indices]
        if significant_pred_movenents is not None:
            significant_pred_movenents = significant_pred_movenents[:, blendshape_indices]
        # plot_gt_pred_time(gt, pred)
        
        original_gt = gt.copy()
        original_pred = pred.copy()

        print(f"  Analyzing {len(blendshape_indices)} blendshapes: {analyzed_names}")
    else:
        analyzed_names = blendshape_names if blendshape_names is not None else None
        print(f"  Analyzing all {gt.shape[1]} blendshapes")
    
    # Calculate frame-to-frame differences if requested
    if use_diff:
        gt_smooth = savgol_filter(gt, 9, 2, axis=0, mode='interp')
        pred_smooth = savgol_filter(pred, 9, 2, axis=0, mode='interp')
        gt = np.diff(gt_smooth, axis=0)
        pred = np.diff(pred_smooth, axis=0)
        print(f"  Using frame-to-frame differences: shape {gt.shape}")
    
    # Global metrics (flatten all values)
    gt_flat = gt.flatten(order='F')
    pred_flat = pred.flatten(order='F')
    
    global_pcc, _ = pearsonr(gt_flat, pred_flat)
    global_l1 = np.mean(np.abs(gt_flat - pred_flat))
    global_l2 = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
    
    # Direction agreement (only meaningful for differences)
    global_direction_agreement = None
    if use_diff:
        # Calculate percentage of time both GT and Pred have the same sign (direction)
        same_direction = np.sign(gt_flat) == np.sign(pred_flat) #sign_match = np.mean(np.sign(da) == np.sign(db))
        global_direction_agreement = np.mean(same_direction) * 100  # Convert to percentage 
    
    # Per-blendshape metrics
    num_blendshapes = gt.shape[1]
    per_blendshape_pcc = []
    per_blendshape_l1 = []
    per_blendshape_l2 = []
    per_blendshape_direction_agreement = []
    per_blendshape_velocity_agreement = []
    per_blendshape_velocity_correlation = []
    per_blendshape_normalized_rmse = []
    
    # Store original gt and pred for velocity agreement calculation (before diff)
    gt_original = gt.copy()
    pred_original = pred.copy()
    
    for i in range(num_blendshapes):
        gt_bs = gt[:, i]*significant_gt_movenents[:,i]
        pred_bs = pred[:, i]*significant_pred_movenents[:,i]
        
        # PCC for this blendshape
        pcc, _ = pearsonr(gt_bs, pred_bs)
        per_blendshape_pcc.append(pcc)
        
        # L1 for this blendshape
        l1 = np.mean(np.abs(gt_bs - pred_bs))
        per_blendshape_l1.append(l1)
        
        # L2 for this blendshape
        l2 = np.sqrt(np.mean((gt_bs - pred_bs) ** 2))
        per_blendshape_l2.append(l2)
        
        # Normalized RMSE (RMSE / mean of GT)
        gt_mean = np.mean(np.abs(gt_bs))
        if gt_mean > 1e-10:  # Avoid division by zero
            normalized_rmse = l2 / gt_mean
        else:
            normalized_rmse = 0.0
        per_blendshape_normalized_rmse.append(normalized_rmse)
        
        # Direction agreement (only for differences)
        if True: #use_diff:
            same_direction = np.sign(gt_bs) == np.sign(pred_bs)
            direction_agreement = np.mean(same_direction) * 100  # Convert to percentage
            per_blendshape_direction_agreement.append(direction_agreement)
        
        # Velocity agreement with Savitzky-Golay filtering
        # If already using diff, pass the diff data directly; otherwise compute from original
        if True: #use_diff:
            # Data is already differenced, use it directly
            vel_agree, vel_corr = velocity_agreement(gt_bs, pred_bs, already_diff=True)
        else:
            # Use original data and let velocity_agreement apply savgol + diff
            vel_agree, vel_corr = velocity_agreement(gt_original[:, i], pred_original[:, i], already_diff=False)
        
        per_blendshape_velocity_agreement.append(vel_agree * 100)  # Convert to percentage
        per_blendshape_velocity_correlation.append(vel_corr)
    
    results = {
        'global_pcc': global_pcc,
        'global_l1': global_l1,
        'global_l2': global_l2,
        'per_blendshape_pcc': np.array(per_blendshape_pcc),
        'per_blendshape_l1': np.array(per_blendshape_l1),
        'per_blendshape_l2': np.array(per_blendshape_l2),
        'per_blendshape_normalized_rmse': np.array(per_blendshape_normalized_rmse),
        'per_blendshape_velocity_agreement': np.array(per_blendshape_velocity_agreement),
        'per_blendshape_velocity_correlation': np.array(per_blendshape_velocity_correlation),
        'analyzed_blendshapes': analyzed_names
    }
    
    # Add direction agreement metrics only if using differences
    if use_diff:
        results['global_direction_agreement'] = global_direction_agreement
        results['per_blendshape_direction_agreement'] = np.array(per_blendshape_direction_agreement)
    
    return results

def print_metrics_comparison(metrics1, metrics2, model1_name, model2_name, title_suffix=""):
    """Print a formatted comparison of metrics between two models."""
    print("\n" + "="*80)
    print(f"CORRELATION AND DISTANCE METRICS COMPARISON{title_suffix}")
    print("="*80)
    
    # Global metrics
    print(f"\n{'GLOBAL METRICS':<30} {model1_name:<30} {model2_name:<30}")
    print("-" * 90)
    print(f"{'PCC (Pearson Correlation)':<30} {metrics1['global_pcc']:<30.6f} {metrics2['global_pcc']:<30.6f}")
    print(f"{'L1 (Mean Absolute Error)':<30} {metrics1['global_l1']:<30.6f} {metrics2['global_l1']:<30.6f}")
    print(f"{'L2 (RMSE)':<30} {metrics1['global_l2']:<30.6f} {metrics2['global_l2']:<30.6f}")
    
    # Direction agreement (only for diff metrics)
    if 'global_direction_agreement' in metrics1:
        print(f"{'Direction Agreement (%)':<30} {metrics1['global_direction_agreement']:<30.2f} {metrics2['global_direction_agreement']:<30.2f}")
    
    # Per-blendshape metrics - use the analyzed_blendshapes from metrics
    blendshape_names = metrics1.get('analyzed_blendshapes')
    if blendshape_names is not None:
        print(f"\n{'PER-BLENDSHAPE METRICS':<30}")
        print("-" * 90)
        
        # PCC
        print(f"\n{'Blendshape':<30} {'PCC ('+model1_name+')':<30} {'PCC ('+model2_name+')':<30}")
        print("-" * 90)
        for i, bs_name in enumerate(blendshape_names):
            print(f"{bs_name:<30} {metrics1['per_blendshape_pcc'][i]:<30.6f} {metrics2['per_blendshape_pcc'][i]:<30.6f}")
        
        # L1
        print(f"\n{'Blendshape':<30} {'L1 ('+model1_name+')':<30} {'L1 ('+model2_name+')':<30}")
        print("-" * 90)
        for i, bs_name in enumerate(blendshape_names):
            print(f"{bs_name:<30} {metrics1['per_blendshape_l1'][i]:<30.6f} {metrics2['per_blendshape_l1'][i]:<30.6f}")
        
        # L2
        print(f"\n{'Blendshape':<30} {'L2 ('+model1_name+')':<30} {'L2 ('+model2_name+')':<30}")
        print("-" * 90)
        for i, bs_name in enumerate(blendshape_names):
            print(f"{bs_name:<30} {metrics1['per_blendshape_l2'][i]:<30.6f} {metrics2['per_blendshape_l2'][i]:<30.6f}")
        
        # Normalized RMSE
        print(f"\n{'Blendshape':<30} {'Norm RMSE ('+model1_name+')':<30} {'Norm RMSE ('+model2_name+')':<30}")
        print("-" * 90)
        for i, bs_name in enumerate(blendshape_names):
            print(f"{bs_name:<30} {metrics1['per_blendshape_normalized_rmse'][i]:<30.6f} {metrics2['per_blendshape_normalized_rmse'][i]:<30.6f}")
        
        # Velocity Agreement (with Savitzky-Golay)
        print(f"\n{'Blendshape':<30} {'Vel Agree % ('+model1_name+')':<30} {'Vel Agree % ('+model2_name+')':<30}")
        print("-" * 90)
        for i, bs_name in enumerate(blendshape_names):
            print(f"{bs_name:<30} {metrics1['per_blendshape_velocity_agreement'][i]:<30.2f} {metrics2['per_blendshape_velocity_agreement'][i]:<30.2f}")
        
        # Velocity Correlation (with Savitzky-Golay)
        print(f"\n{'Blendshape':<30} {'Vel Corr ('+model1_name+')':<30} {'Vel Corr ('+model2_name+')':<30}")
        print("-" * 90)
        for i, bs_name in enumerate(blendshape_names):
            print(f"{bs_name:<30} {metrics1['per_blendshape_velocity_correlation'][i]:<30.6f} {metrics2['per_blendshape_velocity_correlation'][i]:<30.6f}")
        
        # Direction Agreement (only for diff metrics)
        if 'per_blendshape_direction_agreement' in metrics1:
            print(f"\n{'Blendshape':<30} {'Dir Agree % ('+model1_name+')':<30} {'Dir Agree % ('+model2_name+')':<30}")
            print("-" * 90)
            for i, bs_name in enumerate(blendshape_names):
                print(f"{bs_name:<30} {metrics1['per_blendshape_direction_agreement'][i]:<30.2f} {metrics2['per_blendshape_direction_agreement'][i]:<30.2f}")
    
    print("\n" + "="*80)

def plot_metrics_bar_chart(metrics1, metrics2, model1_name, model2_name, title_suffix=""):
    """Create bar plots comparing metrics between two models.
    
    Args:
        metrics1: Metrics dictionary for model 1
        metrics2: Metrics dictionary for model 2
        model1_name: Name of model 1
        model2_name: Name of model 2
        title_suffix: Optional suffix for the plot title
    """
    blendshape_names = metrics1.get('analyzed_blendshapes')
    if blendshape_names is None:
        print("No blendshape names available for plotting")
        return
    
    num_blendshapes = len(blendshape_names)
    
    # Create a color palette for blendshapes
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    # Extend colors if needed
    while len(colors) < num_blendshapes:
        colors.extend(colors)
    colors = colors[:num_blendshapes]
    
    # Check if we have direction agreement metrics (for diff mode)
    has_direction_agreement = 'per_blendshape_direction_agreement' in metrics1
    
    # Create subplots - more columns for additional metrics
    # Row 1: PCC, L1, L2, Normalized RMSE
    # Row 2: Velocity Agreement, Velocity Correlation, Direction Agreement (if diff mode)
    num_cols_row1 = 4
    num_cols_row2 = 3 if has_direction_agreement else 2
    
    subplot_titles_row1 = ['PCC (Pearson Correlation)', 'L1 (Mean Absolute Error)', 'L2 (RMSE)', 'Normalized RMSE']
    subplot_titles_row2 = ['Velocity Agreement (%)', 'Velocity Correlation']
    if has_direction_agreement:
        subplot_titles_row2.append('Direction Agreement (%)')
    
    fig = make_subplots(
        rows=2, cols=max(num_cols_row1, num_cols_row2),
        subplot_titles=tuple(subplot_titles_row1 + subplot_titles_row2),
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    # Prepare data for plotting - use display names for plots
    models = [MODEL_NAME_TO_PLOT, MODEL2_NAME_TO_PLOT]
    
    # Plot PCC
    for i, bs_name in enumerate(blendshape_names):
        pcc_values = [metrics1['per_blendshape_pcc'][i], metrics2['per_blendshape_pcc'][i]]
        fig.add_trace(
            go.Bar(
                x=models,
                y=pcc_values,
                name=bs_name,
                marker_color=colors[i],
                showlegend=True,
                legendgroup=bs_name,
                text=[f'{v:.3f}' for v in pcc_values],
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # Plot L1
    for i, bs_name in enumerate(blendshape_names):
        l1_values = [metrics1['per_blendshape_l1'][i], metrics2['per_blendshape_l1'][i]]
        fig.add_trace(
            go.Bar(
                x=models,
                y=l1_values,
                name=bs_name,
                marker_color=colors[i],
                showlegend=False,
                legendgroup=bs_name,
                text=[f'{v:.3f}' for v in l1_values],
                textposition='outside'
            ),
            row=1, col=2
        )
    
    # Plot L2
    for i, bs_name in enumerate(blendshape_names):
        l2_values = [metrics1['per_blendshape_l2'][i], metrics2['per_blendshape_l2'][i]]
        fig.add_trace(
            go.Bar(
                x=models,
                y=l2_values,
                name=bs_name,
                marker_color=colors[i],
                showlegend=False,
                legendgroup=bs_name,
                text=[f'{v:.3f}' for v in l2_values],
                textposition='outside'
            ),
            row=1, col=3
        )
    
    # Plot Normalized RMSE
    for i, bs_name in enumerate(blendshape_names):
        norm_rmse_values = [metrics1['per_blendshape_normalized_rmse'][i], metrics2['per_blendshape_normalized_rmse'][i]]
        fig.add_trace(
            go.Bar(
                x=models,
                y=norm_rmse_values,
                name=bs_name,
                marker_color=colors[i],
                showlegend=False,
                legendgroup=bs_name,
                text=[f'{v:.3f}' for v in norm_rmse_values],
                textposition='outside'
            ),
            row=1, col=4
        )
    
    # Plot Velocity Agreement
    for i, bs_name in enumerate(blendshape_names):
        vel_agree_values = [
            metrics1['per_blendshape_velocity_agreement'][i], 
            metrics2['per_blendshape_velocity_agreement'][i]
        ]
        fig.add_trace(
            go.Bar(
                x=models,
                y=vel_agree_values,
                name=bs_name,
                marker_color=colors[i],
                showlegend=False,
                legendgroup=bs_name,
                text=[f'{v:.1f}' for v in vel_agree_values],
                textposition='outside'
            ),
            row=2, col=1
        )
    
    # Plot Velocity Correlation
    for i, bs_name in enumerate(blendshape_names):
        vel_corr_values = [
            metrics1['per_blendshape_velocity_correlation'][i], 
            metrics2['per_blendshape_velocity_correlation'][i]
        ]
        fig.add_trace(
            go.Bar(
                x=models,
                y=vel_corr_values,
                name=bs_name,
                marker_color=colors[i],
                showlegend=False,
                legendgroup=bs_name,
                text=[f'{v:.3f}' for v in vel_corr_values],
                textposition='outside'
            ),
            row=2, col=2
        )
    
    # Plot Direction Agreement (if available, only for diff mode)
    if has_direction_agreement:
        for i, bs_name in enumerate(blendshape_names):
            dir_agree_values = [
                metrics1['per_blendshape_direction_agreement'][i], 
                metrics2['per_blendshape_direction_agreement'][i]
            ]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=dir_agree_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=bs_name,
                    text=[f'{v:.1f}' for v in dir_agree_values],
                    textposition='outside'
                ),
                row=2, col=3
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"Per-Blendshape Metrics, threshold factor: {THRESHOLD_FACTOR}, #samples: {NUM_RANDOM_SAMPLES}, {get_blendshapes_category_name()}",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=20,
        height=900,
        width=2400,
        barmode='group',
        legend=dict(
            title="Blendshapes",
            font=dict(size=12),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Update y-axes labels - Row 1
    fig.update_yaxes(title_text="PCC Value", row=1, col=1)
    fig.update_yaxes(title_text="L1 Error", row=1, col=2)
    fig.update_yaxes(title_text="L2 Error (RMSE)", row=1, col=3)
    fig.update_yaxes(title_text="Normalized RMSE", row=1, col=4)
    
    # Update y-axes labels - Row 2
    fig.update_yaxes(title_text="Velocity Agreement (%)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity Correlation", row=2, col=2)
    if has_direction_agreement:
        fig.update_yaxes(title_text="Direction Agreement (%)", row=2, col=3)
    
    # Update x-axes
    for row in range(1, 3):
        for col in range(1, 5):
            fig.update_xaxes(tickangle=45, row=row, col=col)
    
    fig.show()
    
    return fig

def plot_velocity_agreement_only(metrics1, model1_name):
    """Create a bar plot showing only velocity agreement for model 1.
    
    Args:
        metrics1: Metrics dictionary for model 1
        model1_name: Name of model 1
    """
    blendshape_names = metrics1.get('analyzed_blendshapes')
    if blendshape_names is None:
        print("No blendshape names available for plotting")
        return
    
    num_blendshapes = len(blendshape_names)
    
    # Create a color palette for blendshapes
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    # Extend colors if needed
    while len(colors) < num_blendshapes:
        colors.extend(colors)
    colors = colors[:num_blendshapes]
    
    # Create single plot for velocity agreement
    fig = go.Figure()
    
    # Plot Velocity Agreement for model 1 only
    for i, bs_name in enumerate(blendshape_names):
        vel_agree_value = metrics1['per_blendshape_velocity_agreement'][i]
        fig.add_trace(
            go.Bar(
                x=[bs_name],
                y=[vel_agree_value],
                name=bs_name,
                marker_color=colors[i],
                showlegend=False,
                text=[f'{vel_agree_value:.1f}%'],
                textposition='outside'
            )
        )
    
    # Update layout
    category_name = get_blendshapes_category_name()
    fig.update_layout(
        title_text=f"Velocity Agreement (%) - {model1_name}, threshold factor: {THRESHOLD_FACTOR}, #samples: {NUM_RANDOM_SAMPLES}, {category_name}",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=20,
        height=600,
        width=800,  # Narrower figure width
        yaxis_title="Velocity Agreement (%)",
        xaxis_title="Blendshapes",
        xaxis_tickangle=45
    )
    
    fig.show()
    
    return fig

def plot_all_categories_metrics(all_category_metrics, model1_name, model2_name, title_suffix=""):
    """Create subplots comparing metrics across all blendshape categories.
    
    Args:
        all_category_metrics: List of tuples (category_name, metrics1, metrics2)
        model1_name: Name of model 1
        model2_name: Name of model 2
        title_suffix: Optional suffix for the plot title
    """
    num_categories = len(all_category_metrics)
    if num_categories == 0:
        print("No category metrics available for plotting")
        return
    
    # Create a color palette for blendshapes
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Check if we have direction agreement metrics (for diff mode)
    has_direction_agreement = 'per_blendshape_direction_agreement' in all_category_metrics[0][1]
    
    # Create subplots - 2 rows, each category gets one column
    # Row 1: PCC, L1, L2, Normalized RMSE
    # Row 2: Velocity Agreement, Velocity Correlation, Direction Agreement (if diff mode)
    num_cols_row1 = 4
    num_cols_row2 = 3 if has_direction_agreement else 2
    
    subplot_titles = []
    for category_name, _, _ in all_category_metrics:
        subplot_titles.append(f'{category_name}')
    
    # Create figure with subplots - one row per category
    num_metrics = 7 if has_direction_agreement else 6
    fig = make_subplots(
        rows=num_categories, 
        cols=num_metrics,
        subplot_titles=None,  # We'll add custom titles
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        specs=[[{'type': 'bar'} for _ in range(num_metrics)] for _ in range(num_categories)]
    )
    
    # Prepare data for plotting - use display names for plots
    models = [MODEL_NAME_TO_PLOT, MODEL2_NAME_TO_PLOT]
    
    metric_titles = ['PCC', 'L1', 'L2', 'Norm RMSE', 'Vel Agree %', 'Vel Corr']
    if has_direction_agreement:
        metric_titles.append('Dir Agree %')
    
    # Plot each category
    for cat_idx, (category_name, metrics1, metrics2) in enumerate(all_category_metrics):
        row = cat_idx + 1
        blendshape_names = metrics1.get('analyzed_blendshapes')
        num_blendshapes = len(blendshape_names)
        
        # Extend colors if needed
        colors = base_colors.copy()
        while len(colors) < num_blendshapes:
            colors.extend(base_colors)
        colors = colors[:num_blendshapes]
        
        # Plot PCC (col 1)
        for i, bs_name in enumerate(blendshape_names):
            pcc_values = [metrics1['per_blendshape_pcc'][i], metrics2['per_blendshape_pcc'][i]]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=pcc_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=(cat_idx == 0),
                    legendgroup=bs_name,
                    text=[f'{v:.2f}' for v in pcc_values],
                    textposition='outside',
                    textfont=dict(size=8)
                ),
                row=row, col=1
            )
        
        # Plot L1 (col 2)
        for i, bs_name in enumerate(blendshape_names):
            l1_values = [metrics1['per_blendshape_l1'][i], metrics2['per_blendshape_l1'][i]]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=l1_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=bs_name,
                    text=[f'{v:.2f}' for v in l1_values],
                    textposition='outside',
                    textfont=dict(size=8)
                ),
                row=row, col=2
            )
        
        # Plot L2 (col 3)
        for i, bs_name in enumerate(blendshape_names):
            l2_values = [metrics1['per_blendshape_l2'][i], metrics2['per_blendshape_l2'][i]]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=l2_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=bs_name,
                    text=[f'{v:.2f}' for v in l2_values],
                    textposition='outside',
                    textfont=dict(size=8)
                ),
                row=row, col=3
            )
        
        # Plot Normalized RMSE (col 4)
        for i, bs_name in enumerate(blendshape_names):
            norm_rmse_values = [metrics1['per_blendshape_normalized_rmse'][i], metrics2['per_blendshape_normalized_rmse'][i]]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=norm_rmse_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=bs_name,
                    text=[f'{v:.2f}' for v in norm_rmse_values],
                    textposition='outside',
                    textfont=dict(size=8)
                ),
                row=row, col=4
            )
        
        # Plot Velocity Agreement (col 5)
        for i, bs_name in enumerate(blendshape_names):
            vel_agree_values = [
                metrics1['per_blendshape_velocity_agreement'][i], 
                metrics2['per_blendshape_velocity_agreement'][i]
            ]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=vel_agree_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=bs_name,
                    text=[f'{v:.1f}' for v in vel_agree_values],
                    textposition='outside',
                    textfont=dict(size=8)
                ),
                row=row, col=5
            )
        
        # Plot Velocity Correlation (col 6)
        for i, bs_name in enumerate(blendshape_names):
            vel_corr_values = [
                metrics1['per_blendshape_velocity_correlation'][i], 
                metrics2['per_blendshape_velocity_correlation'][i]
            ]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=vel_corr_values,
                    name=bs_name,
                    marker_color=colors[i],
                    showlegend=False,
                    legendgroup=bs_name,
                    text=[f'{v:.2f}' for v in vel_corr_values],
                    textposition='outside',
                    textfont=dict(size=8)
                ),
                row=row, col=6
            )
        
        # Plot Direction Agreement (col 7, if available)
        if has_direction_agreement:
            for i, bs_name in enumerate(blendshape_names):
                dir_agree_values = [
                    metrics1['per_blendshape_direction_agreement'][i], 
                    metrics2['per_blendshape_direction_agreement'][i]
                ]
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=dir_agree_values,
                        name=bs_name,
                        marker_color=colors[i],
                        showlegend=False,
                        legendgroup=bs_name,
                        text=[f'{v:.1f}' for v in dir_agree_values],
                        textposition='outside',
                        textfont=dict(size=8)
                    ),
                    row=row, col=7
                )
        
        # Add category name as y-axis title for the first column
        fig.update_yaxes(title_text=category_name, row=row, col=1, title_font=dict(size=10))
    
    # Update layout
    fig.update_layout(
        title_text=f"Per-Blendshape Metrics Across Categories{title_suffix}<br><sub>threshold factor: {THRESHOLD_FACTOR}, #samples: {NUM_RANDOM_SAMPLES}</sub>",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=18,
        height=300 * num_categories,
        width=2800,
        barmode='group',
        legend=dict(
            title="Blendshapes",
            font=dict(size=10),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Add column titles at the top
    for col_idx, metric_title in enumerate(metric_titles):
        fig.add_annotation(
            text=f"<b>{metric_title}</b>",
            xref="x domain" if col_idx == 0 else f"x{col_idx+1} domain",
            yref="paper",
            x=0.5,
            y=1.02,
            showarrow=False,
            font=dict(size=14),
            xanchor='center'
        )
    
    # Update x-axes
    for row in range(1, num_categories + 1):
        for col in range(1, num_metrics + 1):
            fig.update_xaxes(tickangle=45, row=row, col=col, tickfont=dict(size=8))
    
    fig.show()
    
    return fig

def plot_velocity_agreement_all_categories(all_category_metrics, model1_name, title_suffix=""):
    """Create subplots showing only velocity agreement for model 1 across all categories.
    Each blendshape has its own color that is consistent across all categories.
    
    Args:
        all_category_metrics: List of tuples (category_name, metrics1, metrics2)
        model1_name: Name of model 1
        title_suffix: Optional suffix for the plot title
    """
    num_categories = len(all_category_metrics)
    if num_categories == 0:
        print("No category metrics available for plotting")
        return
    
    def clean_blendshape_name(name):
        """Remove 'Right' and 'Left' from blendshape names for display."""
        return name.replace('Right', '').replace('Left', '')
    
    # Create a color palette - collect all unique CLEANED blendshape names first
    # This ensures same color for Left/Right versions
    all_cleaned_blendshapes = set()
    for category_name, metrics1, _ in all_category_metrics:
        blendshape_names = metrics1.get('analyzed_blendshapes')
        if blendshape_names:
            for bs_name in blendshape_names:
                # Skip Left if Right exists
                if 'Left' in bs_name:
                    right_version = bs_name.replace('Left', 'Right')
                    if right_version in blendshape_names:
                        continue
                cleaned_name = clean_blendshape_name(bs_name)
                all_cleaned_blendshapes.add(cleaned_name)
    
    # Sort blendshapes for consistent ordering
    all_cleaned_blendshapes = sorted(list(all_cleaned_blendshapes))
    
    # Create color mapping for each unique blendshape - using vibrant color palette
    nice_colors = [
        '#E63946',  # Red (vibrant crimson)
        '#F77F00',  # Orange
        '#FCBF49',  # Yellow
        '#06D6A0',  # Teal (mint/turquoise)
        '#118AB2',  # Blue (ocean blue)
        '#073B4C',  # Dark Blue (navy)
        '#8338EC',  # Purple (bright violet)
        '#FF006E',  # Pink/Magenta
        '#FB5607',  # Bright Orange
        '#FFBE0B',  # Golden Yellow
        '#3A86FF',  # Bright Blue (cornflower)
        '#8AC926',  # Lime Green
        '#06FFA5',  # Mint (bright cyan)
        '#4361EE',  # Royal Blue
        '#F72585',  # Hot Pink
        '#7209B7',  # Deep Purple
        '#560BAD',  # Dark Purple (indigo)
        '#B5179E',  # Violet
        '#F72585',  # Rose
        # Additional colors if needed
        '#D90429',  # Crimson Red
        '#EF476F',  # Watermelon
        '#FFD60A',  # Bright Yellow
        '#06FFA5',  # Aqua
        '#4CC9F0',  # Sky Blue
        '#4361EE',  # Ultramarine
        '#7209B7',  # Purple
        '#F72585',  # Persian Rose
    ]
    
    # Extend colors if needed
    while len(nice_colors) < len(all_cleaned_blendshapes):
        nice_colors.extend(nice_colors)
    
    # Create blendshape to color mapping based on CLEANED names
    blendshape_color_map = {bs: nice_colors[i] for i, bs in enumerate(all_cleaned_blendshapes)}
    
    # Create subplots - one subplot per category
    fig = make_subplots(
        rows=1, 
        cols=num_categories,
        subplot_titles=[cat_name for cat_name, _, _ in all_category_metrics],
        horizontal_spacing=0.08
    )
    
    # First pass: collect all velocity agreement values to determine y-axis range
    all_vel_values = []
    for category_name, metrics1, _ in all_category_metrics:
        blendshape_names = metrics1.get('analyzed_blendshapes')
        if blendshape_names is None:
            continue
        
        for i, bs_name in enumerate(blendshape_names):
            # Skip if this is a Left blendshape and a Right version exists
            if 'Left' in bs_name:
                right_version = bs_name.replace('Left', 'Right')
                if right_version in blendshape_names:
                    continue
            
            vel_agree_value = metrics1['per_blendshape_velocity_agreement'][i]
            all_vel_values.append(vel_agree_value)
    
    # Determine y-axis range
    if all_vel_values:
        y_min = 0  # Start from 0 for percentage
        y_max = max(all_vel_values) * 1.15  # Add 15% padding for text labels
    else:
        y_min, y_max = 0, 100
    
    # Plot each category
    for cat_idx, (category_name, metrics1, _) in enumerate(all_category_metrics):
        col = cat_idx + 1
        blendshape_names = metrics1.get('analyzed_blendshapes')
        
        if blendshape_names is None:
            continue
        
        # Filter to only plot Right blendshapes (or blendshapes without Left/Right)
        # Skip Left blendshapes to avoid duplicates
        for i, bs_name in enumerate(blendshape_names):
            # Skip if this is a Left blendshape and a Right version exists
            if 'Left' in bs_name:
                right_version = bs_name.replace('Left', 'Right')
                if right_version in blendshape_names:
                    continue  # Skip Left, we'll plot the Right version instead
            
            vel_agree_value = metrics1['per_blendshape_velocity_agreement'][i]
            display_name = clean_blendshape_name(bs_name)
            
            fig.add_trace(
                go.Bar(
                    x=[display_name],
                    y=[vel_agree_value],
                    name=display_name,
                    marker_color=blendshape_color_map[display_name],  # Use cleaned name for color lookup
                    showlegend=(cat_idx == 0),  # Only show legend for first category
                    legendgroup=display_name,  # Group by cleaned name
                    text=[f'{vel_agree_value:.1f}%'],
                    textposition='outside',
                    textfont=dict(size=10)
                ),
                row=1, col=col
            )
        
        # Update y-axis for this subplot with shared range
        fig.update_yaxes(
            title_text="Velocity Agreement (%)", 
            row=1, col=col,
            range=[y_min, y_max]
        )
        fig.update_xaxes(tickangle=45, row=1, col=col)
    
    # Update layout
    fig.update_layout(
        title_text=f"Velocity Agreement (%) ",#- {model1_name} Across Categories{title_suffix}<br><sub>threshold factor: {THRESHOLD_FACTOR}, #samples: {NUM_RANDOM_SAMPLES}</sub>",
        # title_text=f"Velocity Agreement (%) - {model1_name} Across Categories{title_suffix}<br><sub>threshold factor: {THRESHOLD_FACTOR}, #samples: {NUM_RANDOM_SAMPLES}</sub>",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=18,
        height=600,
        width=400 * num_categories,
        showlegend=True,
        legend=dict(
            title="Blendshapes",
            font=dict(size=10),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    fig.show()
    
    return fig

def plot_velocity_agreement_above_threshold(metrics1, model1_name, threshold=80.0, title_suffix=""):
    """Plot all blendshapes with velocity agreement above the threshold.
    Only uses Right blendshapes and displays names without 'Right' suffix.
    
    Args:
        metrics1: Metrics dictionary for model 1
        model1_name: Name of model 1
        threshold: Minimum velocity agreement percentage to include
        title_suffix: Optional suffix for the plot title
    """
    blendshape_names = metrics1.get('analyzed_blendshapes')
    if blendshape_names is None:
        print("No blendshape names available for plotting")
        return
    
    def clean_blendshape_name(name):
        """Remove 'Right' and 'Left' from blendshape names for display."""
        return name.replace('Right', '').replace('Left', '')
    
    # Filter blendshapes: only Right versions (or no side), and above threshold
    filtered_blendshapes = []
    for i, bs_name in enumerate(blendshape_names):
        # Skip if this is a Left blendshape and a Right version exists
        if 'Left' in bs_name:
            right_version = bs_name.replace('Left', 'Right')
            if right_version in blendshape_names:
                continue  # Skip Left, we'll use the Right version instead
        
        vel_agree_value = metrics1['per_blendshape_velocity_agreement'][i]
        
        # Only include if above threshold
        if vel_agree_value >= threshold:
            display_name = clean_blendshape_name(bs_name)
            filtered_blendshapes.append((display_name, vel_agree_value, bs_name))
    
    if not filtered_blendshapes:
        print(f"No blendshapes found with velocity agreement >= {threshold}%")
        return
    
    # Define custom order for blendshapes
    custom_order = [
        'mouthSmile',
        'mouthFrown', 
        'mouthDimple',
        'mouthPucker',
        'mouthFunnel',
        'mouthClose',
        'jawOpen',
        'jaw',  # This will match jawRight (will be displayed as "jaw Side")
        'cheekPuff'
    ]
    
    def get_sort_key(item):
        """Return sort key based on custom order."""
        display_name, vel_agree_value, original_name = item
        
        # Special handling for jaw - add "Side" to the display name
        if 'jaw' in original_name.lower() and 'open' not in original_name.lower():
            # This is jawRight or jawLeft
            return (custom_order.index('jaw'), display_name)
        
        # Find the base name in custom order
        for idx, order_name in enumerate(custom_order):
            if order_name.lower() in display_name.lower():
                return (idx, display_name)
        
        # If not in custom order, put at the end
        return (len(custom_order), display_name)
    
    # Sort by custom order
    filtered_blendshapes.sort(key=get_sort_key)
    
    # Update display names for jaw (add "Side")
    filtered_blendshapes = [
        (display_name + ' Side' if 'jaw' in original_name.lower() and 'open' not in original_name.lower() else display_name,
         vel_agree_value, 
         original_name)
        for display_name, vel_agree_value, original_name in filtered_blendshapes
    ]
    
    # Create color palette - same vibrant palette as category plots
    nice_colors = [
        '#E63946',  # Red (vibrant crimson)
        '#F77F00',  # Orange
        '#FCBF49',  # Yellow
        '#06D6A0',  # Teal (mint/turquoise)
        '#118AB2',  # Blue (ocean blue)
        '#073B4C',  # Dark Blue (navy)
        '#8338EC',  # Purple (bright violet)
        '#FF006E',  # Pink/Magenta
        '#FB5607',  # Bright Orange
        '#FFBE0B',  # Golden Yellow
        '#3A86FF',  # Bright Blue (cornflower)
        '#8AC926',  # Lime Green
        '#06FFA5',  # Mint (bright cyan)
        '#4361EE',  # Royal Blue
        '#F72585',  # Hot Pink
        '#7209B7',  # Deep Purple
        '#560BAD',  # Dark Purple (indigo)
        '#B5179E',  # Violet
        '#F72585',  # Rose
        # Additional colors if needed
        '#D90429',  # Crimson Red
        '#EF476F',  # Watermelon
        '#FFD60A',  # Bright Yellow
        '#06FFA5',  # Aqua
        '#4CC9F0',  # Sky Blue
        '#4361EE',  # Ultramarine
        '#7209B7',  # Purple
        '#F72585',  # Persian Rose
    ]
    
    # Extend colors if needed
    while len(nice_colors) < len(filtered_blendshapes):
        nice_colors.extend(nice_colors)
    
    # Create figure
    fig = go.Figure()
    
    # Plot each blendshape
    for i, (display_name, vel_agree_value, original_name) in enumerate(filtered_blendshapes):
        fig.add_trace(
            go.Bar(
                x=[display_name],
                y=[vel_agree_value],
                name=display_name,
                marker_color=nice_colors[i],
                showlegend=True,
                text=[f'{vel_agree_value:.1f}%'],
                textposition='outside',
                textfont=dict(size=10)
            )
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Velocity Agreement (%)",# - Blendshapes Above {threshold}%",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=18,
        height=600,
        width=max(800, len(filtered_blendshapes) * 60),  # Dynamic width based on number of bars
        yaxis_title="Velocity Agreement (%)",
        xaxis_title="Blendshapes",
        xaxis_tickangle=45,
        showlegend=True,
        legend=dict(
            title="Blendshapes",
            font=dict(size=10),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        )
    )
    
    # Set y-axis range
    y_max = max([v for _, v, _ in filtered_blendshapes]) * 1.15
    fig.update_yaxes(range=[0, y_max])
    
    fig.show()
    
    return fig

def get_significant_movenents_mask(blendshapes, threshold_factor=0.5):
    significant_movenents_mask = np.zeros(blendshapes.shape, dtype=bool)
    for i in range(blendshapes.shape[1]):
        th = threshold_factor*np.max(blendshapes[:, i])
        significant_movenents_mask[:, i] = significant_movenents_mask[:, i] | (np.abs(blendshapes[:, i]) > th)
    return significant_movenents_mask

def normalize_blinks(data):
    return (data - np.median(data)) / (1 - np.median(data))
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("BLINKING ANALYSIS - SPLIT VERSION")
    print("="*80)
    
    # Get run paths, tar_ids, and side from split (filter for right side only)
    print("\n" + "="*80)
    print("LOADING RUN PATHS FROM SPLIT (RIGHT SIDE ONLY)")
    print("="*80)
    run_path_tar_id_side_tuples = get_run_paths_from_split(
        SPLIT_DF_PATH, 
        ROW_INDICES, 
        side_filter=None, #'right',
        num_random_samples=NUM_RANDOM_SAMPLES,
        random_seed=RANDOM_SEED
    )
    
    print(f"\n" + "="*80)
    print(f"PROCESSING {len(run_path_tar_id_side_tuples)} FILES")
    print("="*80)
    
    # Collect data from all files
    all_gt_blendshapes = []
    all_pred_blendshapes = []
    all_pred_blendshapes_model2 = []
    file_boundaries = []
    current_frame = 0
    all_diff_gt_blendshapes = []
    all_diff_pred_blendshapes = []
    all_diff_pred_blendshapes_model2 = []
    all_significant_gt_movenents = []
    all_significant_pred_movenents = []
    all_significant_pred_model2_movenents = []
    successfully_loaded_tuples = []  # Track which files were successfully loaded
    file_frame_mapping = {}  # Map (run_path, tar_id, side) to (start_frame, end_frame)
    
    # Process each file
    for run_path, tar_id, side in tqdm(run_path_tar_id_side_tuples, desc="Loading files"):
        print(f"\nProcessing: {run_path} (tar_id: {tar_id}, side: {side})")
        
        # Load ground truth
        gt_blendshapes = load_gt_blendshapes(run_path)
        if gt_blendshapes is None:
            print(f"Skipping {run_path} - could not load ground truth")
            continue
        
        # Load prediction from Model 1
        pred_blendshapes = load_prediction(run_path, tar_id, side, MODEL_NAME, HISTORY_SIZE, LOOKAHEAD_SIZE)
        if pred_blendshapes is None:
            print(f"Skipping {run_path}/{tar_id} (side: {side}) - could not load prediction for {MODEL_NAME}")
            continue
        
        # Load prediction from Model 2
        pred_blendshapes_model2 = load_prediction(run_path, tar_id, side, MODEL2_NAME, HISTORY_SIZE, LOOKAHEAD_SIZE)
        if pred_blendshapes_model2 is None:
            print(f"Skipping {run_path}/{tar_id} (side: {side}) - could not load prediction for {MODEL2_NAME}")
            continue
        
        print(f"GT shape: {gt_blendshapes.shape}, Pred1 shape: {pred_blendshapes.shape}, Pred2 shape: {pred_blendshapes_model2.shape}")
        
        # Align lengths (pad or truncate prediction to match GT)
        gt_len = gt_blendshapes.shape[0]
        pred_len = pred_blendshapes.shape[0]
        pred_len_model2 = pred_blendshapes_model2.shape[0]
        
        # Align Model 1 predictions
        if pred_len < gt_len:
            # Pad prediction with zeros
            num_blendshapes = pred_blendshapes.shape[1]
            padding = np.zeros((gt_len - pred_len, num_blendshapes))
            pred_blendshapes = np.concatenate([pred_blendshapes, padding], axis=0)
            print(f"Padded {MODEL_NAME} prediction: {pred_len} -> {gt_len} frames")
        elif pred_len > gt_len:
            # Truncate prediction
            pred_blendshapes = pred_blendshapes[:gt_len]
            print(f"Truncated {MODEL_NAME} prediction: {pred_len} -> {gt_len} frames")
        
        # Align Model 2 predictions
        if pred_len_model2 < gt_len:
            # Pad prediction with zeros
            num_blendshapes = pred_blendshapes_model2.shape[1]
            padding = np.zeros((gt_len - pred_len_model2, num_blendshapes))
            pred_blendshapes_model2 = np.concatenate([pred_blendshapes_model2, padding], axis=0)
            print(f"Padded {MODEL2_NAME} prediction: {pred_len_model2} -> {gt_len} frames")
        elif pred_len_model2 > gt_len:
            # Truncate prediction
            pred_blendshapes_model2 = pred_blendshapes_model2[:gt_len]
            print(f"Truncated {MODEL2_NAME} prediction: {pred_len_model2} -> {gt_len} frames")
        
        
        gt_blendshapes_smooth = savgol_filter(gt_blendshapes, 9, 2, axis=0, mode='interp')
        pred_blendshapes_smooth = savgol_filter(pred_blendshapes, 9, 2, axis=0, mode='interp')
        # plot_one_bs(gt_blendshapes_smooth[:,10], pred_blendshapes_smooth[:,10])
        # plot_one_bs(zscore(gt_blendshapes_smooth[:,10]), zscore(pred_blendshapes_smooth[:,10]))
        # plot_one_bs(gt_blendshapes_smooth[:,10], pred_blendshapes_smooth[:,10])
        # plot_one_bs(normalize_blinks(gt_blendshapes_smooth[:,10]), normalize_blinks(pred_blendshapes_smooth[:,10]))
        pred_blendshapes_model2_smooth = savgol_filter(pred_blendshapes_model2, 9, 2, axis=0, mode='interp')
        
        diff_gt_blendshapes = np.diff(gt_blendshapes_smooth, axis=0)
        diff_pred_blendshapes = np.diff(pred_blendshapes_smooth, axis=0)
        diff_pred_blendshapes_model2 = np.diff(pred_blendshapes_model2_smooth, axis=0)
        
        significant_gt_movenents = get_significant_movenents_mask(gt_blendshapes_smooth[:-1,:], threshold_factor=THRESHOLD_FACTOR)
        significant_pred_movenents = significant_gt_movenents #get_significant_movenents_mask(pred_blendshapes_smooth[:-1,:], threshold_factor=THRESHOLD_FACTOR)
        significant_pred_model2_movenents = significant_gt_movenents #get_significant_movenents_mask(pred_blendshapes_model2_smooth[:-1,:], threshold_factor=THRESHOLD_FACTOR)
        
        # Store data
        start_frame = current_frame
        num_frames = gt_blendshapes.shape[0]
        end_frame = start_frame + num_frames
        
        # Add to file frame mapping
        file_frame_mapping[(run_path, tar_id, side)] = (start_frame, end_frame)
        
        all_gt_blendshapes.append(gt_blendshapes)
        all_diff_gt_blendshapes.append(diff_gt_blendshapes)
        all_diff_pred_blendshapes.append(diff_pred_blendshapes)
        all_pred_blendshapes.append(pred_blendshapes)
        all_diff_pred_blendshapes_model2.append(diff_pred_blendshapes_model2)
        all_pred_blendshapes_model2.append(pred_blendshapes_model2)
        current_frame = end_frame
        file_boundaries.append(current_frame)
        all_significant_gt_movenents.append(significant_gt_movenents)
        all_significant_pred_movenents.append(significant_pred_movenents)
        all_significant_pred_model2_movenents.append(significant_pred_model2_movenents)
        successfully_loaded_tuples.append((run_path, tar_id, side))
    
    # Remove last boundary (end of last file)
    if file_boundaries:
        file_boundaries = file_boundaries[:-1]
    
    # Concatenate all data
    print(f"\n\nConcatenating all files...")
    concatenated_gt = np.concatenate(all_gt_blendshapes, axis=0)
    concatenated_pred = np.concatenate(all_pred_blendshapes, axis=0)
    concatenated_pred_model2 = np.concatenate(all_pred_blendshapes_model2, axis=0)
    concatenated_diff_gt = np.concatenate(all_diff_gt_blendshapes, axis=0)
    concatenated_diff_pred = np.concatenate(all_diff_pred_blendshapes, axis=0)
    concatenated_diff_pred_model2 = np.concatenate(all_diff_pred_blendshapes_model2, axis=0)
    concatenated_significant_gt_movenents = np.concatenate(all_significant_gt_movenents, axis=0)
    concatenated_significant_pred_movenents = np.concatenate(all_significant_pred_movenents, axis=0)
    concatenated_significant_pred_model2_movenents = np.concatenate(all_significant_pred_model2_movenents, axis=0)
    
    print(f"Concatenated GT shape: {concatenated_gt.shape}")
    print(f"Concatenated {MODEL_NAME} Pred shape: {concatenated_pred.shape}")
    print(f"Concatenated {MODEL2_NAME} Pred shape: {concatenated_pred_model2.shape}")
    print(f"Concatenated {MODEL_NAME} Diff GT shape: {concatenated_diff_gt.shape}")
    print(f"Concatenated {MODEL_NAME} Diff Pred shape: {concatenated_diff_pred.shape}")
    print(f"Concatenated {MODEL2_NAME} Diff Pred shape: {concatenated_diff_pred_model2.shape}")
    print(f"Concatenated significant GT movenents shape: {concatenated_significant_gt_movenents.shape}")
    print(f"Concatenated significant Pred movenents shape: {concatenated_significant_pred_movenents.shape}")
    print(f"Concatenated significant Pred2 movenents shape: {concatenated_significant_pred_model2_movenents.shape}")
    
    # Calculate correlation and distance metrics on raw values
    if CALCULATE_CORRELATION_METRICS:
        print(f"\n\nCalculating correlation and distance metrics (raw values)...")
        
        if PLOT_BY_CATEGORIES:
            # Mode 1: Plot by predefined categories
            # Define all blendshape categories to iterate over
            blendshape_categories = [
                ('PROMINANT_MOUTH_BLENDSHAPES', PROMINANT_MOUTH_BLENDSHAPES),
                ('PROMINANT_EYE_BLENDSHAPES', PROMINANT_EYE_BLENDSHAPES),
                ('PROMINANT_JAW_AND_CHEEKS_BLENDSHAPES', PROMINANT_JAW_AND_CHEEKS_BLENDSHAPES),
                ('BROW_BLENDSHAPES', BROW_BLENDSHAPES)
            ]
            
            # Store metrics for all categories
            all_category_metrics = []
            
            # Iterate over each blendshape category
            for category_var_name, category_blendshapes in blendshape_categories:
                category_name = get_blendshapes_category_name(category_blendshapes)
                print(f"\n{'='*80}")
                print(f"Processing category: {category_name}")
                print(f"{'='*80}")
                
                # Calculate metrics for model 1
                metrics_model1 = calculate_correlation_and_distance_metrics(
                    concatenated_diff_gt, 
                    concatenated_diff_pred, 
                    concatenated_significant_gt_movenents,
                    concatenated_significant_pred_movenents,
                    blendshape_names=BLENDSHAPES_ORDERED,
                    use_diff=False,
                    blendshapes_to_analyze=category_blendshapes
                )
                
                # Calculate metrics for model 2
                metrics_model2 = calculate_correlation_and_distance_metrics(
                    concatenated_diff_gt, 
                    concatenated_diff_pred_model2, 
                    concatenated_significant_gt_movenents,
                    concatenated_significant_pred_model2_movenents,
                    blendshape_names=BLENDSHAPES_ORDERED,
                    use_diff=False,
                    blendshapes_to_analyze=category_blendshapes
                )
                
                # Store metrics for later plotting
                all_category_metrics.append((category_name, metrics_model1, metrics_model2))
                
                # Print metrics comparison for this category
                print_metrics_comparison(
                    metrics_model1, 
                    metrics_model2, 
                    MODEL_NAME, 
                    MODEL2_NAME, 
                    title_suffix=f" (Raw Values) - {category_name}"
                )
            
            # Plot all categories in one figure with subplots
            print(f"\n{'='*80}")
            print("Plotting all categories in subplots...")
            print(f"{'='*80}")
            # plot_all_categories_metrics(
            #     all_category_metrics,
            #     MODEL_NAME_TO_PLOT,
            #     MODEL2_NAME_TO_PLOT,
            #     title_suffix=" (Raw Values)"
            # )
            
            # Plot only velocity agreement for model 1 across all categories
            plot_velocity_agreement_all_categories(
                all_category_metrics,
                MODEL_NAME_TO_PLOT,
                title_suffix=" (Raw Values)"
            )
        else:
            # Mode 2: Plot all blendshapes above threshold
            print(f"\n{'='*80}")
            print(f"Processing all blendshapes (BLENDSHAPES_ORDERED)")
            print(f"{'='*80}")
            
            # Calculate metrics for all blendshapes
            metrics_model1 = calculate_correlation_and_distance_metrics(
                concatenated_diff_gt, 
                concatenated_diff_pred, 
                concatenated_significant_gt_movenents,
                concatenated_significant_pred_movenents,
                blendshape_names=BLENDSHAPES_ORDERED,
                use_diff=False,
                blendshapes_to_analyze=BLENDSHAPES_ORDERED
            )
            
            # Plot blendshapes above threshold
            print(f"\n{'='*80}")
            print(f"Plotting blendshapes with velocity agreement >= {VELOCITY_AGREEMENT_THRESHOLD}%...")
            print(f"{'='*80}")
            plot_velocity_agreement_above_threshold(
                metrics_model1,
                MODEL_NAME_TO_PLOT,
                threshold=VELOCITY_AGREEMENT_THRESHOLD,
                title_suffix=" (Raw Values)"
            )
    
    # Calculate correlation and distance metrics on frame-to-frame differences
    if CALCULATE_DIFF_METRICS:
        print(f"\n\nCalculating correlation and distance metrics (frame-to-frame differences)...")
        metrics_diff_model1 = calculate_correlation_and_distance_metrics(
            concatenated_gt, 
            concatenated_pred, 
            concatenated_significant_gt_movenents = None,
            blendshape_names=BLENDSHAPES_ORDERED,
            use_diff=True,
            blendshapes_to_analyze=BLENDSHAPES_TO_PLOT
        )
        metrics_diff_model2 = calculate_correlation_and_distance_metrics(
            concatenated_gt, 
            concatenated_pred_model2, 
            concatenated_significant_gt_movenents = None,
            blendshape_names=BLENDSHAPES_ORDERED,
            use_diff=True,
            blendshapes_to_analyze=BLENDSHAPES_TO_PLOT
        )
        
        # Print metrics comparison
        if False:
            print_metrics_comparison(
                metrics_diff_model1, 
                metrics_diff_model2, 
                MODEL_NAME, 
                MODEL2_NAME, 
                title_suffix=" (Frame-to-Frame Differences)"
            )
        
        # Plot metrics comparison
        plot_metrics_bar_chart(
            metrics_diff_model1,
            metrics_diff_model2,
            MODEL_NAME,
            MODEL2_NAME,
            title_suffix=" (Frame-to-Frame Differences)"
        )
        
        # Plot velocity agreement only
        plot_velocity_agreement_only(
            metrics_diff_model1,
            MODEL_NAME
        )
    
    # Create subplot figure
    print(f"\n\nGenerating plot...")
    num_plots = len(BLENDSHAPES_TO_PLOT)
    max_spacing = 1.0 / (num_plots - 1) if num_plots > 1 else 0.2
    vertical_spacing = min(0.008, max_spacing * 0.5)

    # fig = make_subplots(
    #     rows=num_plots, 
    #     cols=1, 
    #     shared_xaxes=True,
    #     subplot_titles=tuple(f"<span style='font-size:16px'>{bs}</span>" for bs in BLENDSHAPES_TO_PLOT),
    #     vertical_spacing=vertical_spacing
    # )

    # # Plot each blendshape
    # for plot_idx, bs_name in enumerate(BLENDSHAPES_TO_PLOT, 1):
    #     bs_idx = BLENDSHAPES_ORDERED.index(bs_name)
        
    #     # Get data for this blendshape
    #     gt_data = concatenated_gt[:, bs_idx]
    #     pred_data = concatenated_pred[:, bs_idx]
        
    #     # Add GT trace
    #     fig.add_trace(
    #         go.Scatter(
    #             x=np.arange(len(gt_data)),
    #             y=gt_data,
    #             name='GT',
    #             mode='lines',
    #             line=dict(color='blue', width=2),
    #             showlegend=(plot_idx == 1),
    #             legendgroup='GT'
    #         ),
    #         row=plot_idx, col=1
    #     )
        
    #     # Add prediction trace
    #     fig.add_trace(
    #         go.Scatter(
    #             x=np.arange(len(pred_data)),
    #             y=pred_data,
    #             name='Pred',
    #             mode='lines',
    #             line=dict(color='#d62728', width=2),  # Red
    #             showlegend=(plot_idx == 1),
    #             legendgroup='Pred'
    #         ),
    #         row=plot_idx, col=1
    #     )
        
    #     # Add vertical lines at file boundaries
    #     for boundary in file_boundaries:
    #         fig.add_vline(
    #             x=boundary, 
    #             line_dash="dash", 
    #             line_color="gray", 
    #             opacity=0.5,
    #             row=plot_idx, 
    #             col=1
    #         )

    # # Update layout
    # fig.update_layout(
    #     title_text=f"Blendshapes Comparison: GT vs Prediction (Right Side Only)<br>"
    #                f"<sub>{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}</sub><br>"
    #                f"<sub>Concatenated: {len(run_path_tar_id_side_tuples)} files from split</sub>",
    #     title_x=0.5,
    #     title_xanchor='center',
    #     title_font_size=20,
    #     height=300 * num_plots,
    #     hovermode='x unified',
    #     legend=dict(
    #         font=dict(size=9),
    #         yanchor="top",
    #         y=1,
    #         xanchor="right",
    #         x=-0.01,
    #         bgcolor="rgba(255, 255, 255, 0.8)",
    #         bordercolor="gray",
    #         borderwidth=1
    #     ),
    #     margin=dict(l=100)
    # )

    # fig.update_xaxes(title_text="Frame", row=num_plots, col=1)
    # fig.show()
    
    # Blink analysis
    if CALCULATE_BLINK_METRICS:
        print("\n" + "="*80)
        print("BLINK DETECTION ANALYSIS (ROC CURVE)")
        print("="*80)
        
        gt_th = 0.06
        quantizer = 8
        dense_region = np.linspace(-0.5, 6.5, 350, endpoint=True)
        # sparse_region1 = np.linspace(-10, -1, 2, endpoint=True)
        # sparse_region2 = np.linspace(-1, 0, 2, endpoint=False)
        # sparse_region3 = np.linspace(0.1, 3, 2, endpoint=True)

        th_list = np.unique(np.concatenate([dense_region])) #!!!!!!!!!!!!!!
        # th_list = np.unique(np.concatenate([sparse_region1, sparse_region2, dense_region, sparse_region3])) #!!!!!!!!!!!!!!
        # th_list = np.array([0.02, 0.05, 0.1])
        
        # Load manual GT peaks if enabled
        manual_gt_peaks = None
        if USE_MANUAL_GT_PEAKS:
            print("\n" + "="*80)
            print("LOADING MANUALLY EDITED GT PEAKS AND MANUAL BLINK LABELS")
            print("="*80)
            
            # First, try to load existing manual GT peaks from pickle file
            manual_gt_peaks = load_manual_gt_peaks(MANUAL_GT_PEAKS_FILE)
            if manual_gt_peaks is not None:
                print(f"✓ Loaded {len(manual_gt_peaks)} GT peaks from {MANUAL_GT_PEAKS_FILE}")
            else:
                print(f"  No existing manual GT peaks file found")
            
            # Then, load and add manual blink labels from CSV
            print(f"\nLoading manual blink labels from CSV...")
            manual_blink_peaks = load_manual_blink_labels_and_add_to_gt(
                MANUAL_BLINK_LABELS_CSV, 
                file_frame_mapping
            )
            
            if manual_blink_peaks is not None:
                print(f"✓ Loaded {len(manual_blink_peaks)} new GT peaks from manual blink labels CSV")
                
                # Combine with existing manual GT peaks
                if manual_gt_peaks is not None:
                    # Merge both sources
                    combined_peaks = np.concatenate([manual_gt_peaks, manual_blink_peaks])
                    combined_peaks = np.unique(combined_peaks)  # Remove duplicates and sort
                    print(f"\n✓ Combined total: {len(combined_peaks)} GT peaks")
                    print(f"  - From pickle file: {len(manual_gt_peaks)}")
                    print(f"  - From CSV: {len(manual_blink_peaks)}")
                    print(f"  - After deduplication: {len(combined_peaks)}")
                    manual_gt_peaks = combined_peaks
                else:
                    # Only CSV peaks available
                    manual_gt_peaks = manual_blink_peaks
                    print(f"✓ Using {len(manual_gt_peaks)} GT peaks from CSV only")
            else:
                print(f"  No manual blink labels found in CSV")
            
            if manual_gt_peaks is not None:
                print(f"\n✓ Using {len(manual_gt_peaks)} manually edited GT peaks for analysis")
            else:
                print(f"\n✗ No manual GT peaks found, falling back to automatic detection")
        
        # Load 'dont_know' labeled peaks to exclude from predictions
        print(f"\n" + "="*80)
        print("LOADING 'DONT_KNOW' PEAKS TO EXCLUDE FROM PREDICTIONS")
        print("="*80)
        exclude_pred_peaks = load_dont_know_peaks_to_exclude(
            MANUAL_BLINK_LABELS_CSV,
            file_frame_mapping
        )
        
        if exclude_pred_peaks is not None:
            print(f"✓ Will exclude {len(exclude_pred_peaks)} 'dont_know' peaks from prediction analysis")
        else:
            print(f"  No 'dont_know' peaks to exclude")
        
        print(f"\nRunning blink analysis for {MODEL_NAME}...")
        TPR_lst, FNR_lst, FPR_lst, blink_analysis_model = run_blink_analysis(th_list, all_gt_blendshapes, all_pred_blendshapes, quantizer, gt_th, all_significant_gt_movenents, manual_gt_peaks=manual_gt_peaks, edge_margin=EDGE_MARGIN, file_boundaries=file_boundaries, exclude_pred_peaks=exclude_pred_peaks)
        
        # Save concatenated data for peak editor (with timestamp to avoid overwriting)
        print("\n" + "="*80)
        print("SAVING CONCATENATED DATA FOR PEAK EDITOR")
        print("="*80)
        try:
            from save_concatenated_for_editing import save_for_peak_editor
            saved_file = save_for_peak_editor(
                gt_concat=blink_analysis_model['gt_concat'],
                pred_concat=blink_analysis_model['pred_concat'],
                matches=blink_analysis_model['matches'],
                output_file="concatenated_signals.pkl",
                use_timestamp=True  # Adds timestamp to avoid overwriting
            )
            print(f"✓ Concatenated data saved successfully!")
            print(f"  To edit GT peaks, run:")
            print(f"    python launch_peak_editor_from_analysis.py {saved_file}")
        except Exception as e:
            print(f"✗ Warning: Could not save concatenated data: {e}")
            print(f"  You can still manually save the data if needed.")
        
        # print(f"Running blink analysis for {MODEL2_NAME}...")
        # TPR_lst2, FNR_lst2, FPR_lst2, _ = run_blink_analysis(th_list, all_gt_blendshapes, all_pred_blendshapes_model2, quantizer, gt_th, all_significant_gt_movenents)

        # Create ROC curve using plotly
        import plotly.graph_objects as go
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=FPR_lst, y=TPR_lst, mode='lines+markers', name=MODEL_NAME, line=dict(color='blue', width=3), marker=dict(size=8)))
        # fig_roc.add_trace(go.Scatter(x=FPR_lst2, y=TPR_lst2, mode='lines+markers', name=MODEL2_NAME, line=dict(color='red', width=3), marker=dict(size=8)))
        fig_roc.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Random Classifier', line=dict(color='gray', width=2, dash='dash')))
        
        # Determine x-axis range based on actual data (with some padding)
        # max_fpr = max(max(FPR_lst), max(FPR_lst2))
        # min_fpr = min(min(FPR_lst), min(FPR_lst2))
        # fpr_range = max_fpr - min_fpr
        # x_range_max = min(100, max_fpr + 0.1 * fpr_range) if fpr_range > 0 else 100
        # x_range_min = max(0, min_fpr - 0.1 * fpr_range) if fpr_range > 0 else 0
        
        fig_roc.update_layout(
            title=dict(text=f'ROC Curve - Blink Detection', font=dict(size=42)), 
            xaxis_title=dict(text='# False blinks in minute', font=dict(size=30)), 
            # xaxis_title=dict(text='False Positive Rate (%)', font=dict(size=30)), 
            yaxis_title=dict(text='True Positive Rate (%)', font=dict(size=30)), 
            xaxis=dict(tickfont=dict(size=24)), 
            yaxis=dict(range=[0, 100], tickfont=dict(size=24)), 
            showlegend=True, 
            legend=dict(font=dict(size=24)), 
            width=1400,  # Increased width for better visibility
            height=800   # Increased height as well
        )
        # fig_roc.update_layout(title=f'ROC Curve - Blink Detection - Quantizer {quantizer}, gt_th {gt_th}', xaxis_title='False Positive Rate (%)', yaxis_title='True Positive Rate (%)', xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]), showlegend=True, width=800, height=600)
        fig_roc.show()
        
        # Save unmatched peaks for threshold that gives ~2 false positives per minute
        print("\n" + "="*80)
        print("SAVING UNMATCHED PRED PEAKS FOR THRESHOLD WITH ~2 FP/MIN")
        print("="*80)
        
        # Find threshold closest to 2 false positives per minute
        target_fpr = 2.0  # false positives per minute
        fpr_diff = np.abs(np.array(FPR_lst) - target_fpr)
        best_tpr_idx = np.argmin(fpr_diff)
        best_th = th_list[best_tpr_idx]
        best_tpr = TPR_lst[best_tpr_idx]
        best_fpr = FPR_lst[best_tpr_idx]
        
        print(f"Selected threshold: {best_th:.4f}")
        print(f"  TPR: {best_tpr:.2f}%")
        print(f"  FPR: {best_fpr:.2f} false positives per minute (target: {target_fpr})")
        
        # Re-run analysis with best threshold to get unmatched peaks
        analyzer = BlinkAnalyzer()
        best_blink_analysis = analyzer.analyze_blinks(
            gt_th=gt_th,
            model_th=best_th,
            blendshapes_list=all_gt_blendshapes,
            pred_blends_list=all_pred_blendshapes,
            max_offset=quantizer,
            significant_gt_movenents=all_significant_gt_movenents,
            manual_gt_peaks=manual_gt_peaks
        )
        
        unmatched_pred_peaks = best_blink_analysis['matches']['unmatched_pred']
        print(f"Number of unmatched pred peaks: {len(unmatched_pred_peaks)}")
        
        # Map unmatched peaks back to original files
        unmatched_peaks_data = []
        
        for peak_frame in unmatched_pred_peaks:
            # Find which file this peak belongs to
            file_idx = 0
            for i, boundary in enumerate(file_boundaries):
                if peak_frame < boundary:
                    file_idx = i
                    break
            
            # Calculate frame number within the original file
            if file_idx == 0:
                frame_in_file = peak_frame
            else:
                frame_in_file = peak_frame - file_boundaries[file_idx - 1]
            
            # Get run_path, tar_id, side for this file
            run_path, tar_id, side = run_path_tar_id_side_tuples[file_idx]
            
            # Get the peak value from the concatenated prediction signal
            peak_value = best_blink_analysis['pred_concat'][peak_frame]
            
            # Convert frame numbers from 25 fps (blendshapes) to 30 fps (video)
            # video_frame = blendshapes_frame * (30 / 25) = blendshapes_frame * 1.2
            video_frame_in_file = int(frame_in_file * 30 / 25)
            
            unmatched_peaks_data.append({
                'peak_frame_concat': peak_frame,
                'peak_frame_in_file': frame_in_file,
                'video_frame_in_file': video_frame_in_file,
                'peak_value': peak_value,
                'run_path': run_path,
                'tar_id': tar_id,
                'side': side,
                'threshold': best_th
            })
        
        # Convert to DataFrame and save
        unmatched_df = pd.DataFrame(unmatched_peaks_data)
        output_file = f"unmatched_pred_peaks_2fpm_th{best_th:.4f}.pkl"
        unmatched_df.to_pickle(output_file)
        
        print(f"✓ Saved {len(unmatched_peaks_data)} unmatched pred peaks to: {output_file}")
        print(f"\nDataFrame columns: {list(unmatched_df.columns)}")
        print(f"\nFirst few rows:")
        print(unmatched_df.head())
        
        # Plot concatenated pred signal with unmatched peaks using pre-generated plot
        print("\n" + "="*80)
        print("PLOTTING CONCATENATED PRED SIGNAL WITH UNMATCHED PEAKS")
        print("="*80)
        
        # Use the pre-generated plot from blink_analysis_model (much faster!)
        # The plot is stored in best_blink_analysis['plots']
        if 'plots' in best_blink_analysis and len(best_blink_analysis['plots']) > 0:
            # Get the plot (usually the second plot shows the concatenated signals)
            plot_items = list(best_blink_analysis['plots'].items())
            if len(plot_items) > 1:
                _, fig_pred_peaks = plot_items[1]
            else:
                _, fig_pred_peaks = plot_items[0]
            
            # Update the title to include threshold, TPR, and FPR information
            fig_pred_peaks.update_layout(
                title_text=f"Concatenated Predicted Signal with Unmatched Peaks<br>"
                          f"<sub>Pred Threshold: {best_th:.4f} | TPR: {best_tpr:.2f}% | FPR: {best_fpr:.2f} FP/min</sub><br>"
                          f"<sub>{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE} | {len(unmatched_peaks_data)} unmatched peaks</sub>",
                title_x=0.5,
                title_xanchor='center',
                title_font_size=20,
            )
            
            fig_pred_peaks.show()
        else:
            print("Warning: No plots found in blink_analysis_model")
    
    print("\n" + "="*80)
    print("ALL ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()