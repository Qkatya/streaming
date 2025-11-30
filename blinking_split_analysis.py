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
ROW_INDICES = [3663]  # Will be randomly sampled from the dataframe in main()
# ROW_INDICES = [3663, 3953, 3642, 317, 6117, 1710, 3252, 1820, 3847]
# ROW_INDICES = list(range(0, 4))#[3663, 3953, 3642, 317, 6117, 1710, 3252, 1820, 3847]

# Random sampling configuration
NUM_RANDOM_SAMPLES = None#   None  # Set to None to process all rows, or a number to randomly sample
RANDOM_SEED = 42  # Set seed for reproducibility (None for different samples each run)

# Model configuration - which prediction file to load
MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
MODEL2_NAME = 'new21_baseline_blendshapes_normalized'
# MODEL_NAME = 'causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1

# Blendshapes to plot
BLENDSHAPES_TO_PLOT = ['eyeBlinkRight', 'jawOpen', 'mouthFunnel', 'cheekPuff', 'mouthSmileRight']

# Analysis flags - control which metrics to calculate
CALCULATE_BLINK_METRICS = False# True  # ROC curve analysis for blink detection
CALCULATE_CORRELATION_METRICS = False  # PCC, L1, L2 on raw predictions
CALCULATE_DIFF_METRICS = True  # PCC, L1, L2 on frame-to-frame differences

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    P = matched_gt + unmatched_gt
    N = len(pred_concat) / quantizer - P    
    TP = matched_pred
    FP = unmatched_pred
    
    TPR = (TP / P)*100
    FNR = (100 - TPR)
    FPR = (FP / N)*100
            
    return TPR, FNR, FPR

def run_blink_analysis(th_list, blendshapes_list, pred_blends_list, quantizer, gt_th):
    TPR_lst = []
    FNR_lst = []
    FPR_lst = []
    for model_th in th_list:
        analyzer = BlinkAnalyzer()
        blink_analysis_model = analyzer.analyze_blinks(gt_th=gt_th,model_th=model_th, blendshapes_list=blendshapes_list, pred_blends_list=pred_blends_list, max_offset=quantizer)
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
        
    return TPR_lst, FNR_lst, FPR_lst

def velocity_agreement(gt_bs, pred_bs, already_diff=False):
    """Calculate velocity agreement metrics between ground truth and prediction for a single blendshape.
    
    Args:
        gt_bs: Ground truth for one blendshape (1D array)
        pred_bs: Prediction for one blendshape (1D array)
        already_diff: If True, data is already differenced, don't apply savgol and diff again
        
    Returns:
        sign_match: Percentage of time velocity has same sign (0-1)
        r: Pearson correlation of velocities
    """
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
            
def calculate_correlation_and_distance_metrics(gt, pred, blendshape_names=None, use_diff=False, blendshapes_to_analyze=None):
    """Calculate PCC, L1, L2, and direction agreement metrics for GT vs Pred.
    
    Args:
        gt: Ground truth array of shape (num_frames, num_blendshapes)
        pred: Prediction array of shape (num_frames, num_blendshapes)
        blendshape_names: Optional list of ALL blendshape names (for indexing)
        use_diff: If True, calculate metrics on frame-to-frame differences instead of raw values
        blendshapes_to_analyze: Optional list of specific blendshape names to analyze. If None, analyze all.
        
    Returns:
        Dictionary containing:
            - global_pcc: Pearson correlation coefficient across all values
            - global_l1: Mean absolute error (L1 distance)
            - global_l2: Root mean squared error (L2 distance)
            - global_direction_agreement: % of time movement is in same direction (only for diff mode)
            - per_blendshape_pcc: PCC for each blendshape
            - per_blendshape_l1: L1 for each blendshape
            - per_blendshape_l2: L2 for each blendshape
            - per_blendshape_direction_agreement: Direction agreement for each blendshape (only for diff mode)
            - analyzed_blendshapes: List of blendshape names that were analyzed
    """
    assert gt.shape == pred.shape, f"GT and Pred shapes must match: {gt.shape} vs {pred.shape}"
    
    # Determine which blendshapes to analyze
    if blendshapes_to_analyze is not None and blendshape_names is not None:
        # Get indices of blendshapes to analyze
        blendshape_indices = [blendshape_names.index(bs) for bs in blendshapes_to_analyze if bs in blendshape_names]
        analyzed_names = [blendshape_names[i] for i in blendshape_indices]
        
        # Filter GT and Pred to only include selected blendshapes
        gt = gt[:, blendshape_indices]
        pred = pred[:, blendshape_indices]
        # plot_gt_pred_time(gt, pred)
        
        print(f"  Analyzing {len(blendshape_indices)} blendshapes: {analyzed_names}")
    else:
        analyzed_names = blendshape_names if blendshape_names is not None else None
        print(f"  Analyzing all {gt.shape[1]} blendshapes")
    
    # Calculate frame-to-frame differences if requested
    if use_diff:
        gt = np.diff(gt, axis=0)
        pred = np.diff(pred, axis=0)
        print(f"  Using frame-to-frame differences: shape {gt.shape}")
    
    # Global metrics (flatten all values)
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    global_pcc, _ = pearsonr(gt_flat, pred_flat)
    global_l1 = np.mean(np.abs(gt_flat - pred_flat))
    global_l2 = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
    
    # Direction agreement (only meaningful for differences)
    global_direction_agreement = None
    if use_diff:
        # Calculate percentage of time both GT and Pred have the same sign (direction)
        same_direction = np.sign(gt_flat) == np.sign(pred_flat)
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
        gt_bs = gt[:, i]
        pred_bs = pred[:, i]
        
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
        if use_diff:
            same_direction = np.sign(gt_bs) == np.sign(pred_bs)
            direction_agreement = np.mean(same_direction) * 100  # Convert to percentage
            per_blendshape_direction_agreement.append(direction_agreement)
        
        # Velocity agreement with Savitzky-Golay filtering
        # If already using diff, pass the diff data directly; otherwise compute from original
        if use_diff:
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
    
    # Prepare data for plotting
    models = [model1_name, model2_name]
    
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
                legendgroup=bs_name
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
                legendgroup=bs_name
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
                legendgroup=bs_name
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
                legendgroup=bs_name
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
                legendgroup=bs_name
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
                legendgroup=bs_name
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
                    legendgroup=bs_name
                ),
                row=2, col=3
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"Per-Blendshape Metrics Comparison{title_suffix}",
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
        
        # Store data
        all_gt_blendshapes.append(gt_blendshapes)
        all_pred_blendshapes.append(pred_blendshapes)
        all_pred_blendshapes_model2.append(pred_blendshapes_model2)
        current_frame += gt_blendshapes.shape[0]
        file_boundaries.append(current_frame)
    
    # Remove last boundary (end of last file)
    if file_boundaries:
        file_boundaries = file_boundaries[:-1]
    
    # Concatenate all data
    print(f"\n\nConcatenating all files...")
    concatenated_gt = np.concatenate(all_gt_blendshapes, axis=0)
    concatenated_pred = np.concatenate(all_pred_blendshapes, axis=0)
    concatenated_pred_model2 = np.concatenate(all_pred_blendshapes_model2, axis=0)
    
    print(f"Concatenated GT shape: {concatenated_gt.shape}")
    print(f"Concatenated {MODEL_NAME} Pred shape: {concatenated_pred.shape}")
    print(f"Concatenated {MODEL2_NAME} Pred shape: {concatenated_pred_model2.shape}")
    
    # Calculate correlation and distance metrics on raw values
    if CALCULATE_CORRELATION_METRICS:
        print(f"\n\nCalculating correlation and distance metrics (raw values)...")
        metrics_model1 = calculate_correlation_and_distance_metrics(
            concatenated_gt, 
            concatenated_pred, 
            blendshape_names=BLENDSHAPES_ORDERED,
            use_diff=False,
            blendshapes_to_analyze=BLENDSHAPES_TO_PLOT
        )
        metrics_model2 = calculate_correlation_and_distance_metrics(
            concatenated_gt, 
            concatenated_pred_model2, 
            blendshape_names=BLENDSHAPES_ORDERED,
            use_diff=False,
            blendshapes_to_analyze=BLENDSHAPES_TO_PLOT
        )
        
        # Print metrics comparison
        print_metrics_comparison(
            metrics_model1, 
            metrics_model2, 
            MODEL_NAME, 
            MODEL2_NAME, 
            title_suffix=" (Raw Values)"
        )
        
        # Plot metrics comparison
        plot_metrics_bar_chart(
            metrics_model1,
            metrics_model2,
            MODEL_NAME,
            MODEL2_NAME,
            title_suffix=" (Raw Values)"
        )
    
    # Calculate correlation and distance metrics on frame-to-frame differences
    if CALCULATE_DIFF_METRICS:
        print(f"\n\nCalculating correlation and distance metrics (frame-to-frame differences)...")
        metrics_diff_model1 = calculate_correlation_and_distance_metrics(
            concatenated_gt, 
            concatenated_pred, 
            blendshape_names=BLENDSHAPES_ORDERED,
            use_diff=True,
            blendshapes_to_analyze=BLENDSHAPES_TO_PLOT
        )
        metrics_diff_model2 = calculate_correlation_and_distance_metrics(
            concatenated_gt, 
            concatenated_pred_model2, 
            blendshape_names=BLENDSHAPES_ORDERED,
            use_diff=True,
            blendshapes_to_analyze=BLENDSHAPES_TO_PLOT
        )
        
        # Print metrics comparison
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
        dense_region = np.linspace(0, 0.1, 300, endpoint=True)
        sparse_region1 = np.linspace(-10, -1, 30, endpoint=True)
        sparse_region2 = np.linspace(-1, 0, 50, endpoint=False)
        sparse_region3 = np.linspace(0.1, 3, 40, endpoint=True)

        th_list = np.unique(np.concatenate([sparse_region1, sparse_region2, dense_region, sparse_region3]))
        # th_list = np.array([0.02, 0.05, 0.1])
        
        print(f"Running blink analysis for {MODEL_NAME}...")
        TPR_lst, FNR_lst, FPR_lst = run_blink_analysis(th_list, all_gt_blendshapes, all_pred_blendshapes, quantizer, gt_th)
        
        print(f"Running blink analysis for {MODEL2_NAME}...")
        TPR_lst2, FNR_lst2, FPR_lst2 = run_blink_analysis(th_list, all_gt_blendshapes, all_pred_blendshapes_model2, quantizer, gt_th)

        # Create ROC curve using plotly
        import plotly.graph_objects as go
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=FPR_lst, y=TPR_lst, mode='lines+markers', name=MODEL_NAME, line=dict(color='blue', width=3), marker=dict(size=8)))
        fig_roc.add_trace(go.Scatter(x=FPR_lst2, y=TPR_lst2, mode='lines+markers', name=MODEL2_NAME, line=dict(color='red', width=3), marker=dict(size=8)))
        fig_roc.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Random Classifier', line=dict(color='gray', width=2, dash='dash')))
        fig_roc.update_layout(title=dict(text=f'ROC Curve - Blink Detection', font=dict(size=30)), xaxis_title=dict(text='False Positive Rate (%)', font=dict(size=20)), yaxis_title=dict(text='True Positive Rate (%)', font=dict(size=20)), xaxis=dict(range=[0, 100], tickfont=dict(size=16)), yaxis=dict(range=[0, 100], tickfont=dict(size=16)), showlegend=True, legend=dict(font=dict(size=18)), width=800, height=600)
        # fig_roc.update_layout(title=f'ROC Curve - Blink Detection - Quantizer {quantizer}, gt_th {gt_th}', xaxis_title='False Positive Rate (%)', yaxis_title='True Positive Rate (%)', xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]), showlegend=True, width=800, height=600)
        fig_roc.show()
    
    print("\n" + "="*80)
    print("ALL ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()