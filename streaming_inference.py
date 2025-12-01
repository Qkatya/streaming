"""
Sliding Window Offline Inference Simulator

This script simulates streaming inference by:
1. Loading input features from files
2. Dividing them into sliding windows with configurable history and lookahead
3. Running inference on each window through one or more models
4. Reconstructing the output by taking chunks from each window
5. Plotting ground truth vs predictions with different lookaheads
"""

import os
import sys
import glob
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import shared utilities
from blendshapes_utils import (
    BLENDSHAPES_ORDERED,
    init_fairseq,
    load_fairseq_model,
    load_nemo_model,
    infer_fairseq_model,
    infer_nemo_model,
    unnormalize_blendshapes,
    load_ground_truth_blendshapes,
    find_zip_file_id,
    prepare_fairseq_sample
)

pio.renderers.default = "browser"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Paths
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
FEATURES_PATH = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")
INFERENCE_OUTPUTS_PATH = Path("inference_outputs")  # Output folder for predictions

# Sliding window parameters
HISTORY_SIZES = [800]  # List of history sizes to test
CHUNK_SIZE = 24   # Number of frames to extract from each window (the "present")
LOOKAHEAD_SIZES = [1]#, 50]#, 20, 30]  # List of lookahead sizes to test

# Model configurations
# Each entry: {'path': str, 'type': 'fairseq' or 'nemo', 'name': str, 'blendshape_indices': list, 'color': str}
MODEL_CONFIGS = [
    {
        'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_loud/0/checkpoints/checkpoint_last.pt',
        'type': 'fairseq',
        'name': 'blendshapes_loud',
        'blendshape_indices': list(range(1, 52)),
        'color': '#ff7f0e',  # Orange
    },
    # {
    #     'path': '/mnt/ML/TrainResults/ido.kazma/D2V/V2/2025_04_15/new21_baseline_blendshapes_normalized/0/checkpoints/checkpoint_last.pt',
    #     'type': 'fairseq',
    #     'name': 'new21_baseline_blendshapes_normalized',
    #     'blendshape_indices': list(range(1, 52)),
    #     'color': '#d62728',  # Red'
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_blendshapes_heads_only/checkpoints/causal_fastconformer_layernorm_landmarks_blendshapes_heads_only.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_fastconformer_layernorm_landmarks_blendshapes_heads_only',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'color': '#2ca02c',  # Green
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides/checkpoints/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_fastconformer_landmarks_all_blendshapes_two_sides',
    #     'blendshape_indices': list(range(1, 52)),
    #     'color': '#9467bd',  # Purple
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/fastconformer_blendshapes_landmarks/checkpoints/fastconformer_blendshapes_landmarks.nemo',
    #     'type': 'nemo',
    #     'name': 'fastconformer_blendshapes_landmarks',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'color': '#8c564b',  # Brown
    # },

    {
        'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side/checkpoints/causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side.nemo',
        'type': 'nemo',
        'name': 'causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side',
        'blendshape_indices': [2, 3, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 26, 27, 29, 31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 45, 47, 49, 51],
        'color': '#e377c2',  # Pink
    },
    # { 
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_preprocessor_encoder_with_smile/checkpoints/causal_preprocessor_encoder_with_smile.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_preprocessor_encoder_with_smile',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 31, 32, 38, 45],
    #     'color': '#17becf',  # Cyan
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_mae_fastconformer_layernorm_landmarks_blendshapes_heads_only/checkpoints/causal_mae_fastconformer_layernorm_landmarks_blendshapes_heads_only.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_mae_fastconformer_layernorm_landmarks_blendshapes_heads_only',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'color': '#e377c2',  # Pink
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_mae_preprocessor_encoder_with_smile/checkpoints/causal_mae_preprocessor_encoder_with_smile.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_mae_preprocessor_encoder_with_smile',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'color': '#9467bd',  # Purple
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/quartznet_landmarks_blendshapes/checkpoints/quartznet_landmarks_blendshapes.nemo',
    #     'type': 'nemo',
    #     'name': 'quartznet_landmarks_blendshapes',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'color': '#8c564b',  # Brown
    # },
]

# Device configuration
DEVICE = 'cuda:0'
USE_FP16 = True

# Normalization factors for fairseq models
BLENDSHAPES_NORMALIZE_PATH = '/home/ido.kazma/projects/notebooks-qfairseq/stats_df.pkl'

# Input files/folders to process
# Can be:
# - Single run_path string: "2025/11/06/KatyaIvantsiv-143719/1_0_38909aea-68df-4630-b952-13f638262f9c_loud"
# - List of run_paths
# - Path to a folder containing run_paths
INPUT_PATHS = [
    # '2025/05/13/QaQa-155207/7_0_793098ac-498d-4c40-94b9-a372fbeed985_loud',
    # '2025/05/13/QaQa-155207/10_0_defae8c0-5238-477e-8a93-0d608732518c_silent',
    # '2025/05/13/QaQa-155207/16_0_1e9e99b8-3a6e-413f-887b-4e18102cd374_silent',
    # '2025/05/13/QaQa-155207/24_0_831543c7-3343-4cb1-9f17-a32f4e8e5ff0_silent']
    
    "2025/11/06/KatyaIvantsiv-143719/6_0_6f61c83b-ed81-4f81-8efe-3898cd4475c2_lip", # smiling
    "2025/11/06/KatyaIvantsiv-143719/9_0_aab0fc3a-57c9-4ef5-8da6-46dd7e9df350_loud", # blinking
    "2025/11/06/KatyaIvantsiv-143719/6_1_6f61c83b-ed81-4f81-8efe-3898cd4475c2_lip", # mouth opening
    "2025/11/06/KatyaIvantsiv-143719/4_1_38909aea-68df-4630-b952-13f638262f9c_silent"]# count 1-10
    
    # "2025/03/19/NuggetCitadel-113505/88_0_6ba7722d-2f2a-4ad6-9c3e-971cc869dcd1_loud"] # eyes move to sides
    # "2025/11/06/KatyaIvantsiv-143719/8_0_6f61c83b-ed81-4f81-8efe-3898cd4475c2_silent",# mouth move left and right
    # "2025/11/06/KatyaIvantsiv-143719/7_0_6f61c83b-ed81-4f81-8efe-3898cd4475c2_soft",] # duck face and in the end u
    # "2025/11/06/KatyaIvantsiv-143719/10_0_aab0fc3a-57c9-4ef5-8da6-46dd7e9df350_lip", # smiling    # "2025/11/06/KatyaIvantsiv-143719/5_0_6f61c83b-ed81-4f81-8efe-3898cd4475c2_loud",
    # "2025/11/06/KatyaIvantsiv-143719/4_0_38909aea-68df-4630-b952-13f638262f9c_silent", blinking and then close eyes strong
# ]

# Blendshapes to plot
BLENDSHAPES_TO_PLOT = [
    '_neutral',  'browDownRight', 'browInnerUp', 'browOuterUpRight', 'cheekPuff', 'cheekSquintRight', 
    'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
    'eyeSquintRight',  'eyeWideRight', 'jawForward', 'jawOpen', 'jawRight', 'mouthClose',  'mouthDimpleRight', 
    'mouthFrownRight', 'mouthFunnel',  'mouthLowerDownRight', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 
    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileRight', 'mouthStretchRight', 'mouthUpperUpRight', 'noseSneerRight'
]
# BLENDSHAPES_TO_PLOT = ['eyeBlinkRight', 'jawOpen', 'mouthFunnel', 'cheekPuff', 'mouthSmileRight']

# Visualization options
SHOW_WINDOW_VISUALIZATION = False  # Set to True to visualize sliding window division for first file

# Line styles for history-lookahead combinations
# We'll use a combination of dash styles to distinguish different configurations
LINE_STYLES = [None, 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot', (5, 2, 1, 2), (5, 1)]

# Default color for models without a specified color
DEFAULT_MODEL_COLOR = '#17becf'  # Cyan

# Note: Model loading and inference functions are now imported from blendshapes_utils

# ============================================================================
# FILE HANDLING FUNCTIONS
# ============================================================================

def load_input_features(run_path: str) -> Optional[np.ndarray]:
    """Load input features from .npy file."""
    full_run_path = ALL_DATA_PATH / run_path
    
    # Find the zip file ID
    zip_id = find_zip_file_id(full_run_path)
    if zip_id is None:
        print(f"Warning: No .right.zip file found in {full_run_path}")
        return None
    
    # Load features
    features_file = FEATURES_PATH / run_path / f"{zip_id}.npy"
    if not features_file.exists():
        print(f"Warning: Features file not found: {features_file}")
        return None
    
    return np.load(features_file)

def load_gt_blendshapes(run_path: str) -> Optional[np.ndarray]:
    """Load ground truth blendshapes from landmarks_and_blendshapes.npz."""
    full_run_path = ALL_DATA_PATH / run_path
    
    try:
        return load_ground_truth_blendshapes(full_run_path, downsample=True)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None

def get_run_paths(input_paths) -> List[str]:
    """Convert input paths specification to list of run_paths."""
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    
    run_paths = []
    for path in input_paths:
        full_path = ALL_DATA_PATH / path
        if full_path.is_dir():
            # If it's a directory, find all subdirectories with .right.zip files
            for zip_file in full_path.rglob("*.right.zip"):
                run_path = str(zip_file.parent.relative_to(ALL_DATA_PATH))
                run_paths.append(run_path)
        else:
            # Assume it's a run_path string
            run_paths.append(path)
    
    return run_paths

def save_predictions(run_path: str, predictions: Dict[str, Dict], output_base_path: Path):
    """
    Save predictions for a single run_path.
    
    Args:
        run_path: The run path (e.g., "2025/11/06/KatyaIvantsiv-143719/9_0_aab0fc3a-57c9-4ef5-8da6-46dd7e9df350_loud")
        predictions: Dict of {pred_key: {'pred': array, 'history': int, 'lookahead': int, 'blendshape_indices': list}}
                     where pred_key is "{model_name}_H{history}_LA{lookahead}"
        output_base_path: Base path for output (e.g., Path("inference_outputs"))
    """
    # Create output directory for this run_path
    output_dir = output_base_path / run_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each model's predictions
    for pred_key, pred_info in predictions.items():
        pred_array = pred_info['pred']
        
        # pred_key already contains the full name with H and LA, so use it directly
        output_file = output_dir / f"pred_right_{pred_key}.npy"
        np.save(output_file, pred_array)
        print(f"    Saved {output_file} (shape: {pred_array.shape})")
    
    return output_dir

# ============================================================================
# SLIDING WINDOW LOGIC
# ============================================================================

def create_sliding_windows(input_features: np.ndarray, history_size: int, chunk_size: int, lookahead_size: int) -> List[Tuple[int, int, int, int]]:
    """
    Create sliding window specifications.
    
    Returns list of tuples: (window_start, window_end, extract_start, extract_end)
    where extract_start and extract_end are relative to the window.
    """
    total_frames = len(input_features)
    windows = []
    
    # Start position for extraction (after history)
    current_pos = 0
    
    while current_pos < total_frames:
        # Window boundaries
        window_start = max(0, current_pos - history_size)
        window_end = min(total_frames, current_pos + chunk_size + lookahead_size)
        
        # Extract boundaries (relative to window)
        extract_start = current_pos - window_start
        extract_end = min(extract_start + chunk_size, window_end - window_start)
        
        # Actual frames to extract (absolute positions)
        actual_extract_start = current_pos
        actual_extract_end = min(current_pos + chunk_size, total_frames)
        
        if actual_extract_start >= total_frames:
            break
            
        windows.append((window_start, window_end, extract_start, extract_end))
        
        # Move to next chunk
        current_pos += chunk_size
    
    return windows

def reconstruct_output_from_windows(window_outputs: List[np.ndarray], windows: List[Tuple[int, int, int, int]], total_frames: int) -> np.ndarray:

    if len(window_outputs) == 0:
        return np.array([])
    
    # Simply concatenate all window outputs along the time axis
    reconstructed = np.concatenate(window_outputs, axis=0)
    
    return reconstructed

# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

def run_sliding_window_inference(
    input_features: np.ndarray,
    model,
    model_config: Dict,
    history_size: int,
    chunk_size: int,
    lookahead_size: int,
    device: str,
    normalization_factors=None
) -> np.ndarray:
    """
    Run inference using sliding windows and reconstruct output.
    """
    # Create windows
    windows = create_sliding_windows(input_features, history_size, chunk_size, lookahead_size)
    
    # Run inference on each window
    
    ## open figure
    window_outputs = []
    for window_start, window_end, extract_start, extract_end in windows:
        # Extract window data
        window_data = input_features[window_start:window_end]
        
        # Run inference based on model type
        if model_config['type'] == 'fairseq':
            # Prepare sample for fairseq
            sample, padding_mask = prepare_fairseq_sample(window_data, device)
            ## plot sample like this: plot the sample in sample(:,0,0,0) as a line
            ## next sample will be plotted underneath the first one
            blendshapes = infer_fairseq_model(model, sample, padding_mask, model_config['blendshape_indices'])
        elif model_config['type'] == 'nemo':
            blendshapes = infer_nemo_model(model, window_data, device, model_config['blendshape_indices'])
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        # Unnormalize
        blendshapes_unnorm = unnormalize_blendshapes(
            blendshapes, 
            model_config['type'], 
            model_config['blendshape_indices'],
            normalization_factors
        )
        
        # print(blendshapes_unnorm.shape) 
        # print(extract_start//8)
        # print(extract_end//8)
        window_outputs.append(blendshapes_unnorm[extract_start//8:extract_end//8])
    
    # Reconstruct full output
    reconstructed = reconstruct_output_from_windows(window_outputs, windows, len(input_features))
    
    return reconstructed

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_sliding_windows_visualization(
    total_frames: int,
    windows: List[Tuple[int, int, int, int]],
    history_size: int,
    chunk_size: int,
    lookahead_size: int,
    run_path: str = ""
):
    """
    Visualize how the input data is divided into sliding windows.
    
    Args:
        total_frames: Total number of frames in the input
        windows: List of window specifications (window_start, window_end, extract_start, extract_end)
        history_size: Size of history context
        chunk_size: Size of chunk to extract
        lookahead_size: Size of lookahead context
        run_path: Path being processed (for title)
    """
    fig = go.Figure()
    
    # Color scheme
    HISTORY_COLOR = 'rgba(100, 100, 255, 0.3)'  # Blue for history
    CHUNK_COLOR = 'rgba(255, 100, 100, 0.6)'    # Red for extracted chunk
    LOOKAHEAD_COLOR = 'rgba(100, 255, 100, 0.3)' # Green for lookahead
    
    # Plot each window
    for idx, (window_start, window_end, extract_start, extract_end) in enumerate(windows):
        # Calculate absolute positions
        abs_extract_start = window_start + extract_start
        abs_extract_end = window_start + extract_end
        
        # History region (from window_start to extract_start)
        if extract_start > 0:
            fig.add_trace(go.Scatter(
                x=[window_start, window_start + extract_start, window_start + extract_start, window_start, window_start],
                y=[idx, idx, idx + 0.8, idx + 0.8, idx],
                fill='toself',
                fillcolor=HISTORY_COLOR,
                line=dict(color='blue', width=1),
                mode='lines',
                name='History' if idx == 0 else None,
                legendgroup='history',
                showlegend=(idx == 0),
                hovertemplate=f'Window {idx}<br>History: [{window_start}, {abs_extract_start})<br>Frames: {extract_start}<extra></extra>'
            ))
        
        # Chunk region (extracted part)
        fig.add_trace(go.Scatter(
            x=[abs_extract_start, abs_extract_end, abs_extract_end, abs_extract_start, abs_extract_start],
            y=[idx, idx, idx + 0.8, idx + 0.8, idx],
            fill='toself',
            fillcolor=CHUNK_COLOR,
            line=dict(color='red', width=2),
            mode='lines',
            name='Extracted Chunk' if idx == 0 else None,
            legendgroup='chunk',
            showlegend=(idx == 0),
            hovertemplate=f'Window {idx}<br>Chunk: [{abs_extract_start}, {abs_extract_end})<br>Frames: {abs_extract_end - abs_extract_start}<extra></extra>'
        ))
        
        # Lookahead region (from extract_end to window_end)
        if abs_extract_end < window_end:
            fig.add_trace(go.Scatter(
                x=[abs_extract_end, window_end, window_end, abs_extract_end, abs_extract_end],
                y=[idx, idx, idx + 0.8, idx + 0.8, idx],
                fill='toself',
                fillcolor=LOOKAHEAD_COLOR,
                line=dict(color='green', width=1),
                mode='lines',
                name='Lookahead' if idx == 0 else None,
                legendgroup='lookahead',
                showlegend=(idx == 0),
                hovertemplate=f'Window {idx}<br>Lookahead: [{abs_extract_end}, {window_end})<br>Frames: {window_end - abs_extract_end}<extra></extra>'
            ))
        
        # Add window boundary box
        fig.add_trace(go.Scatter(
            x=[window_start, window_end, window_end, window_start, window_start],
            y=[idx, idx, idx + 0.8, idx + 0.8, idx],
            mode='lines',
            line=dict(color='black', width=1, dash='dot'),
            name='Window Boundary' if idx == 0 else None,
            legendgroup='boundary',
            showlegend=(idx == 0),
            hovertemplate=f'Window {idx}<br>Full window: [{window_start}, {window_end})<br>Total frames: {window_end - window_start}<extra></extra>'
        ))
    
    # Add vertical line at the end of data
    fig.add_vline(
        x=total_frames,
        line_dash="dash",
        line_color="black",
        annotation_text="End of data",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Sliding Window Visualization<br>"
                 f"<sub>History: {history_size}, Chunk: {chunk_size}, Lookahead: {lookahead_size}</sub><br>"
                 f"<sub>{run_path}</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Frame Index",
        yaxis_title="Window Number",
        height=max(400, len(windows) * 30 + 100),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    fig.update_yaxes(
        tickmode='linear',
        tick0=0.4,
        dtick=1,
        ticktext=[f"W{i}" for i in range(len(windows))],
        tickvals=[i + 0.4 for i in range(len(windows))]
    )
    
    fig.show()

def plot_comparison(
    gt_blendshapes: np.ndarray,
    predictions: Dict[str, Dict[int, Dict[int, np.ndarray]]],
    run_path: str,
    model_configs: List[Dict] = None,
    blendshapes_to_plot: List[str] = None,
    file_boundaries: List[int] = None
):
    """
    Plot ground truth vs predictions for multiple models, history sizes, and lookaheads.
    
    Args:
        gt_blendshapes: Ground truth blendshapes array
        predictions: Dict of {model_name: {history: {lookahead: prediction_array}}}
        run_path: Path being processed (for title)
        model_configs: List of model configuration dicts (to get colors)
        blendshapes_to_plot: List of blendshape names to plot
        file_boundaries: List of frame indices where files change (for vertical lines)
    """
    if blendshapes_to_plot is None:
        blendshapes_to_plot = BLENDSHAPES_TO_PLOT
    
    # Create short model name mapping and color mapping from configs
    model_name_map = {}
    model_color_map = {}
    for idx, model_name in enumerate(predictions.keys(), 1):
        # Create short code name like M1, M2, etc.
        model_name_map[model_name] = f"M{idx}"
        
        # Get color from model config
        if model_configs:
            for config in model_configs:
                if config['name'] == model_name:
                    model_color_map[model_name] = config.get('color', DEFAULT_MODEL_COLOR)
                    break
        if model_name not in model_color_map:
            model_color_map[model_name] = DEFAULT_MODEL_COLOR
    
    num_plots = len(blendshapes_to_plot)
    # Calculate appropriate vertical spacing based on number of plots
    # Maximum spacing is 1/(rows-1), we use a smaller percentage for tighter spacing
    max_spacing = 1.0 / (num_plots - 1) if num_plots > 1 else 0.02
    vertical_spacing = min(0.008, max_spacing * 0.5)  # Reduced from 0.05 and 0.8 for tighter spacing
    
    fig = make_subplots(
        rows=num_plots, 
        cols=1, 
        shared_xaxes=True,
        subplot_titles=tuple(f"<span style='font-size:16px'>{bs}</span>" for bs in blendshapes_to_plot),
        vertical_spacing=vertical_spacing
    )
    
    # Process each blendshape
    for plot_idx, bs_name in enumerate(blendshapes_to_plot, 1):
        bs_idx = BLENDSHAPES_ORDERED.index(bs_name)
        gt_data = gt_blendshapes[:, bs_idx]
        
        # Add GT trace (show legend on all plots)
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
        
        # Add prediction traces for each model, history, and lookahead
        model_idx = 0
        for model_name, history_preds in predictions.items():
            # Each model gets its color from the config
            color = model_color_map[model_name]
            short_name = model_name_map[model_name]
            
            # Create a flat list of (history, lookahead) combinations for consistent line style assignment
            combo_idx = 0
            for history in sorted(history_preds.keys()):
                for lookahead in sorted(history_preds[history].keys()):
                    pred = history_preds[history][lookahead]
                    
                    # Align lengths
                    min_len = min(len(gt_data), len(pred))
                    pred_data = pred[:min_len, bs_idx]
                    
                    # Each history-lookahead combination gets a unique line style
                    line_style = LINE_STYLES[combo_idx % len(LINE_STYLES)]
                    
                    line_dict = dict(color=color, width=2)
                    if line_style:
                        line_dict['dash'] = line_style
                    
                    # Use short name with history and lookahead
                    trace_name = f"{short_name}_H{history}_LA{lookahead}"
                    legend_group = f"{short_name}_H{history}_LA{lookahead}"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=np.arange(len(pred_data)),
                            y=pred_data,
                            name=trace_name,
                            mode='lines',
                            line=line_dict,
                            showlegend=(plot_idx == 1),
                            legendgroup=legend_group
                        ),
                        row=plot_idx, col=1
                    )
                    
                    combo_idx += 1
            
            model_idx += 1
    
    # Add vertical lines at file boundaries
    if file_boundaries:
        for boundary in file_boundaries:
            for plot_idx in range(1, num_plots + 1):
                fig.add_vline(
                    x=boundary, 
                    line_dash="dash", 
                    line_color="gray", 
                    opacity=0.5,
                    row=plot_idx, 
                    col=1
                )
    
    # Update layout
    fig.update_layout(
        title_text=f"Sliding Window Inference Comparison<br><sub>{run_path}</sub>",
        title_x=0.5,
        title_xanchor='center',
        title_font_size=20,
        height=150 * num_plots,  # Height per subplot in pixels
        hovermode='x unified',
        legend=dict(
            font=dict(size=9),
            yanchor="top",
            y=1,
            xanchor="right",
            x=-0.01,  # Position on the left side
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=100)  # Add left margin for legend
    )
    
    fig.update_xaxes(title_text="Frame", row=num_plots, col=1)
    fig.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Initialize fairseq
    init_fairseq()
    
    # Load normalization factors for fairseq models
    print("\n" + "="*80)
    print("LOADING NORMALIZATION FACTORS")
    print("="*80)
    print(f"Loading normalization factors from {BLENDSHAPES_NORMALIZE_PATH}")
    with open(BLENDSHAPES_NORMALIZE_PATH, 'rb') as f:
        blendshape_normalization_factors = pickle.load(f)
    print("Normalization factors loaded successfully")
    
    # Load all models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    models = []
    for config in MODEL_CONFIGS:
        print(f"\nLoading {config['name']}...")
        if config['type'] == 'fairseq':
            model, saved_cfg, task = load_fairseq_model(config['path'], DEVICE, USE_FP16)
            models.append({'model': model, 'config': config, 'saved_cfg': saved_cfg, 'task': task})
        elif config['type'] == 'nemo':
            model = load_nemo_model(config['path'], DEVICE, USE_FP16)
            models.append({'model': model, 'config': config})
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
    
    # Get run paths to process
    run_paths = get_run_paths(INPUT_PATHS)
    print(f"\n" + "="*80)
    print(f"PROCESSING {len(run_paths)} FILES")
    print("="*80)
    
    # Collect data from all files
    all_gt_blendshapes = []
    all_predictions = {
        model_info['config']['name']: {
            hist: {la: [] for la in LOOKAHEAD_SIZES} for hist in HISTORY_SIZES
        } for model_info in models
    }
    file_boundaries = []
    current_frame = 0
    
    # Flag to show window visualization only for first file
    show_window_viz = SHOW_WINDOW_VISUALIZATION
    
    # Process each file
    for run_path in tqdm(run_paths, desc="Processing files"):
        print(f"\n\nProcessing: {run_path}")
        
        # Load input features
        input_features = load_input_features(run_path)
        if input_features is None:
            print(f"Skipping {run_path} - could not load features")
            continue
        
        # Load ground truth
        gt_blendshapes = load_gt_blendshapes(run_path)
        if gt_blendshapes is None:
            print(f"Skipping {run_path} - could not load ground truth")
            continue
        
        print(f"Input features shape: {input_features.shape}")
        print(f"Ground truth shape: {gt_blendshapes.shape}")
        
        # Visualize sliding windows for first file and each history/lookahead combination
        if show_window_viz:
            print("\n  Generating sliding window visualizations...")
            for history in HISTORY_SIZES:
                for lookahead in LOOKAHEAD_SIZES:
                    windows = create_sliding_windows(input_features, history, CHUNK_SIZE, lookahead)
                    print(f"    History {history}, Lookahead {lookahead}: {len(windows)} windows")
                    plot_sliding_windows_visualization(
                        len(input_features),
                        windows,
                        history,
                        CHUNK_SIZE,
                        lookahead,
                        run_path
                    )
            show_window_viz = False  # Only show for first file
        
        # Dictionary to store predictions for this file (for saving)
        # Format: {f"{model_name}_H{history}_LA{lookahead}": {'pred': array, 'history': int, 'lookahead': int, 'blendshape_indices': list}}
        file_predictions = {}
        
        # Run inference for all models, history sizes, and lookaheads
        for model_info in models:
            config = model_info['config']
            model = model_info['model']
            model_name = config['name']
            
            print(f"\n  Running inference with {model_name}...")
            
            for history in HISTORY_SIZES:
                for lookahead in LOOKAHEAD_SIZES:
                    print(f"    History: {history}, Lookahead: {lookahead}")
                    pred = run_sliding_window_inference(
                        input_features,
                        model,
                        config,
                        history,
                        CHUNK_SIZE,
                        lookahead,
                        DEVICE,
                        blendshape_normalization_factors
                    )
                    print(f"      Output shape: {pred.shape}")
                    all_predictions[model_name][history][lookahead].append(pred)
                    
                    # Store prediction for this file with all history/lookahead combinations
                    pred_key = f"{model_name}_H{history}_LA{lookahead}"
                    file_predictions[pred_key] = {
                        'pred': pred,
                        'history': history,
                        'lookahead': lookahead,
                        'blendshape_indices': config['blendshape_indices']
                    }
        
        # Save predictions for this file
        print(f"\n  Saving predictions for {run_path}...")
        save_predictions(run_path, file_predictions, INFERENCE_OUTPUTS_PATH)
        
        # Store GT and track boundary
        all_gt_blendshapes.append(gt_blendshapes)
        current_frame += gt_blendshapes.shape[0]
        file_boundaries.append(current_frame)
    
    # Remove last boundary (end of last file)
    if file_boundaries:
        file_boundaries = file_boundaries[:-1]
    
    # Concatenate all data
    print(f"\n\nConcatenating all files...")
    concatenated_gt = np.concatenate(all_gt_blendshapes, axis=0)
    concatenated_predictions = {}
    
    for model_name in all_predictions:
        concatenated_predictions[model_name] = {}
        for history in HISTORY_SIZES:
            concatenated_predictions[model_name][history] = {}
            for lookahead in LOOKAHEAD_SIZES:
                # Pad each prediction to match its corresponding GT length before concatenating
                padded_preds = []
                for i, pred in enumerate(all_predictions[model_name][history][lookahead]):
                    gt_len = all_gt_blendshapes[i].shape[0]
                    pred_len = pred.shape[0]
                    
                    if pred_len < gt_len:
                        # Pad with -1 values to match GT length
                        num_blendshapes = pred.shape[1]
                        padding = np.full((gt_len - pred_len, num_blendshapes), -0.0001)
                        padded_pred = np.concatenate([pred, padding], axis=0)
                        print(f"  Padded {model_name} (H={history}, LA={lookahead}) file {i}: {pred_len} -> {gt_len} frames")
                    else:
                        # Truncate if prediction is longer than GT
                        padded_pred = pred[:gt_len]
                        if pred_len > gt_len:
                            print(f"  Truncated {model_name} (H={history}, LA={lookahead}) file {i}: {pred_len} -> {gt_len} frames")
                    
                    padded_preds.append(padded_pred)
                
                concatenated_predictions[model_name][history][lookahead] = np.concatenate(padded_preds, axis=0)
    
    print(f"Concatenated GT shape: {concatenated_gt.shape}")
    for model_name in concatenated_predictions:
        for history in HISTORY_SIZES:
            for lookahead in LOOKAHEAD_SIZES:
                print(f"{model_name} (H={history}, LA={lookahead}) shape: {concatenated_predictions[model_name][history][lookahead].shape}")
    
    # Plot all concatenated results - all history sizes on one plot
    print(f"\n  Generating concatenated plot...")
    plot_comparison(
        concatenated_gt, 
        concatenated_predictions, 
        f"Concatenated: {len(run_paths)} files",
        MODEL_CONFIGS,
        BLENDSHAPES_TO_PLOT,
        file_boundaries
    )
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    main()

