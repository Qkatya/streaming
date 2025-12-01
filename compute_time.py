print("Starting compute_time.py")
import os
import sys
import pickle
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"

# Import shared utilities
from blendshapes_utils import (
    BLENDSHAPES_ORDERED,
    init_fairseq,
    load_fairseq_model,
    load_nemo_model,
    infer_fairseq_model,
    infer_nemo_model,
    unnormalize_blendshapes,
    find_zip_file_id,
    prepare_fairseq_sample
)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Paths
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
FEATURES_PATH = Path("/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features")

# Sliding window parameters
HISTORY_SIZES = [800, 400, 200]  # List of history sizes to test
CHUNK_SIZE = 24   # Number of frames to extract from each window (the "present")
LOOKAHEAD_SIZE = 1  # Fixed lookahead size

# Number of random samples to test
NUM_SAMPLES = 500

# Model configurations
MODEL_CONFIGS = [
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_loud/0/checkpoints/checkpoint_last.pt',
    #     'type': 'fairseq',
    #     'name': 'blendshapes_loud',
    #     'blendshape_indices': list(range(1, 52)),
    #     'color': '#ff7f0e',  # Orange
    # },
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
    { 
        'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_preprocessor_encoder_with_smile/checkpoints/causal_preprocessor_encoder_with_smile.nemo',
        'type': 'nemo',
        'name': 'causal_preprocessor_encoder_with_smile',
        'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 31, 32, 38, 45],
        'color': '#17becf',  # Cyan
    },
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

# Split dataframe configuration
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_input_features(run_path: str, tar_id: str = None) -> Optional[np.ndarray]:
    """Load input features from .npy file."""
    full_run_path = ALL_DATA_PATH / run_path
    
    if tar_id is not None:
        features_file = FEATURES_PATH / run_path / f"{tar_id}.npy"
    else:
        zip_id = find_zip_file_id(full_run_path)
        if zip_id is None:
            print(f"Warning: No .right.zip file found in {full_run_path}")
            return None
        features_file = FEATURES_PATH / run_path / f"{zip_id}.npy"
    
    if not features_file.exists():
        print(f"Warning: Features file not found: {features_file}")
        return None
    
    return np.load(features_file)

def get_random_samples_from_split(df_path: str, num_samples: int) -> List[Tuple[str, str]]:
    """Get random samples from split dataframe.
    
    Returns:
        List of tuples (run_path, tar_id)
    """
    print(f"Loading dataframe from {df_path}")
    df = pd.read_pickle(df_path)
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Sample random rows
    num_samples = min(num_samples, len(df))
    sampled_rows = df.sample(n=num_samples, random_state=42)
    print(f"Sampled {num_samples} random rows")
    
    # Extract run_path and tar_id pairs
    run_path_tar_id_pairs = []
    for idx, row in sampled_rows.iterrows():
        run_path_tar_id_pairs.append((row.run_path, row.tar_id))
    
    return run_path_tar_id_pairs

def create_inference_window(input_features: np.ndarray, history_size: int, lookahead_size: int) -> np.ndarray:
    """Create a single inference window from the middle of the input features.
    
    Args:
        input_features: Full input features array
        history_size: Size of history context
        lookahead_size: Size of lookahead context
        
    Returns:
        Window data for inference
    """
    total_frames = len(input_features)
    
    # Take window from the middle of the sequence
    # Window size = history + chunk + lookahead
    window_size = history_size + CHUNK_SIZE + lookahead_size
    
    if total_frames < window_size:
        # If input is smaller than window, use entire input
        return input_features
    
    # Start from middle
    middle = total_frames // 2
    window_start = max(0, middle - history_size)
    window_end = min(total_frames, window_start + window_size)
    
    return input_features[window_start:window_end]

def time_single_inference(
    model,
    model_config: Dict,
    window_data: np.ndarray,
    device: str,
    normalization_factors=None,
    warmup: bool = False
) -> float:
    """Time a single inference pass.
    
    Args:
        model: The model to time
        model_config: Model configuration dict
        window_data: Input window data
        device: Device to run on
        normalization_factors: Normalization factors for fairseq models
        warmup: If True, don't return timing (warmup run)
        
    Returns:
        Inference time in seconds (0 if warmup)
    """
    # Synchronize GPU before starting timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    # Run inference based on model type
    if model_config['type'] == 'fairseq':
        sample, padding_mask = prepare_fairseq_sample(window_data, device)
        blendshapes = infer_fairseq_model(model, sample, padding_mask, model_config['blendshape_indices'])
    elif model_config['type'] == 'nemo':
        blendshapes = infer_nemo_model(model, window_data, device, model_config['blendshape_indices'])
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Unnormalize (part of the inference pipeline)
    blendshapes_unnorm = unnormalize_blendshapes(
        blendshapes, 
        model_config['type'], 
        model_config['blendshape_indices'],
        normalization_factors
    )
    
    # Synchronize GPU after inference
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    if warmup:
        return 0.0
    else:
        return end_time - start_time

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
        try:
            if config['type'] == 'fairseq':
                model, saved_cfg, task = load_fairseq_model(config['path'], DEVICE, USE_FP16)
                models.append({'model': model, 'config': config, 'saved_cfg': saved_cfg, 'task': task})
            elif config['type'] == 'nemo':
                model = load_nemo_model(config['path'], DEVICE, USE_FP16)
                models.append({'model': model, 'config': config})
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            print(f"  ✓ Successfully loaded {config['name']}")
        except Exception as e:
            print(f"  ✗ Failed to load {config['name']}: {e}")
            print(f"  Skipping this model...")
            continue
    
    print(f"\nSuccessfully loaded {len(models)} models")
    
    # Get random samples from split
    print("\n" + "="*80)
    print("SAMPLING FROM SPLIT")
    print("="*80)
    run_path_tar_id_pairs = get_random_samples_from_split(SPLIT_DF_PATH, NUM_SAMPLES)
    
    # Load all samples first
    print("\n" + "="*80)
    print("LOADING SAMPLES")
    print("="*80)
    samples_data = []
    for run_path, tar_id in tqdm(run_path_tar_id_pairs, desc="Loading features"):
        input_features = load_input_features(run_path, tar_id)
        if input_features is None:
            print(f"Skipping {run_path}/{tar_id} - could not load features")
            continue
        samples_data.append({
            'run_path': run_path,
            'tar_id': tar_id,
            'features': input_features
        })
    
    print(f"\nSuccessfully loaded {len(samples_data)} samples")
    
    # Storage for timing results
    # Structure: {model_name: {history_size: [list of times]}}
    timing_results = {
        model_info['config']['name']: {
            history: [] for history in HISTORY_SIZES
        } for model_info in models
    }
    
    # Benchmark each model
    print("\n" + "="*80)
    print("BENCHMARKING MODELS")
    print("="*80)
    
    for model_info in models:
        config = model_info['config']
        model = model_info['model']
        model_name = config['name']
        
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        for history in HISTORY_SIZES:
            print(f"\n  History Size: {history}")
            print(f"  {'-'*60}")
            
            # Warmup run on first sample
            if len(samples_data) > 0:
                print(f"  Running warmup...")
                window_data = create_inference_window(
                    samples_data[0]['features'], 
                    history, 
                    LOOKAHEAD_SIZE
                )
                time_single_inference(
                    model,
                    config,
                    window_data,
                    DEVICE,
                    blendshape_normalization_factors,
                    warmup=True
                )
                print(f"  Warmup complete")
            
            # Time on all samples
            print(f"  Timing on {len(samples_data)} samples...")
            times = []
            for sample in tqdm(samples_data, desc=f"  H={history}", leave=False):
                window_data = create_inference_window(
                    sample['features'], 
                    history, 
                    LOOKAHEAD_SIZE
                )
                
                inference_time = time_single_inference(
                    model,
                    config,
                    window_data,
                    DEVICE,
                    blendshape_normalization_factors,
                    warmup=False
                )
                times.append(inference_time)
            
            timing_results[model_name][history] = times
            
            # Calculate and print stats
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            median_time = np.median(times)
            
            print(f"  Results:")
            print(f"    Mean:   {avg_time*1000:.4f} ms (±{std_time*1000:.4f} ms)")
            print(f"    Median: {median_time*1000:.4f} ms")
            print(f"    Min:    {min_time*1000:.4f} ms")
            print(f"    Max:    {max_time*1000:.4f} ms")
            print(f"    FPS:    {1.0/avg_time:.2f} inferences/sec")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print("\nInference Time (ms) - Mean ± Std")
    print("-" * 120)
    
    # Header
    header = f"{'Model Name':<50}"
    for history in HISTORY_SIZES:
        header += f" | H={history:>4}"
    print(header)
    print("-" * 120)
    
    # Rows
    for model_info in models:
        model_name = model_info['config']['name']
        row = f"{model_name:<50}"
        for history in HISTORY_SIZES:
            times = timing_results[model_name][history]
            avg = np.mean(times) * 1000  # Convert to ms
            std = np.std(times) * 1000
            row += f" | {avg:>4.2f}±{std:<4.2f}"
        print(row)
    
    print("-" * 120)
    
    # FPS Summary
    print("\nFrames Per Second (FPS) - Mean")
    print("-" * 120)
    
    # Header
    header = f"{'Model Name':<50}"
    for history in HISTORY_SIZES:
        header += f" | H={history:>4}"
    print(header)
    print("-" * 120)
    
    # Rows
    for model_info in models:
        model_name = model_info['config']['name']
        row = f"{model_name:<50}"
        for history in HISTORY_SIZES:
            times = timing_results[model_name][history]
            avg_time = np.mean(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            row += f" | {fps:>8.2f}"
        print(row)
    
    print("-" * 120)
    
    # Create plotly visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Create histogram plots for each history size
    for history in HISTORY_SIZES:
        print(f"\nGenerating histogram for History={history}...")
        
        fig = go.Figure()
        
        for model_info in models:
            model_name = model_info['config']['name']
            color = model_info['config'].get('color', '#17becf')
            times = timing_results[model_name][history]
            times_ms = np.array(times) * 1000  # Convert to ms
            
            fig.add_trace(go.Histogram(
                x=times_ms,
                name=model_name,
                opacity=0.7,
                marker=dict(color=color),
                nbinsx=30,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Time: %{x:.2f} ms<br>' +
                              'Count: %{y}<br>' +
                              '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f"Inference Time Distribution - History={history}, Lookahead={LOOKAHEAD_SIZE}<br>" +
                     f"<sub>{len(samples_data)} samples</sub>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Inference Time (ms)",
            yaxis_title="Count",
            barmode='overlay',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            hovermode='closest'
        )
        
        fig.show()
    
    # Create box plot comparing all models and history sizes
    print(f"\nGenerating box plot comparison...")
    
    fig = go.Figure()
    
    for model_info in models:
        model_name = model_info['config']['name']
        color = model_info['config'].get('color', '#17becf')
        
        for history in HISTORY_SIZES:
            times = timing_results[model_name][history]
            times_ms = np.array(times) * 1000  # Convert to ms
            
            fig.add_trace(go.Box(
                y=times_ms,
                name=f"{model_name}<br>H={history}",
                marker=dict(color=color),
                boxmean='sd',  # Show mean and std
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Time: %{y:.2f} ms<br>' +
                              '<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text=f"Inference Time Comparison - All Models & History Sizes<br>" +
                 f"<sub>Lookahead={LOOKAHEAD_SIZE}, {len(samples_data)} samples</sub>",
            x=0.5,
            xanchor='center'
        ),
        yaxis_title="Inference Time (ms)",
        height=800,
        showlegend=True,
        hovermode='closest'
    )
    
    fig.show()
    
    # Create bar chart with error bars (mean ± std)
    print(f"\nGenerating bar chart with error bars...")
    
    fig = go.Figure()
    
    x_labels = []
    
    for model_info in models:
        model_name = model_info['config']['name']
        color = model_info['config'].get('color', '#17becf')
        
        for history in HISTORY_SIZES:
            times = timing_results[model_name][history]
            times_ms = np.array(times) * 1000  # Convert to ms
            
            mean_time = np.mean(times_ms)
            std_time = np.std(times_ms)
            
            label = f"{model_name}<br>H={history}"
            x_labels.append(label)
            
            fig.add_trace(go.Bar(
                x=[label],
                y=[mean_time],
                error_y=dict(
                    type='data',
                    array=[std_time],
                    visible=True
                ),
                name=model_name,
                marker=dict(color=color),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>' +
                              'Mean: %{y:.2f} ms<br>' +
                              'Std: %{error_y.array[0]:.2f} ms<br>' +
                              '<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text=f"Mean Inference Time with Standard Deviation<br>" +
                 f"<sub>Lookahead={LOOKAHEAD_SIZE}, {len(samples_data)} samples</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Model and History Size",
        yaxis_title="Inference Time (ms)",
        height=700,
        showlegend=False,
        hovermode='closest'
    )
    
    fig.update_xaxes(tickangle=-45)
    
    fig.show()
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    main()

