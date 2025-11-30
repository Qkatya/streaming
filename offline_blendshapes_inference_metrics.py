
import os
import sys
import random
import pickle
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Import shared utilities
from blendshapes_utils import (
    BLENDSHAPES_ORDERED,
    init_fairseq,
    load_fairseq_model,
    load_nemo_model,
    infer_fairseq_model,
    infer_nemo_model,
    unnormalize_blendshapes,
    prepare_fairseq_sample
)

pio.renderers.default = "browser"

# ============================================================================
# CONSTANTS
# ============================================================================
BLENDSHAPE_COLORS = {'eyeBlinkRight': '#e377c2', 'jawOpen': '#ff7f0e', 'mouthFunnel': '#2ca02c', 'cheekPuff': '#d62728', 'mouthSmileLeft': '#9467bd', 'mouthFrownLeft': '#8c564b'}
PLOT_COLORS = ['#ff7f0e', '#2ca02c', '#d62728']

# ============================================================================
# UTILITY FUNCTIONS (metrics-specific)
# ============================================================================

def velocity_agreement(gt, pred, blendshape_name):
    """Calculate velocity agreement metrics between ground truth and prediction."""
    idx = BLENDSHAPES_ORDERED.index(blendshape_name)
    # # plot the gt and pred blendshapes
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(gt[:, idx])), y=gt[:, idx], name='GT Blendshapes'))
    # fig.add_trace(go.Scatter(x=np.arange(len(pred[:, idx])), y=pred[:, idx], name='Pred Blendshapes'))
    # fig.show()
    gt, pred = savgol_filter(gt[:, idx], 9, 2, mode='interp'), savgol_filter(pred[:, idx], 9, 2, mode='interp')
    da, db = np.diff(gt), np.diff(pred)
    # # plot gt and pred
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(gt)), y=gt, name='GT Blendshapes'))
    # fig.add_trace(go.Scatter(x=np.arange(len(pred)), y=pred, name='Pred Blendshapes'))
    # fig.show()
    # # plot da and db
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(da)), y=da, name='DA'))
    # fig.add_trace(go.Scatter(x=np.arange(len(db)), y=db, name='DB'))
    # fig.show()
    sign_match = np.mean(np.sign(da)==np.sign(db))
    r = np.corrcoef(da, db)[0,1]
    return sign_match, r 


def blinking_counter(blendshape_values, blink_th=2):
    """Count blinks using peak detection on z-scored values."""
    z_scores = zscore(blendshape_values)
    peaks, _ = find_peaks(z_scores, height=blink_th, prominence=1.5, distance=5, width=(None, 20))
    return 0 if np.mean(blendshape_values > 0.44) > 0.5 else len(peaks)

# ============================================================================
# DATA LOADING HELPER FUNCTIONS
# ============================================================================

def load_gt_blendshapes(row, gt_parent_path):
    """Load and downsample ground truth blendshapes from dataframe row."""
    from blendshapes_utils import load_ground_truth_blendshapes
    gt_data_path = gt_parent_path / row['run_path']
    return load_ground_truth_blendshapes(gt_data_path, downsample=True)

def compute_metrics(gt, preds, blendshape_names):
    """Compute velocity agreement, RMSE, and blink metrics for all models."""
    results = {}
    for bs_name in blendshape_names:
        # print(bs_name)
        idx = BLENDSHAPES_ORDERED.index(bs_name)
        for i, pred in enumerate(preds, 1):
            sign_match, r = velocity_agreement(gt, pred, bs_name)
            results[f'sign_match_{bs_name}_model{i}'] = sign_match
            results[f'r_{bs_name}_model{i}'] = r
            # Calculate RMSE for this blendshape
            rmse = np.sqrt(np.mean((gt[:, idx] - pred[:, idx]) ** 2))/np.mean(gt[:, idx])
            # print(rmse)
            results[f'rmse_{bs_name}_model{i}'] = rmse
    
    eyeBlinkRight_idx = BLENDSHAPES_ORDERED.index("eyeBlinkRight")
    results['blink_counter_gt'] = blinking_counter(gt[:, eyeBlinkRight_idx], blink_th=2)
    for i, pred in enumerate(preds, 1):
        results[f'blink_counter_model{i}'] = blinking_counter(pred[:, eyeBlinkRight_idx-1], blink_th=1.8)
    return results

def plot_blendshape_comparison(gt, predictions, model_infos, use_diff=True):
    """Plot GT vs predicted blendshapes for visual comparison."""
    blendshapes_to_plot = ['eyeBlinkRight', 'jawOpen', 'mouthFunnel', 'cheekPuff', 'mouthFrownLeft']
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=tuple(f"<span style='font-size:20px'>{t}</span>" for t in blendshapes_to_plot), vertical_spacing=0.05)
    
    def add_traces(bs_name, row_num, use_diff=False, show_legends=None):
        if show_legends is None:
            show_legends = [False] * (len(predictions) + 1)
        
        idx = BLENDSHAPES_ORDERED.index(bs_name)
        gt_data = gt[:, idx]
        pred_data = [pred[:, idx] for pred in predictions]
        
        if use_diff:
            gt_data = np.diff(savgol_filter(gt_data, 9, 2, mode='interp'))
            pred_data = [np.diff(savgol_filter(x, 9, 2, mode='interp')) for x in pred_data]
        
        # Add GT trace
        fig.add_trace(go.Scatter(x=np.arange(len(gt_data)), y=gt_data, name='GT Blendshapes' if show_legends[0] else None, 
                                mode='lines', line=dict(color='blue', width=3), showlegend=show_legends[0]), row=row_num, col=1)
        
        # Add prediction traces
        for i, (pred, model_info) in enumerate(zip(pred_data, model_infos), 1):
            config = model_info['config']
            line_dict = dict(color=BLENDSHAPE_COLORS[bs_name], width=3)
            if config['line_style']:
                line_dict['dash'] = config['line_style']
            
            fig.add_trace(go.Scatter(x=np.arange(len(pred)), y=pred, 
                                    name=config['display_name'] if show_legends[i] else None, 
                                    mode='lines', line=line_dict, showlegend=show_legends[i]), row=row_num, col=1)
    
    # First row shows all legends
    show_legends_first = [True] * (len(predictions) + 1)
    add_traces('eyeBlinkRight', 1, use_diff=False, show_legends=show_legends_first)
    
    # Subsequent rows show only prediction legends (not GT)
    show_legends_rest = [False] + [True] * len(predictions)
    for i, bs_name in enumerate(blendshapes_to_plot[1:], 2):
        add_traces(bs_name, i, use_diff=use_diff, show_legends=show_legends_rest)
    
    fig.update_layout(title_text="Gt vs pred blendshapes", title_x=0.47, title_xanchor='center', title_font_size=30, legend=dict(font=dict(size=16)))
    fig.show()
    
# ============================================================================
# MODEL & DATA LOADING
# ============================================================================
init_fairseq()

# Configuration
device = 'cuda:0'
use_fp16 = True
st, en = 0, None

# Paths
gt_parent_path = Path("/mnt/A3000/Recordings/v2_data")
features_path = Path('/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features')

# Model configurations: (path, type, display_name, blendshape_indices, line_style)
# type: 'fairseq' or 'nemo'
# blendshape_indices: indices to use for this model
# line_style: None (solid), 'dash', 'dot', 'dashdot'
MODEL_CONFIGS = [
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_loud/0/checkpoints/checkpoint_last.pt',
    #     'type': 'fairseq',
    #     'name': 'blendshapes_loud',
    #     'display_name': 'D2V trained on 400k samples',
    #     'blendshape_indices': list(range(1, 52)),
    #     'line_style': None
    # },
    # {
    #     'path': '/mnt/ML/TrainResults/ido.kazma/D2V/V2/2025_04_15/new21_baseline_blendshapes_normalized/0/checkpoints/checkpoint_last.pt',
    #     'type': 'fairseq',
    #     'name': 'new21_baseline_blendshapes_normalized',
    #     'display_name': 'D2V trained on 2.5M samples',
    #     'blendshape_indices': list(range(1, 52)),
    #     'line_style': 'dashdot'
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_blendshapes_heads_only/checkpoints/causal_fastconformer_layernorm_landmarks_blendshapes_heads_only.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_fastconformer_layernorm_landmarks_blendshapes_heads_only',
    #     'display_name': 'NeMo FastConformer (Causal)',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'line_style': 'dash'
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/quartznet_landmarks_blendshapes/checkpoints/quartznet_landmarks_blendshapes.nemo',
    #     'type': 'nemo',
    #     'name': 'quartznet_landmarks_blendshapes',
    #     'display_name': 'NeMo QuartzNet',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'line_style': 'dot'
    # },
    {
        'path': '/home/katya.ivantsiv/blendshapes_models/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides.nemo',
        'type': 'nemo',
        'name': 'causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides',
        'display_name': 'NeMo FastConformer (Causal) - All Blendshapes, partly trained',
        'blendshape_indices': list(range(1, 52)),
        'line_style': None
    },
    {
        'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides/checkpoints/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides.nemo',
        'type': 'nemo',
        'name': 'causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides',
        'display_name': 'NeMo FastConformer (Causal) - All Blendshapes',
        'blendshape_indices': list(range(1, 52)),
        'line_style': None
    },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/fastconformer_blendshapes_landmarks/checkpoints/fastconformer_blendshapes_landmarks.nemo',
    #     'type': 'nemo',
    #     'name': 'fastconformer_blendshapes_landmarks',
    #     'display_name': 'NeMo FastConformer',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'line_style': 'dashdot'
    # },
]

# Load all models
models = []
for config in MODEL_CONFIGS:
    print(f"\nLoading {config['name']}...")
    if config['type'] == 'fairseq':
        model, saved_cfg, task = load_fairseq_model(config['path'], device, use_fp16)
        models.append({'model': model, 'config': config, 'saved_cfg': saved_cfg, 'task': task})
    elif config['type'] == 'nemo':
        model = load_nemo_model(config['path'], device, use_fp16)
        models.append({'model': model, 'config': config})
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

# Load dataframe
df_path = '/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250402_split_1/LOUD_GIP_general_clean_250415_v2.pkl'
print(f"Loading dataframe from {df_path}")
df = pd.read_pickle(df_path)
print(f"Loaded dataframe with {len(df)} rows")

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
# Load normalization factors (for fairseq models)
blendshapes_normalize_path = '/home/ido.kazma/projects/notebooks-qfairseq/stats_df.pkl'
with open(blendshapes_normalize_path, 'rb') as f:
    blendshape_normalization_factors = pickle.load(f)

# Initialize metric storage dynamically based on number of models
num_models = len(MODEL_CONFIGS)
metrics = {f'{metric}_{bs}_model{m}': [] for metric in ['sign_match', 'r', 'rmse'] for bs in ['jawOpen', 'mouthFunnel', 'cheekPuff'] for m in range(1, num_models + 1)}
blink_counters = {f'blink_counter_{name}': [] for name in ['gt'] + [f'model{i}' for i in range(1, num_models + 1)]}

plot_gt_vs_pred_flag = True

# Sample selection
row_idxs = [3663]  # random.sample(range(0, len(df)), 50)
row_idxs = [3953]  # random.sample(range(0, len(df)), 50)
row_idxs = random.sample(range(0, len(df)), 50) #[556,1004, 2877,1744, 3663]
# row_idxs = [3663,3642,3953] #, 317, 6117, 1710, 3252, 1820, 3847, ] #random.sample(range(0, len(df)+1), 30) #[556,1004, 2877,1744, 3663]
# row_idxs = [3663] #, 317, 6117, 1710, 3252, 1820, 3847, ] #random.sample(range(0, len(df)+1), 30) #[556,1004, 2877,1744, 3663]
# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================
for row_ind in tqdm(row_idxs):
    row = df.iloc[row_ind]
    
    # Load data and prepare sample
    data = np.load(features_path / row.run_path / f'{row.tar_id}.npy')
    sample, padding_mask = prepare_fairseq_sample(data, device, st, en)
    gt_blendshapes = load_gt_blendshapes(row, gt_parent_path)
    
    # Run inference on all models
    all_predictions = []
    for model_info in models:
        config = model_info['config']
        model = model_info['model']
        
        if config['type'] == 'fairseq':
            blendshapes = infer_fairseq_model(model, sample, padding_mask, config['blendshape_indices'])
        elif config['type'] == 'nemo':
            blendshapes = infer_nemo_model(model, data, device, config['blendshape_indices'])
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Unnormalize blendshapes
        blendshapes_unnorm = unnormalize_blendshapes(
            blendshapes, 
            config['type'], 
            config['blendshape_indices'],
            blendshape_normalization_factors
        )
        all_predictions.append(blendshapes_unnorm)
    
    # Plot blendshapes comparison
    # plot_blendshape_comparison(gt_blendshapes, all_predictions, models, use_diff=False)
    
    # Align lengths
    min_len = min(len(gt_blendshapes), *[len(pred) for pred in all_predictions])
    gt_blendshapes = gt_blendshapes[:min_len]
    all_predictions = [pred[:min_len] for pred in all_predictions]
    
    # Compute metrics
    results = compute_metrics(gt_blendshapes, all_predictions, ['jawOpen', 'mouthFunnel', 'cheekPuff'])
    for key, val in results.items():
        if key.startswith('blink'):
            blink_counters[key].append(val)
        else:
            metrics[key].append(val)
    
    # # Optional plotting
    # if plot_gt_vs_pred_flag:
    #     plot_blendshape_comparison(gt_blendshapes, blendshapes_unnorm, blendshapes_unnorm2, blendshapes3) # Plit diff of Blendshapes (looks better)


# ============================================================================
# BLINK DETECTION ANALYSIS
# ============================================================================
def blink_detection_stats(gt, pred, model_name):
    """Calculate and print blink detection statistics."""
    print(f"\n=== Blink Detection Stats for {model_name} ===")
    correct, missed, imagined = 0, 0, 0
    for gt_val, pred_val in zip(gt, pred):
        if gt_val == pred_val:
            correct += gt_val
        elif pred_val < gt_val:
            missed += (gt_val - pred_val)
            correct += pred_val
        else:
            correct += gt_val
            imagined += (pred_val - gt_val)
    total, totl_pred = correct + missed + imagined, correct + missed
    print(f"Correct: {correct}, Missed: {missed}, Imagined: {imagined}, Total: {total}")
    print(f"Correct: {100*correct/totl_pred:.1f}%, Missed: {100*missed/totl_pred:.1f}%, Imagined: {100*imagined/total:.1f}% (Total: {sum(gt)} blinks)")

print(f"\n=== Blink Detection Analysis ===")
blink_gt = np.array(blink_counters['blink_counter_gt'])
for i in range(1, num_models + 1):
    model_name = MODEL_CONFIGS[i-1]['name']
    blink_detection_stats(blink_gt, np.array(blink_counters[f'blink_counter_model{i}']), f"Model {i} ({model_name})")


# ============================================================================
# FINAL RESULTS VISUALIZATION
# ============================================================================
def plot_model_comparison(metrics, blendshape_names, model_configs):
    """Create comparison plot for model performance across blendshapes."""
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=("'%' of time movement is in the same direction", 
                                       "Strength of velocity correlation",
                                       "RMSE"), 
                        shared_yaxes=False, 
                        horizontal_spacing=0.1)
    model_labels = [config['display_name'] for config in model_configs]
    
    for bs_name, color in zip(blendshape_names, PLOT_COLORS):
        # Extract metrics for this blendshape
        sign_means = [np.nanmean(metrics[f'sign_match_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        sign_stds = [np.nanstd(metrics[f'sign_match_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        r_means = [np.nanmean(metrics[f'r_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        r_stds = [np.nanstd(metrics[f'r_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        rmse_means = [np.nanmean(metrics[f'rmse_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        rmse_stds = [np.nanstd(metrics[f'rmse_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        
        fig.add_trace(go.Scatter(x=model_labels, y=sign_means, mode='lines+markers', name=f'{bs_name} (Sign Match)', line=dict(color=color, width=3), marker=dict(size=10, color=color), error_y=dict(type='data', array=sign_stds, visible=True, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=model_labels, y=r_means, mode='lines+markers', name=f'{bs_name}', line=dict(color=color, width=3), marker=dict(size=10, color=color), error_y=dict(type='data', array=r_stds, visible=True, color=color), showlegend=True), row=1, col=2)
        fig.add_trace(go.Scatter(x=model_labels, y=rmse_means, mode='lines+markers', name=f'{bs_name} (RMSE)', line=dict(color=color, width=3), marker=dict(size=10, color=color), error_y=dict(type='data', array=rmse_stds, visible=True, color=color), showlegend=False), row=1, col=3)
    
    fig.update_layout(title='Model Performance Comparison', height=500, width=1500, hovermode='x unified', legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_xaxes(title_text="Model", row=1, col=3)
    fig.show()

plot_model_comparison(metrics, ['jawOpen', 'mouthFunnel', 'cheekPuff'], MODEL_CONFIGS)
a=1
