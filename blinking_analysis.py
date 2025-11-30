import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from blink_analyzer import BlinkAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Import lightweight data utilities (no ML dependencies - fast import!)
from blendshapes_data_utils import load_ground_truth_blendshapes, BLENDSHAPES_ORDERED

# Configuration
SIDE = 'right'  # Filter for right side only

gt_blendshapes_path = Path("/mnt/A3000/Recordings/v2_data")
# Note: The prediction filename includes the side (pred_right_...) to load right side data only
pred_path = Path("/home/katya.ivantsiv/streaming/inference_outputs/2025/11/06/KatyaIvantsiv-143719/4_1_38909aea-68df-4630-b952-13f638262f9c_silent/pred_right_causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side_H800_LA1.npy")
gt_path = gt_blendshapes_path / "2025/11/06/KatyaIvantsiv-143719/4_1_38909aea-68df-4630-b952-13f638262f9c_silent"
gt_blendshapes = load_ground_truth_blendshapes(gt_path, downsample=True)
pred_blendshapes = np.load(pred_path)

print(f"Loading {SIDE} side data:")
print(f"  GT shape: {gt_blendshapes.shape}")
print(f"  Pred shape: {pred_blendshapes.shape}")
# Blendshapes to plot (you can customize this list)
# BLENDSHAPES_TO_PLOT = [
#     '_neutral', 'browDownRight', 'browInnerUp', 'browOuterUpRight', 'cheekPuff', 'cheekSquintRight', 
#     'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
#     'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleRight', 
#     'mouthFrownRight', 'mouthFunnel', 'mouthLowerDownRight', 'mouthPressRight', 'mouthPucker', 'mouthRight', 
#     'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileRight', 
#     'mouthStretchRight', 'mouthUpperUpRight', 'noseSneerRight'
# ]
# BLENDSHAPES_TO_PLOT = ['eyeBlinkRight']
BLENDSHAPES_TO_PLOT = ['eyeBlinkRight', 'jawOpen', 'mouthFunnel', 'cheekPuff', 'mouthSmileRight']

# Create subplot figure
num_plots = len(BLENDSHAPES_TO_PLOT)
max_spacing = 1.0 / (num_plots - 1) if num_plots > 1 else 0.2
vertical_spacing = min(0.008, max_spacing * 0.5)

fig = make_subplots(
    rows=num_plots, 
    cols=1, 
    shared_xaxes=True,
    subplot_titles=tuple(f"<span style='font-size:16px'>{bs}</span>" for bs in BLENDSHAPES_TO_PLOT),
    vertical_spacing=vertical_spacing
)

# Plot each blendshape
for plot_idx, bs_name in enumerate(BLENDSHAPES_TO_PLOT, 1):
    bs_idx = BLENDSHAPES_ORDERED.index(bs_name)
    
    # Get data for this blendshape
    gt_data = gt_blendshapes[:, bs_idx]
    pred_data = pred_blendshapes[:, bs_idx]
    
    # Align lengths
    min_len = min(len(gt_data), len(pred_data))
    gt_data = gt_data[:min_len]
    pred_data = pred_data[:min_len]
    
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
fig.update_layout(
    title_text=f"Blendshapes Comparison: GT vs Prediction ({SIDE.capitalize()} Side)<br><sub>{pred_path.parent.name}</sub>",
    title_x=0.5,
    title_xanchor='center',
    title_font_size=20,
    height=300 * num_plots,
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

fig.update_xaxes(title_text="Frame", row=num_plots, col=1)
fig.show()


# Blink analysis

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
    
gt_th = 0.06
quantizer = 8
dense_region = np.linspace(0, 0.1, 300, endpoint=True)
sparse_region1 = np.linspace(-10, -1, 300, endpoint=True)
sparse_region2 = np.linspace(-1, 0, 50, endpoint=False)
sparse_region3 = np.linspace(0.1, 3, 400, endpoint=True)

th_list = np.unique(np.concatenate([sparse_region1, sparse_region2, dense_region, sparse_region3]))
TPR_lst, FNR_lst, FPR_lst = run_blink_analysis(th_list, blendshapes_list, pred_blends_list, quantizer, gt_th)
TPR_lst2, FNR_lst2, FPR_lst2 = run_blink_analysis(th_list, blendshapes_list, pred_blends2_list, quantizer, gt_th)

# Create ROC curve using plotly
import plotly.graph_objects as go
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=FPR_lst, y=TPR_lst, mode='lines+markers', name='2.5M samples', line=dict(color='blue', width=3), marker=dict(size=8)))
fig_roc.add_trace(go.Scatter(x=FPR_lst2, y=TPR_lst2, mode='lines+markers', name='400k samples', line=dict(color='red', width=3), marker=dict(size=8)))
fig_roc.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Random Classifier', line=dict(color='gray', width=2, dash='dash')))
fig_roc.update_layout(title=dict(text=f'ROC Curve - Blink Detection', font=dict(size=30)), xaxis_title=dict(text='False Positive Rate (%)', font=dict(size=20)), yaxis_title=dict(text='True Positive Rate (%)', font=dict(size=20)), xaxis=dict(range=[0, 100], tickfont=dict(size=16)), yaxis=dict(range=[0, 100], tickfont=dict(size=16)), showlegend=True, legend=dict(font=dict(size=18)), width=800, height=600)
# fig_roc.update_layout(title=f'ROC Curve - Blink Detection - Quantizer {quantizer}, gt_th {gt_th}', xaxis_title='False Positive Rate (%)', yaxis_title='True Positive Rate (%)', xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]), showlegend=True, width=800, height=600)
fig_roc.show()
    