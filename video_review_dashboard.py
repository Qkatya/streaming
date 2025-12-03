#!/usr/bin/env python3
"""
Dashboard to review video snippets with prediction plots showing all peaks.

Shows:
- Video snippet with red border around the peak frame
- Prediction plot with all detected peaks (matched and unmatched)
- Special highlight around the current unmatched peak being reviewed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import zscore
import sys

# Configuration
PICKLE_FILE = "unmatched_pred_peaks_best_tpr_th0.0000.pkl"
VIDEO_SNIPPETS_DIR = Path("video_snippets")
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'

MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1
BLENDSHAPE_FPS = 25  # Blendshape frame rate

# Peak detection parameters (matching the analysis)
PROMINENCE_PRED = 0.5
HEIGHT_THRESHOLD = 0.0
DISTANCE = 5
MAX_OFFSET = 10  # For matching peaks

# Import data loading functions
sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes

# Load unmatched predictions
df = pd.read_pickle(PICKLE_FILE)
print(f"\n{'='*80}")
print(f"VIDEO REVIEW DASHBOARD - UNMATCHED PREDICTIONS")
print(f"{'='*80}")
print(f"Total unmatched peaks: {len(df)}")
print(f"Threshold used: {df['threshold'].iloc[0]:.4f}")
print(f"{'='*80}\n")

if len(df) == 0:
    print("No samples found.")
    exit(1)

# Load split dataframe to get additional info
split_df = pd.read_pickle(SPLIT_DF_PATH)


def load_pred_and_gt_data(row):
    """Load prediction and GT blendshapes for a given row."""
    run_path = row['run_path']
    tar_id = row['tar_id']
    side = row['side']
    
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
    """Detect all peaks in GT and prediction signals and match them."""
    # Extract eyeBlinkRight (index 10) or eyeBlinkLeft (index 9) based on side
    gt_blink = gt_bs[:, 10]  # Right blink
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
        'gt_peaks': gt_peaks,
        'pred_peaks': pred_peaks,
        'matched_gt': matched_gt,
        'matched_pred': matched_pred,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred
    }


# Preload all data
print("Loading data for all samples...")
video_data = []

for idx, row in df.iterrows():
    # Get video path from filename pattern
    run_name = row['run_path'].replace('/', '_').replace('\\', '_')
    peak_frame_25fps = row['peak_frame_in_file']
    peak_frame_30fps = row['video_frame_in_file']
    peak_value = row['peak_value']
    tar_id = str(row['tar_id']).replace('-', '')[:8]  # Shorten tar_id for filename
    side = row['side']
    
    # Match the filename pattern from download_unmatched_peak_videos.py
    filename = f"{run_name}_tar{tar_id}_{side}_frame{peak_frame_30fps}_25fps{peak_frame_25fps}_peak{peak_value:.3f}.mp4"
    video_path = VIDEO_SNIPPETS_DIR / filename
    
    if not video_path.exists():
        # Try to find the video with a different pattern
        matching_files = list(VIDEO_SNIPPETS_DIR.glob(f"*_frame{peak_frame_30fps}_25fps{peak_frame_25fps}_peak{peak_value:.3f}.mp4"))
        if matching_files:
            video_path = matching_files[0]
        else:
            print(f"Video not found: {filename}")
            continue
    
    # Load GT and pred data
    gt_bs, pred_bs = load_pred_and_gt_data(row)
    
    if gt_bs is None:
        continue
    
    # Detect all peaks
    peak_data = detect_all_peaks(gt_bs, pred_bs)
    
    video_data.append({
        'path': video_path,
        'info': row.to_dict(),
        'peak_frame': row['peak_frame_in_file'],
        'peak_data': peak_data
    })

print(f"Loaded {len(video_data)} samples successfully")

if len(video_data) == 0:
    print("No data could be loaded. Exiting.")
    exit(1)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dcc.Store(id='current-index', data=0),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Video Review Dashboard", className="text-center mb-4"),
            html.H5("Reviewing unmatched predictions with peak visualization", className="text-center text-muted mb-4"),
        ])
    ]),
    
    # Progress
    dbc.Row([
        dbc.Col([
            html.Div(id='progress-text', className="text-center mb-2"),
            dbc.Progress(id='progress-bar', value=0, className="mb-4"),
        ])
    ]),
    
    # Sample info
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Sample Information", className="card-title"),
                    html.Div(id='sample-info'),
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Video player
    dbc.Row([
        dbc.Col([
            html.Video(
                id='video-player',
                controls=True,
                autoPlay=True,
                loop=True,
                style={'width': '100%', 'maxWidth': '800px', 'margin': 'auto', 'display': 'block'}
            )
        ])
    ], className="mb-4"),
    
    # Prediction plot with all peaks
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='prediction-plot', style={'height': '600px'})
        ])
    ], className="mb-4"),
    
    # Navigation buttons
    dbc.Row([
        dbc.Col([
            dbc.Button("← Previous", id='btn-prev', color="primary", size="lg", className="w-100")
        ], width=6),
        dbc.Col([
            dbc.Button("Next →", id='btn-next', color="primary", size="lg", className="w-100")
        ], width=6),
    ], className="mb-4"),
    
], fluid=True, style={'maxWidth': '1400px', 'padding': '20px'})


def get_video_base64(video_path):
    """Read video file and encode as base64."""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode('ascii')
    return f"data:video/mp4;base64,{video_base64}"


def create_prediction_plot(peak_data, current_peak_frame):
    """Create plot showing prediction with all peaks marked."""
    pred_smooth = peak_data['pred_smooth']
    pred_zscore = peak_data['pred_zscore']
    gt_peaks = peak_data['gt_peaks']
    pred_peaks = peak_data['pred_peaks']
    matched_gt = peak_data['matched_gt']
    matched_pred = peak_data['matched_pred']
    unmatched_pred = peak_data['unmatched_pred']
    
    frames = np.arange(len(pred_smooth))
    
    # Create figure with subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("eyeBlinkRight - Smoothed Signal", "eyeBlinkRight - Z-Score (Peak Detection)"),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Plot 1: Smoothed signal
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pred_smooth,
            mode='lines',
            name='Prediction (smoothed)',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Plot 2: Z-score signal
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pred_zscore,
            mode='lines',
            name='Prediction (z-score)',
            line=dict(color='red', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add matched peaks (green)
    if len(matched_pred) > 0:
        matched_values = pred_zscore[matched_pred]
        fig.add_trace(
            go.Scatter(
                x=matched_pred,
                y=matched_values,
                mode='markers',
                name='Matched peaks',
                marker=dict(color='green', size=12, symbol='circle'),
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Add unmatched peaks (orange)
    if len(unmatched_pred) > 0:
        unmatched_values = pred_zscore[unmatched_pred]
        fig.add_trace(
            go.Scatter(
                x=unmatched_pred,
                y=unmatched_values,
                mode='markers',
                name='Unmatched peaks',
                marker=dict(color='orange', size=12, symbol='circle'),
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Add GT peaks for reference (blue diamonds)
    if len(gt_peaks) > 0:
        gt_values = pred_zscore[gt_peaks]
        fig.add_trace(
            go.Scatter(
                x=gt_peaks,
                y=gt_values,
                mode='markers',
                name='GT peaks',
                marker=dict(color='blue', size=10, symbol='diamond'),
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Highlight the current peak being reviewed
    fig.add_vline(
        x=current_peak_frame,
        line_dash="dash",
        line_color="purple",
        line_width=3,
        annotation_text=f"Current Peak (frame {current_peak_frame})",
        annotation_position="top",
        row=1, col=1
    )
    
    fig.add_vline(
        x=current_peak_frame,
        line_dash="dash",
        line_color="purple",
        line_width=3,
        row=2, col=1
    )
    
    # Highlight region around current peak (±20 frames)
    peak_start = max(0, current_peak_frame - 20)
    peak_end = min(len(pred_smooth), current_peak_frame + 20)
    
    fig.add_vrect(
        x0=peak_start,
        x1=peak_end,
        fillcolor="purple",
        opacity=0.15,
        layer="below",
        line_width=0,
        row=1, col=1
    )
    
    fig.add_vrect(
        x0=peak_start,
        x1=peak_end,
        fillcolor="purple",
        opacity=0.15,
        layer="below",
        line_width=0,
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Frame (25 fps)", row=1, col=1)
    fig.update_xaxes(title_text="Frame (25 fps)", row=2, col=1)
    fig.update_yaxes(title_text="Blendshape Value", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    
    fig.update_layout(
        title=f"Prediction Signal with All Peaks<br><sub>Total: {len(pred_peaks)} pred peaks ({len(matched_pred)} matched, {len(unmatched_pred)} unmatched) | {len(gt_peaks)} GT peaks</sub>",
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


@app.callback(
    [Output('video-player', 'src'),
     Output('sample-info', 'children'),
     Output('prediction-plot', 'figure'),
     Output('progress-text', 'children'),
     Output('progress-bar', 'value'),
     Output('current-index', 'data')],
    [Input('btn-prev', 'n_clicks'),
     Input('btn-next', 'n_clicks')],
    [State('current-index', 'data')]
)
def update_display(btn_prev, btn_next, current_index):
    """Update display based on navigation."""
    ctx = dash.callback_context
    
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'btn-prev':
            current_index = max(0, current_index - 1)
        elif button_id == 'btn-next':
            current_index = min(len(video_data) - 1, current_index + 1)
    
    # Get current data
    data = video_data[current_index]
    info = data['info']
    
    # Create info display
    info_display = [
        html.P([
            html.Strong("Run Path: "), info['run_path']
        ]),
        html.P([
            html.Strong("TAR ID: "), f"{info['tar_id']} | ",
            html.Strong("Side: "), f"{info['side']} | ",
            html.Strong("Peak Frame (25fps): "), f"{info['peak_frame_in_file']} | ",
            html.Strong("Video Frame (30fps): "), f"{info['video_frame_in_file']} | ",
            html.Strong("Peak Value: "), f"{info['peak_value']:.4f}"
        ]),
        html.P([
            html.Strong("Peak Stats: "),
            f"{len(data['peak_data']['matched_pred'])} matched, ",
            f"{len(data['peak_data']['unmatched_pred'])} unmatched pred peaks | ",
            f"{len(data['peak_data']['gt_peaks'])} GT peaks"
        ]),
    ]
    
    # Progress
    progress_value = (current_index + 1) / len(video_data) * 100
    progress_text = f"Sample {current_index + 1} of {len(video_data)}"
    
    # Load video
    video_src = get_video_base64(data['path'])
    
    # Create prediction plot
    pred_fig = create_prediction_plot(data['peak_data'], data['peak_frame'])
    
    return video_src, info_display, pred_fig, progress_text, progress_value, current_index


# Add keyboard shortcuts
app.clientside_callback(
    """
    function(id) {
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowLeft') {
                document.getElementById('btn-prev').click();
            } else if (event.key === 'ArrowRight') {
                document.getElementById('btn-next').click();
            }
        });
        return '';
    }
    """,
    Output('video-player', 'id'),
    Input('video-player', 'id')
)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Server starting...")
    print("Open your browser and go to: http://localhost:8053")
    print("Use arrow keys to navigate: ← Previous | → Next")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8053)

