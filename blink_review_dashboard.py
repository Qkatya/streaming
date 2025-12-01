#!/usr/bin/env python3
"""
Dashboard to review manually labeled samples with GT, prediction plots, and detection matching.
Shows video, blendshape plots for eyeBlinkRight, and the full detection comparison plot.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
from scipy.signal import savgol_filter

# Configuration
# NOTE: Update this to match the prominence value from your analysis
# The file is now named with prominence instead of threshold (e.g., "unmatched_pred_peaks_prom0.5000.pkl")
PICKLE_FILE = "unmatched_pred_peaks_prom0.5000.pkl"  # Update with your prominence value
VIDEO_SNIPPETS_DIR = Path("video_snippets_marked")
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'

MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1
VIDEO_FPS = 30  # Video frame rate
BLENDSHAPE_FPS = 25  # Blendshape frame rate (downsampled)

# Import data loading functions and BlinkAnalyzer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes
from blink_analyzer import BlinkAnalyzer

# Load all unmatched predictions
df = pd.read_pickle(PICKLE_FILE)

# Add 'is_blink' column if it doesn't exist
if 'is_blink' not in df.columns:
    df['is_blink'] = None

# Sort by global_frame to maintain the order from the z-scored signal plot
df = df.sort_values('global_frame').reset_index(drop=True)

# Use all samples (not just labeled ones)
all_samples = df.copy()

# Count by label type
labeled_count = df['is_blink'].notna().sum()
unlabeled_count = df['is_blink'].isna().sum()
blink_count = (df['is_blink'] == True).sum()
no_blink_count = (df['is_blink'] == False).sum()
dont_know_count = (df['is_blink'] == 'unknown').sum()

print(f"\n{'='*80}")
print(f"BLINK REVIEW DASHBOARD - ALL UNMATCHED PREDICTIONS")
print(f"{'='*80}")
print(f"Total samples: {len(all_samples)}")
print(f"  - Labeled: {labeled_count}")
print(f"    - BLINK: {blink_count}")
print(f"    - NO BLINK: {no_blink_count}")
print(f"    - DON'T KNOW: {dont_know_count}")
print(f"  - Unlabeled: {unlabeled_count}")
print(f"{'='*80}\n")

if len(all_samples) == 0:
    print("No samples found.")
    exit(1)

# Load split dataframe to get run_path info
split_df = pd.read_pickle(SPLIT_DF_PATH)

def load_gt_pred_data(row):
    """Load GT and prediction data for a given row."""
    # Get run_path, tar_id, and side directly from the row (they're in the pickle file)
    run_path = row['run_path']
    
    # Check if tar_id and side are in the row (new pickle format)
    if 'tar_id' in row and 'side' in row:
        tar_id = row['tar_id']
        side = row['side']
    else:
        # Fallback: look up in split_df (for old pickle files)
        print(f"Warning: tar_id/side not in pickle file, looking up in split_df for {run_path}")
        matching_rows = split_df[split_df['run_path'] == run_path]
        
        if len(matching_rows) == 0:
            print(f"Error: run_path '{run_path}' not found in split_df")
            return None, None, None
        
        split_row = matching_rows.iloc[0]
        tar_id = split_row['tar_id']
        side = split_row['side']
    
    # Load GT blendshapes
    full_run_path = ALL_DATA_PATH / run_path
    try:
        gt_blendshapes = load_ground_truth_blendshapes(full_run_path, downsample=True)
    except Exception as e:
        print(f"Error loading GT for {run_path}: {e}")
        return None, None, None
    
    # Load prediction
    pred_file = INFERENCE_OUTPUTS_PATH / run_path / f"pred_{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}_{tar_id}.npy"
    
    if not pred_file.exists():
        print(f"Prediction file not found: {pred_file}")
        return None, None, None
    
    pred_blendshapes = np.load(pred_file)
    
    # Align lengths
    gt_len = gt_blendshapes.shape[0]
    pred_len = pred_blendshapes.shape[0]
    
    if pred_len < gt_len:
        padding = np.zeros((gt_len - pred_len, pred_blendshapes.shape[1]))
        pred_blendshapes = np.concatenate([pred_blendshapes, padding], axis=0)
    elif pred_len > gt_len:
        pred_blendshapes = pred_blendshapes[:gt_len]
    
    return gt_blendshapes, pred_blendshapes, row['local_frame']

def run_blink_detection_for_sample(gt_bs, pred_bs):
    """Run blink detection analysis for a single sample."""
    # Use prominence-based detection (matching blinking_split_analysis.py)
    gt_th = None  # Disable absolute height threshold
    model_th = None  # Disable absolute height threshold
    quantizer = 8
    
    # Prominence parameters (matching the ROC analysis)
    prominence_gt = 1.0
    prominence_pred = 0.5000  # This should match the value from your pickle filename
    distance = 5
    width_min = 2
    width_max = 20
    use_derivative_validation = True
    
    # Create significance mask
    threshold_factor = 0.25
    significant_gt = np.zeros(gt_bs.shape, dtype=bool)
    for i in range(gt_bs.shape[1]):
        th = threshold_factor * np.max(gt_bs[:, i])
        significant_gt[:, i] = np.abs(gt_bs[:, i]) > th
    
    # Run blink analysis with prominence-based detection
    analyzer = BlinkAnalyzer()
    result = analyzer.analyze_blinks(
        gt_th=gt_th,
        model_th=model_th,
        blendshapes_list=[gt_bs],
        pred_blends_list=[pred_bs],
        max_offset=quantizer,
        significant_gt_movenents=[significant_gt],
        prominence_gt=prominence_gt,
        prominence_pred=prominence_pred,
        distance=distance,
        width_min=width_min,
        width_max=width_max,
        use_derivative_validation=use_derivative_validation
    )
    
    return result

# Preload all data
print("Loading GT and prediction data for all samples...")
video_data = []
for idx, row in all_samples.iterrows():
    # Get video path
    global_frame = row['global_frame']
    file_index = row['file_index']
    local_frame = row['local_frame']
    timestamp_seconds = row['timestamp_seconds']
    run_path = row['run_path'].replace('/', '_').replace('\\', '_')
    
    filename = f"{global_frame}_{file_index}_{local_frame}_{timestamp_seconds:.2f}_{run_path}.mp4"
    video_path = VIDEO_SNIPPETS_DIR / filename
    
    if not video_path.exists():
        continue
    
    # Load GT and pred data
    gt_bs, pred_bs, peak_frame = load_gt_pred_data(row)
    
    if gt_bs is None:
        continue
    
    # Run blink detection
    detection_result = run_blink_detection_for_sample(gt_bs, pred_bs)
    
    video_data.append({
        'path': video_path,
        'info': row.to_dict(),
        'gt_blendshapes': gt_bs,
        'pred_blendshapes': pred_bs,
        'peak_frame': peak_frame,
        'label': row['is_blink'],
        'detection_result': detection_result
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
            html.H1("Blink Review Dashboard", className="text-center mb-4"),
            html.H5("Reviewing all unmatched predictions", className="text-center text-muted mb-4"),
        ])
    ]),
    
    # Progress
    dbc.Row([
        dbc.Col([
            html.Div(id='progress-text', className="text-center mb-2"),
            dbc.Progress(id='progress-bar', value=0, className="mb-4"),
        ])
    ]),
    
    # Video info
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
    
    # Blendshape plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='blendshape-plot', style={'height': '500px'})
        ])
    ], className="mb-4"),
    
    # Detection matching plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='detection-plot', style={'height': '500px'})
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

def create_blendshape_plot(gt_bs, pred_bs, peak_frame_video, local_frame):
    """Create plot of GT and prediction for eyeBlinkRight (index 10)."""
    # Extract eyeBlinkRight (index 10)
    gt_blink = gt_bs[:, 10]
    pred_blink = pred_bs[:, 10]
    
    # Apply smoothing
    gt_smooth = savgol_filter(gt_blink, 9, 2, mode='interp')
    pred_smooth = savgol_filter(pred_blink, 9, 2, mode='interp')
    
    # peak_frame_video is actually local_frame which is already in blendshape space (25 fps)
    # No conversion needed!
    peak_frame_blendshape = peak_frame_video
    
    # Create frame indices in blendshape space
    frames = np.arange(len(gt_blink))
    
    # Create figure
    fig = go.Figure()
    
    # Add GT trace
    fig.add_trace(go.Scatter(
        x=frames,
        y=gt_smooth,
        mode='lines',
        name='GT',
        line=dict(color='blue', width=2)
    ))
    
    # Add Prediction trace
    fig.add_trace(go.Scatter(
        x=frames,
        y=pred_smooth,
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=2)
    ))
    
    # Add vertical line at peak frame (converted to blendshape fps)
    fig.add_vline(
        x=peak_frame_blendshape,
        line_dash="dash",
        line_color="green",
        line_width=3,
        annotation_text=f"Peak (BS frame {peak_frame_blendshape})",
        annotation_position="top"
    )
    
    # Highlight region around peak (±8 frames in blendshape space)
    peak_start = max(0, peak_frame_blendshape - 8)
    peak_end = min(len(gt_blink), peak_frame_blendshape + 8)
    
    fig.add_vrect(
        x0=peak_start,
        x1=peak_end,
        fillcolor="yellow",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    
    # Update layout
    fig.update_layout(
        title=f"eyeBlinkRight - GT vs Prediction<br><sub>Peak at blendshape frame {peak_frame_blendshape} (25 fps)</sub>",
        xaxis_title="Blendshape Frame (25 fps)",
        yaxis_title="Blendshape Value",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=500
    )
    
    return fig

def create_detection_plot(detection_result):
    """Create the detection matching plot from BlinkAnalyzer results."""
    # Get the plot from detection result
    if 'plots' in detection_result and len(detection_result['plots']) > 0:
        # Get the eyeBlinkRight plot (should be the second plot)
        plot_key = list(detection_result['plots'].keys())[1] if len(detection_result['plots']) > 1 else list(detection_result['plots'].keys())[0]
        fig = detection_result['plots'][plot_key]
        
        # Update layout for better display
        fig.update_layout(height=500, title_font_size=16)
        return fig
    else:
        # Create empty plot if no detection plot available
        fig = go.Figure()
        fig.add_annotation(
            text="No detection plot available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=500)
        return fig

@app.callback(
    [Output('video-player', 'src'),
     Output('sample-info', 'children'),
     Output('blendshape-plot', 'figure'),
     Output('detection-plot', 'figure'),
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
    label = data['label']
    
    # Determine label display
    if label == True:
        label_text = "BLINK"
        label_color = "success"
    elif label == False:
        label_text = "NO BLINK"
        label_color = "danger"
    elif label == 'unknown':
        label_text = "DON'T KNOW"
        label_color = "warning"
    else:
        label_text = "UNLABELED"
        label_color = "secondary"
    
    # Create info display
    info_display = [
        html.P([
            html.Strong("Manual Label: "),
            dbc.Badge(label_text, color=label_color, className="me-3"),
            html.Strong("Global Frame: "), f"{info['global_frame']} | ",
            html.Strong("File Index: "), f"{info['file_index']} | ",
            html.Strong("Local Frame: "), f"{info['local_frame']} | ",
            html.Strong("Peak Frame: "), f"{data['peak_frame']} | ",
            html.Strong("Timestamp: "), f"{info['timestamp_seconds']:.2f}s"
        ]),
        html.P([
            html.Strong("Run Path: "), info['run_path']
        ]),
    ]
    
    # Progress
    progress_value = (current_index + 1) / len(video_data) * 100
    progress_text = f"Sample {current_index + 1} of {len(video_data)}"
    
    # Load video
    video_src = get_video_base64(data['path'])
    
    # Create blendshape plot
    blendshape_fig = create_blendshape_plot(
        data['gt_blendshapes'],
        data['pred_blendshapes'],
        data['peak_frame'],
        info['local_frame']
    )
    
    # Create detection plot
    detection_fig = create_detection_plot(data['detection_result'])
    
    return video_src, info_display, blendshape_fig, detection_fig, progress_text, progress_value, current_index

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
    print("Open your browser and go to: http://localhost:8052")
    print("Use arrow keys to navigate: ← Previous | → Next")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8052)

