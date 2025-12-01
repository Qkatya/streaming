#!/usr/bin/env python3
"""
Web-based GUI application for labeling video snippets as blink or no-blink using Plotly Dash.
Runs in browser - perfect for VM environments.
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
import sys

# Configuration
PICKLE_FILE = "unmatched_pred_peaks_prom0.5000.pkl"  # Updated to use prominence-based pickle
VIDEO_SNIPPETS_DIR = Path("video_snippets_marked")
ALL_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
INFERENCE_OUTPUTS_PATH = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_inference")
SPLIT_DF_PATH = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'

MODEL_NAME = 'causal_preprocessor_encoder_with_smile'
HISTORY_SIZE = 800
LOOKAHEAD_SIZE = 1
VIDEO_FPS = 30
BLENDSHAPE_FPS = 25

# Import data loading functions and BlinkAnalyzer
sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes
from blink_analyzer import BlinkAnalyzer

# Load data
df = pd.read_pickle(PICKLE_FILE)

# Add 'is_blink' column if it doesn't exist
if 'is_blink' not in df.columns:
    df['is_blink'] = None

# Sort by global_frame to maintain order
df = df.sort_values('global_frame').reset_index(drop=True)

# Load split dataframe
split_df = pd.read_pickle(SPLIT_DF_PATH)

def load_gt_pred_data(row):
    """Load GT and prediction data for a given row."""
    run_path = row['run_path']
    
    if 'tar_id' in row and 'side' in row:
        tar_id = row['tar_id']
        side = row['side']
    else:
        matching_rows = split_df[split_df['run_path'] == run_path]
        if len(matching_rows) == 0:
            return None, None, None
        split_row = matching_rows.iloc[0]
        tar_id = split_row['tar_id']
        side = split_row['side']
    
    # Load GT blendshapes
    full_run_path = ALL_DATA_PATH / run_path
    try:
        gt_blendshapes = load_ground_truth_blendshapes(full_run_path, downsample=True)
    except Exception as e:
        return None, None, None
    
    # Load prediction
    pred_file = INFERENCE_OUTPUTS_PATH / run_path / f"pred_{MODEL_NAME}_H{HISTORY_SIZE}_LA{LOOKAHEAD_SIZE}_{tar_id}.npy"
    
    if not pred_file.exists():
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
    gt_th = None
    model_th = None
    quantizer = 8
    
    prominence_gt = 1.0
    prominence_pred = 0.5000
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
    
    # Run blink analysis
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
video_files = []
for idx, row in df.iterrows():
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
    
    video_files.append({
        'path': video_path,
        'df_index': idx,
        'info': row.to_dict(),
        'gt_blendshapes': gt_bs,
        'pred_blendshapes': pred_bs,
        'peak_frame': peak_frame,
        'detection_result': detection_result
    })

print(f"\n{'='*80}")
print(f"BLINK LABELING TOOL - WEB VERSION")
print(f"{'='*80}")
print(f"Found {len(video_files)} videos to label")
print(f"Starting web server...")
print(f"{'='*80}\n")

def create_blendshape_plot(gt_bs, pred_bs, peak_frame):
    """Create plot of GT and prediction for eyeBlinkRight (index 10)."""
    # Extract eyeBlinkRight (index 10)
    gt_blink = gt_bs[:, 10]
    pred_blink = pred_bs[:, 10]
    
    # Apply smoothing
    gt_smooth = savgol_filter(gt_blink, 9, 2, mode='interp')
    pred_smooth = savgol_filter(pred_blink, 9, 2, mode='interp')
    
    # peak_frame is already in blendshape space (25 fps) - no conversion needed!
    peak_frame_blendshape = peak_frame
    
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
    
    # Add vertical line at peak frame
    fig.add_vline(
        x=peak_frame_blendshape,
        line_dash="dash",
        line_color="green",
        line_width=3,
        annotation_text=f"Peak (frame {peak_frame_blendshape})",
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
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_detection_plot(detection_result):
    """Create detection matching plot from blink analysis result."""
    if not detection_result or 'plots' not in detection_result:
        return go.Figure()
    
    plots = detection_result['plots']
    
    # Use the derivatives plot which shows z-scored signals with peak matching
    if 'derivatives' in plots:
        return plots['derivatives']
    
    return go.Figure()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dcc.Store(id='current-index', data=0),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Blink Labeling Tool", className="text-center mb-4"),
        ])
    ]),
    
    # Progress bar
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
                    html.H5("Video Information", className="card-title"),
                    html.Div(id='video-info'),
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
    
    # Labeling buttons
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "✓ WAS BLINK",
                id='btn-blink',
                color="success",
                size="lg",
                className="w-100",
                style={'fontSize': '24px', 'padding': '20px'}
            )
        ], width=4),
        dbc.Col([
            dbc.Button(
                "✗ NO BLINK",
                id='btn-no-blink',
                color="danger",
                size="lg",
                className="w-100",
                style={'fontSize': '24px', 'padding': '20px'}
            )
        ], width=4),
        dbc.Col([
            dbc.Button(
                "? DON'T KNOW",
                id='btn-dont-know',
                color="warning",
                size="lg",
                className="w-100",
                style={'fontSize': '24px', 'padding': '20px'}
            )
        ], width=4),
    ], className="mb-3"),
    
    # Blendshape plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='blendshape-plot')
        ])
    ], className="mb-4"),
    
    # Detection plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='detection-plot')
        ])
    ], className="mb-4"),
    
    # Navigation buttons
    dbc.Row([
        dbc.Col([
            dbc.Button("← Previous", id='btn-prev', color="secondary", className="w-100")
        ], width=3),
        dbc.Col([
            dbc.Button("Skip →", id='btn-skip', color="secondary", className="w-100")
        ], width=3),
        dbc.Col([
            dbc.Button("💾 Save Progress", id='btn-save', color="info", className="w-100")
        ], width=3),
        dbc.Col([
            html.Div(id='save-status', className="text-center")
        ], width=3),
    ], className="mb-4"),
    
    # Keyboard shortcuts info
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.Strong("Keyboard Shortcuts: "),
                "B = Blink | N = No Blink | U = Don't Know | ← = Previous | → = Skip"
            ], color="info", className="text-center")
        ])
    ]),
    
], fluid=True, style={'maxWidth': '1200px', 'padding': '20px'})

def get_video_base64(video_path):
    """Read video file and encode as base64."""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode('ascii')
    return f"data:video/mp4;base64,{video_base64}"

@app.callback(
    [Output('video-player', 'src'),
     Output('video-info', 'children'),
     Output('progress-text', 'children'),
     Output('progress-bar', 'value'),
     Output('current-index', 'data'),
     Output('blendshape-plot', 'figure'),
     Output('detection-plot', 'figure')],
    [Input('btn-blink', 'n_clicks'),
     Input('btn-no-blink', 'n_clicks'),
     Input('btn-dont-know', 'n_clicks'),
     Input('btn-prev', 'n_clicks'),
     Input('btn-skip', 'n_clicks')],
    [State('current-index', 'data')]
)
def update_video(btn_blink, btn_no_blink, btn_dont_know, btn_prev, btn_skip, current_index):
    """Update video display based on button clicks."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle button actions
    if button_id == 'btn-blink' and btn_blink:
        # Label as blink and move to next
        video_info = video_files[current_index]
        df.loc[video_info['df_index'], 'is_blink'] = True
        current_index = min(current_index + 1, len(video_files) - 1)
    elif button_id == 'btn-no-blink' and btn_no_blink:
        # Label as no blink and move to next
        video_info = video_files[current_index]
        df.loc[video_info['df_index'], 'is_blink'] = False
        current_index = min(current_index + 1, len(video_files) - 1)
    elif button_id == 'btn-dont-know' and btn_dont_know:
        # Label as don't know (use string 'unknown') and move to next
        video_info = video_files[current_index]
        df.loc[video_info['df_index'], 'is_blink'] = 'unknown'
        current_index = min(current_index + 1, len(video_files) - 1)
    elif button_id == 'btn-prev' and btn_prev:
        # Move to previous
        current_index = max(current_index - 1, 0)
    elif button_id == 'btn-skip' and btn_skip:
        # Skip to next
        current_index = min(current_index + 1, len(video_files) - 1)
    
    # Get current video info
    video_info = video_files[current_index]
    info = video_info['info']
    
    # Get current label
    current_label = df.loc[video_info['df_index'], 'is_blink']
    if current_label is None:
        label_text = "Not labeled yet"
        label_color = "secondary"
    elif current_label == 'unknown':
        label_text = "DON'T KNOW"
        label_color = "warning"
    elif current_label:
        label_text = "BLINK"
        label_color = "success"
    else:
        label_text = "NO BLINK"
        label_color = "danger"
    
    # Create info display
    info_display = [
        html.P([
            html.Strong("Global Frame: "), f"{info['global_frame']} | ",
            html.Strong("File Index: "), f"{info['file_index']} | ",
            html.Strong("Local Frame: "), f"{info['local_frame']} | ",
            html.Strong("Peak Frame: "), f"{video_info['peak_frame']} | ",
            html.Strong("Timestamp: "), f"{info['timestamp_seconds']:.2f}s"
        ]),
        html.P([
            html.Strong("Run Path: "), info['run_path']
        ]),
        html.P([
            html.Strong("Current Label: "),
            dbc.Badge(label_text, color=label_color, className="ms-2")
        ])
    ]
    
    # Progress
    progress_value = (current_index + 1) / len(video_files) * 100
    progress_text = f"Video {current_index + 1} of {len(video_files)}"
    
    # Load video
    video_src = get_video_base64(video_info['path'])
    
    # Create plots
    blendshape_fig = create_blendshape_plot(
        video_info['gt_blendshapes'],
        video_info['pred_blendshapes'],
        video_info['peak_frame']
    )
    
    detection_fig = create_detection_plot(video_info['detection_result'])
    
    return video_src, info_display, progress_text, progress_value, current_index, blendshape_fig, detection_fig

@app.callback(
    Output('save-status', 'children'),
    Input('btn-save', 'n_clicks'),
    prevent_initial_call=True
)
def save_progress(n_clicks):
    """Save progress to pickle file."""
    if n_clicks:
        try:
            # Save back to the original pickle file to preserve labels
            df.to_pickle(PICKLE_FILE)
            labeled_count = df['is_blink'].notna().sum()
            return dbc.Alert(
                f"Saved to {PICKLE_FILE}! {labeled_count}/{len(df)} labeled",
                color="success",
                duration=3000
            )
        except Exception as e:
            return dbc.Alert(f"Error: {e}", color="danger", duration=3000)
    return ""

# Add keyboard shortcuts support
app.clientside_callback(
    """
    function(id) {
        document.addEventListener('keydown', function(event) {
            if (event.key === 'b' || event.key === 'B') {
                document.getElementById('btn-blink').click();
            } else if (event.key === 'n' || event.key === 'N') {
                document.getElementById('btn-no-blink').click();
            } else if (event.key === 'u' || event.key === 'U') {
                document.getElementById('btn-dont-know').click();
            } else if (event.key === 'ArrowLeft') {
                document.getElementById('btn-prev').click();
            } else if (event.key === 'ArrowRight') {
                document.getElementById('btn-skip').click();
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
    print("Open your browser and go to: http://localhost:8051")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8051)
