"""
Interactive Dash application for editing ground truth peaks.
Displays z-score comparison with peak matching and allows adding/removing GT peaks.
"""
import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from scipy.stats import zscore
from scipy.signal import find_peaks, filtfilt

# Import from existing modules
from blink_analyzer import BlinkAnalyzer
from visualization import _match_peaks

# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW_SIZE_SECONDS = 50  # Size of each window in seconds
SAMPLE_RATE = 30.0  # Hz
WINDOW_SIZE_FRAMES = int(WINDOW_SIZE_SECONDS * SAMPLE_RATE)  # 1500 frames

# File paths for saving modifications
REMOVED_PEAKS_FILE = "removed_peaks.pkl"
ADDED_PEAKS_FILE = "added_peaks.pkl"
FINAL_GT_PEAKS_FILE = "final_gt_peaks.pkl"

# ============================================================================
# DATA MANAGEMENT CLASS
# ============================================================================

class PeakEditor:
    """Manages GT peaks and their modifications."""
    
    def __init__(self, gt_signal, pred_signal, sample_rate=30.0):
        """
        Initialize the peak editor.
        
        Parameters:
        -----------
        gt_signal : np.ndarray
            Ground truth blink signal
        pred_signal : np.ndarray
            Predicted blink signal
        sample_rate : float
            Sampling rate in Hz
        """
        self.gt_signal = gt_signal
        self.pred_signal = pred_signal
        self.sample_rate = sample_rate
        
        # Filter GT signal
        b = np.ones(3) / 3
        self.gt_filtered = filtfilt(b, 1, gt_signal)
        
        # Calculate z-scores
        self.gt_zscore = zscore(self.gt_filtered)
        self.pred_zscore = zscore(pred_signal)
        
        # Detect initial peaks
        self.original_gt_peaks = self._detect_initial_peaks(
            self.gt_filtered, height_threshold=0.5, min_prominence=1.0
        )
        
        # Track modifications
        self.removed_peaks = set()  # Indices of removed peaks
        self.added_peaks = set()    # Indices of added peaks
        
        # Load previous modifications if they exist
        self._load_modifications()
    
    def _detect_initial_peaks(self, signal, height_threshold=0.5, min_prominence=1.0):
        """Detect initial GT peaks using z-score."""
        signal_zscore = zscore(signal)
        peaks, _ = find_peaks(signal_zscore, height=height_threshold, prominence=min_prominence)
        return peaks
    
    def _load_modifications(self):
        """Load previously saved modifications."""
        if os.path.exists(REMOVED_PEAKS_FILE):
            with open(REMOVED_PEAKS_FILE, 'rb') as f:
                self.removed_peaks = pickle.load(f)
                print(f"Loaded {len(self.removed_peaks)} removed peaks")
        
        if os.path.exists(ADDED_PEAKS_FILE):
            with open(ADDED_PEAKS_FILE, 'rb') as f:
                self.added_peaks = pickle.load(f)
                print(f"Loaded {len(self.added_peaks)} added peaks")
    
    def save_modifications(self):
        """Save current modifications to pickle files."""
        with open(REMOVED_PEAKS_FILE, 'wb') as f:
            pickle.dump(self.removed_peaks, f)
        
        with open(ADDED_PEAKS_FILE, 'wb') as f:
            pickle.dump(self.added_peaks, f)
        
        # Save final GT peaks
        final_peaks = self.get_current_gt_peaks()
        with open(FINAL_GT_PEAKS_FILE, 'wb') as f:
            pickle.dump(final_peaks, f)
        
        print(f"Saved modifications: {len(self.removed_peaks)} removed, {len(self.added_peaks)} added")
        print(f"Final GT peaks: {len(final_peaks)} peaks")
    
    def get_current_gt_peaks(self):
        """Get current GT peaks after modifications."""
        # Start with original peaks
        current_peaks = set(self.original_gt_peaks)
        
        # Remove peaks that were marked for removal
        current_peaks -= self.removed_peaks
        
        # Add peaks that were added
        current_peaks |= self.added_peaks
        
        # Convert to sorted array
        return np.array(sorted(current_peaks))
    
    def toggle_peak(self, frame_idx):
        """
        Toggle a peak at the given frame index.
        If it's a GT peak, remove it. If it's not, add it.
        
        Parameters:
        -----------
        frame_idx : int
            Frame index to toggle
        
        Returns:
        --------
        action : str
            'added' or 'removed'
        """
        current_peaks = self.get_current_gt_peaks()
        
        # Check if this frame is close to an existing peak (within 5 frames = 0.17 seconds)
        if len(current_peaks) > 0:
            distances = np.abs(current_peaks - frame_idx)
            closest_peak_idx = np.argmin(distances)
            closest_peak = current_peaks[closest_peak_idx]
            
            if distances[closest_peak_idx] <= 5:
                # Remove this peak
                if closest_peak in self.added_peaks:
                    self.added_peaks.remove(closest_peak)
                else:
                    self.removed_peaks.add(closest_peak)
                return 'removed'
        
        # Add new peak
        self.added_peaks.add(frame_idx)
        return 'added'
    
    def get_window_data(self, window_idx):
        """
        Get data for a specific window.
        
        Parameters:
        -----------
        window_idx : int
            Window index (0-based)
        
        Returns:
        --------
        dict with window data
        """
        start_frame = window_idx * WINDOW_SIZE_FRAMES
        end_frame = min(start_frame + WINDOW_SIZE_FRAMES, len(self.gt_signal))
        
        # Get current peaks in this window (after modifications)
        current_peaks = self.get_current_gt_peaks()
        window_gt_peaks = current_peaks[
            (current_peaks >= start_frame) & (current_peaks < end_frame)
        ] - start_frame
        
        # Get pred peaks in this window
        pred_peaks = self._detect_initial_peaks(
            self.pred_signal[start_frame:end_frame],
            height_threshold=0.5,
            min_prominence=1.0
        )
        
        # Create matches manually based on current GT peaks
        matched_gt = []
        matched_pred = []
        unmatched_gt = []
        unmatched_pred = list(pred_peaks)
        
        # Match each GT peak with closest pred peak within max_offset
        for gt_peak in window_gt_peaks:
            if len(pred_peaks) > 0:
                distances = np.abs(pred_peaks - gt_peak)
                close_peaks = np.where(distances <= 10)[0]
                
                if len(close_peaks) > 0:
                    closest_idx = close_peaks[np.argmin(distances[close_peaks])]
                    closest_pred_peak = pred_peaks[closest_idx]
                    
                    matched_gt.append(gt_peak)
                    matched_pred.append(closest_pred_peak)
                    
                    if closest_pred_peak in unmatched_pred:
                        unmatched_pred.remove(closest_pred_peak)
                else:
                    unmatched_gt.append(gt_peak)
            else:
                unmatched_gt.append(gt_peak)
        
        matches = {
            'matched_gt': matched_gt,
            'matched_pred': matched_pred,
            'unmatched_gt': unmatched_gt,
            'unmatched_pred': unmatched_pred,
            'gt_peaks': window_gt_peaks,
            'pred_peaks': pred_peaks
        }
        
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'gt_signal': self.gt_filtered[start_frame:end_frame],
            'pred_signal': self.pred_signal[start_frame:end_frame],
            'gt_zscore': self.gt_zscore[start_frame:end_frame],
            'pred_zscore': self.pred_zscore[start_frame:end_frame],
            'gt_peaks': window_gt_peaks,
            'pred_peaks': pred_peaks,
            'matches': matches
        }
    
    def get_num_windows(self):
        """Get total number of windows."""
        return int(np.ceil(len(self.gt_signal) / WINDOW_SIZE_FRAMES))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_window_figure(window_data, sample_rate):
    """Create the main figure for a window."""
    
    # Create time axis
    time = np.arange(len(window_data['gt_signal'])) / sample_rate
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Plot z-scores
    fig.add_trace(go.Scatter(
        x=time,
        y=window_data['gt_zscore'],
        mode='lines',
        name='GT Z-Score',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time,
        y=window_data['pred_zscore'],
        mode='lines',
        name='Pred Z-Score',
        line=dict(color='red', width=2)
    ))
    
    # Add matched GT peaks
    if len(window_data['matches']['matched_gt']) > 0:
        matched_gt_times = time[window_data['matches']['matched_gt']]
        matched_gt_values = window_data['gt_zscore'][window_data['matches']['matched_gt']]
        fig.add_trace(go.Scatter(
            x=matched_gt_times,
            y=matched_gt_values,
            mode='markers',
            name='Matched GT Peaks',
            marker=dict(color='green', size=12, symbol='circle', line=dict(color='darkgreen', width=2))
        ))
    
    # Add matched pred peaks
    if len(window_data['matches']['matched_pred']) > 0:
        matched_pred_times = time[window_data['matches']['matched_pred']]
        matched_pred_values = window_data['pred_zscore'][window_data['matches']['matched_pred']]
        fig.add_trace(go.Scatter(
            x=matched_pred_times,
            y=matched_pred_values,
            mode='markers',
            name='Matched Pred Peaks',
            marker=dict(color='lightgreen', size=10, symbol='x')
        ))
    
    # Add unmatched GT peaks
    if len(window_data['matches']['unmatched_gt']) > 0:
        unmatched_gt_times = time[window_data['matches']['unmatched_gt']]
        unmatched_gt_values = window_data['gt_zscore'][window_data['matches']['unmatched_gt']]
        fig.add_trace(go.Scatter(
            x=unmatched_gt_times,
            y=unmatched_gt_values,
            mode='markers',
            name='Unmatched GT Peaks',
            marker=dict(color='orange', size=12, symbol='circle', line=dict(color='darkorange', width=2))
        ))
    
    # Add unmatched pred peaks
    if len(window_data['matches']['unmatched_pred']) > 0:
        unmatched_pred_times = time[window_data['matches']['unmatched_pred']]
        unmatched_pred_values = window_data['pred_zscore'][window_data['matches']['unmatched_pred']]
        fig.add_trace(go.Scatter(
            x=unmatched_pred_times,
            y=unmatched_pred_values,
            mode='markers',
            name='Unmatched Pred Peaks',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # Update layout
    fig.update_layout(
        title='Z-Score Comparison with Peak Matching (Click to Add/Remove GT Peaks)',
        xaxis_title='Time (s)',
        yaxis_title='Z-Score',
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# ============================================================================
# DASH APP
# ============================================================================

def create_app(gt_signal, pred_signal, sample_rate=30.0):
    """Create and configure the Dash app."""
    
    # Initialize peak editor
    editor = PeakEditor(gt_signal, pred_signal, sample_rate)
    
    # Create Dash app
    app = Dash(__name__)
    
    # App layout
    app.layout = html.Div([
        html.H1("Ground Truth Peak Editor", style={'textAlign': 'center'}),
        
        # Info panel
        html.Div([
            html.Div(id='info-panel', style={
                'padding': '20px',
                'backgroundColor': '#f0f0f0',
                'borderRadius': '5px',
                'marginBottom': '20px'
            }),
        ]),
        
        # Navigation controls
        html.Div([
            html.Button('◄◄ Previous Window', id='prev-button', n_clicks=0, 
                       style={'fontSize': '16px', 'padding': '10px 20px', 'marginRight': '10px'}),
            html.Span(id='window-display', style={'fontSize': '18px', 'fontWeight': 'bold', 'margin': '0 20px'}),
            html.Button('Next Window ►►', id='next-button', n_clicks=0,
                       style={'fontSize': '16px', 'padding': '10px 20px', 'marginLeft': '10px'}),
            html.Button('💾 Save Changes', id='save-button', n_clicks=0,
                       style={'fontSize': '16px', 'padding': '10px 20px', 'marginLeft': '30px', 
                              'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Instructions
        html.Div([
            html.P("Click on the graph to add or remove GT peaks. Click near an existing peak to remove it, click elsewhere to add a new peak.",
                  style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
        ]),
        
        # Main graph
        dcc.Graph(id='main-graph', style={'height': '600px'}),
        
        # Hidden div to store current window index
        html.Div(id='current-window', children='0', style={'display': 'none'}),
        
        # Status message
        html.Div(id='status-message', style={
            'textAlign': 'center',
            'padding': '10px',
            'marginTop': '10px',
            'fontSize': '14px',
            'fontWeight': 'bold'
        })
    ])
    
    # Callback for updating the graph
    @app.callback(
        [Output('main-graph', 'figure'),
         Output('window-display', 'children'),
         Output('info-panel', 'children'),
         Output('current-window', 'children'),
         Output('status-message', 'children')],
        [Input('prev-button', 'n_clicks'),
         Input('next-button', 'n_clicks'),
         Input('main-graph', 'clickData')],
        [State('current-window', 'children')]
    )
    def update_graph(prev_clicks, next_clicks, click_data, current_window_str):
        ctx = callback_context
        
        # Get current window index
        current_window = int(current_window_str)
        status_msg = ""
        
        # Handle navigation
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'prev-button' and current_window > 0:
                current_window -= 1
            elif button_id == 'next-button' and current_window < editor.get_num_windows() - 1:
                current_window += 1
            elif button_id == 'main-graph' and click_data is not None:
                # Handle click on graph
                clicked_x = click_data['points'][0]['x']
                # Convert time to frame index (relative to window)
                clicked_frame = int(clicked_x * sample_rate)
                
                # Get window data to convert to absolute frame
                window_data = editor.get_window_data(current_window)
                absolute_frame = window_data['start_frame'] + clicked_frame
                
                # Toggle peak
                action = editor.toggle_peak(absolute_frame)
                if action == 'removed':
                    status_msg = f"✓ Removed peak at frame {absolute_frame} (time: {absolute_frame/sample_rate:.2f}s)"
                else:
                    status_msg = f"✓ Added peak at frame {absolute_frame} (time: {absolute_frame/sample_rate:.2f}s)"
        
        # Get window data
        window_data = editor.get_window_data(current_window)
        
        # Create figure
        fig = create_window_figure(window_data, sample_rate)
        
        # Create window display text
        window_text = f"Window {current_window + 1} / {editor.get_num_windows()}"
        
        # Create info panel
        current_peaks = editor.get_current_gt_peaks()
        info_text = [
            html.P(f"Total GT Peaks: {len(current_peaks)}"),
            html.P(f"Removed Peaks: {len(editor.removed_peaks)}"),
            html.P(f"Added Peaks: {len(editor.added_peaks)}"),
            html.P(f"Window: Frames {window_data['start_frame']} - {window_data['end_frame']}"),
            html.P(f"GT Peaks in Window: {len(window_data['gt_peaks'])}"),
            html.P(f"Matched Peaks: {len(window_data['matches']['matched_gt'])}"),
            html.P(f"Unmatched GT Peaks: {len(window_data['matches']['unmatched_gt'])}"),
            html.P(f"Unmatched Pred Peaks: {len(window_data['matches']['unmatched_pred'])}")
        ]
        
        return fig, window_text, info_text, str(current_window), status_msg
    
    # Callback for save button
    @app.callback(
        Output('status-message', 'children', allow_duplicate=True),
        Input('save-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def save_changes(n_clicks):
        editor.save_modifications()
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"💾 Changes saved successfully at {timestamp}"
    
    return app

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def load_data_example():
    """
    Load example data for testing.
    Replace this with your actual data loading logic.
    """
    # This is a placeholder - you'll need to load your actual data
    # For example, from the blinking_split_analysis.py workflow
    
    print("Please provide gt_signal and pred_signal arrays")
    print("Example usage:")
    print("  from peak_editor_app import create_app")
    print("  app = create_app(gt_signal, pred_signal)")
    print("  app.run_server(debug=True)")
    
    return None, None

def main():
    """Main entry point."""
    # Load your data here
    gt_signal, pred_signal = load_data_example()
    
    if gt_signal is None or pred_signal is None:
        print("\nTo use this app, you need to provide GT and prediction signals.")
        print("See the load_data_example() function for guidance.")
        return
    
    # Create and run app
    app = create_app(gt_signal, pred_signal)
    app.run(debug=True, port=8050)

if __name__ == '__main__':
    main()

