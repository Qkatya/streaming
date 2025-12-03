"""
Visualization functions for blink analysis.
"""
import numpy as np
from typing import Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import filtfilt, find_peaks, savgol_filter
from scipy.stats import zscore


def _detect_blink_peaks(signal: np.ndarray, 
                        signal_diff: np.ndarray,
                        height_threshold: float = 0.5,
                        min_prominence: float = 1.0) -> np.ndarray:
    """
    Detect blink peaks using only z-score of the signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        The raw blink signal
    signal_diff : np.ndarray
        The derivative of the blink signal (unused, kept for compatibility)
    height_threshold : float
        Minimum height of peak in z-score units
    min_prominence : float
        Minimum prominence of peak in z-score units
    
    Returns:
    --------
    peaks : np.ndarray
        Indices of detected peaks
    """
    # Smooth the signal using Savitzky-Golay filter before peak detection
    signal_smooth = savgol_filter(signal, window_length=9, polyorder=2, mode='interp')
    
    # Calculate z-score of the smoothed signal
    signal_zscore = zscore(signal_smooth)
    
    # Find peaks in the z-score signal
    peaks, _ = find_peaks(signal_zscore, height=height_threshold, prominence=min_prominence)
    
    return peaks


def _match_peaks(gt_signal: np.ndarray,
                pred_signal: np.ndarray,
                gt_diff: np.ndarray, 
                pred_diff: np.ndarray, 
                max_offset: int = 10,
                gt_th: float = 0.5,
                model_th: float = 0.5,
                gt_prominence: float = 1.0,
                pred_prominence: float = 1.0,
                manual_gt_peaks: np.ndarray = None) -> Dict:
    """
    Match peaks between ground truth and prediction using z-score peaks.
    
    Parameters:
    -----------
    gt_signal : np.ndarray
        Ground truth blink signal
    pred_signal : np.ndarray
        Predicted blink signal
    gt_diff : np.ndarray
        Ground truth derivative (unused, kept for compatibility)
    pred_diff : np.ndarray
        Prediction derivative (unused, kept for compatibility)
    max_offset : int
        Maximum time offset for matching peaks
    gt_th : float
        Height threshold for GT z-score peaks
    model_th : float
        Height threshold for prediction z-score peaks
    gt_prominence : float
        Prominence threshold for GT z-score peaks
    pred_prominence : float
        Prominence threshold for prediction z-score peaks
    manual_gt_peaks : np.ndarray, optional
        Pre-detected GT peaks (if provided, skips automatic detection)
    """
    # Use manual GT peaks if provided, otherwise detect automatically
    if manual_gt_peaks is not None:
        gt_peaks = manual_gt_peaks
    else:
        gt_peaks = _detect_blink_peaks(gt_signal, gt_diff, height_threshold=gt_th, min_prominence=gt_prominence)
    
    pred_peaks = _detect_blink_peaks(pred_signal, pred_diff, height_threshold=model_th, min_prominence=pred_prominence)
    
    matched_gt = []
    matched_pred = []
    unmatched_gt = []
    unmatched_pred = list(pred_peaks)
    
    # Match peaks within max_offset
    for gt_peak in gt_peaks:
        # Find closest prediction peak within max_offset
        distances = np.abs(pred_peaks - gt_peak)
        close_peaks = np.where(distances <= max_offset)[0]
        
        if len(close_peaks) > 0:
            # Find the closest one
            closest_idx = close_peaks[np.argmin(distances[close_peaks])]
            closest_pred_peak = pred_peaks[closest_idx]
            
            matched_gt.append(gt_peak)
            matched_pred.append(closest_pred_peak)
            
            # Remove from unmatched list
            if closest_pred_peak in unmatched_pred:
                unmatched_pred.remove(closest_pred_peak)
        else:
            unmatched_gt.append(gt_peak)
    
    return {
        'matched_gt': matched_gt,
        'matched_pred': matched_pred,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred,
        'gt_peaks': gt_peaks,
        'pred_peaks': pred_peaks
    }


def _create_signals_plot(time_gt: np.ndarray, 
                        time_pred: np.ndarray, 
                        gt_filtered: np.ndarray, 
                        pred_concat: np.ndarray) -> go.Figure:
    """Create plot comparing ground truth and predicted signals."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_gt,
        y=gt_filtered,
        mode='lines',
        name='Ground Truth',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_pred,
        y=pred_concat,
        mode='lines',
        name='Prediction',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Blink Signals Comparison',
        xaxis_title='Time (s)',
        yaxis_title='Blink Intensity',
        hovermode='x unified',
        height=400
    )
    
    return fig


def _create_derivatives_plot(time_gt: np.ndarray,
                            time_pred: np.ndarray,
                            gt_signal: np.ndarray,
                            pred_signal: np.ndarray,
                            matches: Dict) -> go.Figure:
    """Create plot comparing z-score signals with matched peaks."""
    fig = go.Figure()
    
    # Calculate z-scores
    gt_zscore = zscore(gt_signal)
    pred_zscore = zscore(pred_signal)
    
    # Plot z-score signals
    fig.add_trace(go.Scatter(
        x=time_gt,
        y=gt_zscore,
        mode='lines',
        name='GT Z-Score',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_pred,
        y=pred_zscore,
        mode='lines',
        name='Pred Z-Score',
        line=dict(color='red', width=1)
    ))
    
    # Add matched peaks on z-score
    if len(matches['matched_gt']) > 0:
        fig.add_trace(go.Scatter(
            x=time_gt[matches['matched_gt']],
            y=gt_zscore[matches['matched_gt']],
            mode='markers',
            name='Matched GT Peaks',
            marker=dict(color='green', size=10, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_pred[matches['matched_pred']],
            y=pred_zscore[matches['matched_pred']],
            mode='markers',
            name='Matched Pred Peaks',
            marker=dict(color='lightgreen', size=8, symbol='x')
        ))
    
    # Add unmatched peaks on z-score
    if len(matches['unmatched_gt']) > 0:
        fig.add_trace(go.Scatter(
            x=time_gt[matches['unmatched_gt']],
            y=gt_zscore[matches['unmatched_gt']],
            mode='markers',
            name='Unmatched GT Peaks',
            marker=dict(color='orange', size=10, symbol='circle')
        ))
    
    if len(matches['unmatched_pred']) > 0:
        fig.add_trace(go.Scatter(
            x=time_pred[matches['unmatched_pred']],
            y=pred_zscore[matches['unmatched_pred']],
            mode='markers',
            name='Unmatched Pred Peaks',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title='Z-Score Comparison with Peak Matching',
        xaxis_title='Time (s)',
        yaxis_title='Z-Score',
        hovermode='x unified',
        height=400
    )
    
    return fig


def _create_histogram_plot(gt_diff: np.ndarray, pred_diff: np.ndarray) -> go.Figure:
    """Create histogram comparing derivative distributions."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=gt_diff,
        name='GT Derivative',
        opacity=0.6,
        nbinsx=100,
        marker_color='blue'
    ))
    
    fig.add_trace(go.Histogram(
        x=pred_diff,
        name='Pred Derivative',
        opacity=0.6,
        nbinsx=100,
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Derivative Distribution',
        xaxis_title='Derivative Value',
        yaxis_title='Count',
        barmode='overlay',
        height=400
    )
    
    return fig


def plot_blink_analysis(gt_concat: np.ndarray, 
                       pred_concat: np.ndarray, 
                       sample_rate: float = 30.0,
                       gt_diff: np.ndarray = None,
                       pred_diff: np.ndarray = None,
                       gt_th: float = 0.5,
                       model_th: float = 0.5,
                       max_offset: int = 10,
                       gt_prominence: float = 1.0,
                       pred_prominence: float = 1.0,
                       significant_gt_movenents: np.ndarray = None,
                       manual_gt_peaks: np.ndarray = None) -> Tuple[Dict[str, go.Figure], Dict]:
    """
    Generate comprehensive blink analysis plots.
    
    Parameters:
    -----------
    gt_concat : np.ndarray
        Ground truth blink signal
    pred_concat : np.ndarray
        Predicted blink signal
    sample_rate : float
        Sampling rate in Hz
    gt_diff : np.ndarray
        Pre-computed GT derivative (optional, unused)
    pred_diff : np.ndarray
        Pre-computed prediction derivative (optional, unused)
    gt_th : float
        Height threshold for GT z-score peaks (default 0.5)
    model_th : float
        Height threshold for prediction z-score peaks (default 0.5)
    max_offset : int
        Maximum time offset for matching peaks
    gt_prominence : float
        Prominence threshold for GT z-score peaks (default 1.0)
    pred_prominence : float
        Prominence threshold for prediction z-score peaks (default 1.0)
    """
    time_gt = np.arange(len(gt_concat)) / sample_rate
    time_pred = np.arange(len(pred_concat)) / sample_rate
    
    # Filter ground truth signal
    b = np.ones(3) / 3
    gt_filtered = filtfilt(b, 1, gt_concat)
    
    # Compute or use provided derivatives
    if gt_diff is not None and pred_diff is not None:
        # Use provided differential signals
        pass
    else:
        # Compute derivatives
        gt_diff = np.diff(gt_filtered)
        pred_diff = np.diff(pred_concat)
    
    # Match peaks using z-score detection
    matches = _match_peaks(
        gt_filtered, pred_concat,
        gt_diff, pred_diff, 
        max_offset=max_offset, 
        gt_th=gt_th, 
        model_th=model_th,
        gt_prominence=gt_prominence,
        pred_prominence=pred_prominence,
        manual_gt_peaks=manual_gt_peaks
    )
    
    # Generate plots
    plots = {
        'signals': _create_signals_plot(time_gt, time_pred, gt_filtered, pred_concat),
        'derivatives': _create_derivatives_plot(
            time_gt, time_pred, gt_filtered, pred_concat, matches
        ),
        'histogram': _create_histogram_plot(gt_diff, pred_diff)
    }
    
    return plots, matches

