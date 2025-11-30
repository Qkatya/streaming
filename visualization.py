"""
Visualization functions for blink analysis.
"""
import numpy as np
from typing import Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import filtfilt, find_peaks


def _match_peaks(gt_diff: np.ndarray, 
                pred_diff: np.ndarray, 
                max_offset: int = 10,
                gt_th: float = 0.1,
                model_th: float = 0.06) -> Dict:
    """Match peaks between ground truth and prediction derivatives."""
    # Find peaks in both signals
    gt_peaks, _ = find_peaks(gt_diff, height=gt_th)
    pred_peaks, _ = find_peaks(pred_diff, height=model_th)
    
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


def _match_peaks_from_diff(gt_diff: np.ndarray, 
                          pred_diff: np.ndarray, 
                          max_offset: int = 10) -> Dict:
    """Match peaks from pre-computed differential signals."""
    # Find peaks in both signals
    gt_peaks, _ = find_peaks(gt_diff)
    pred_peaks, _ = find_peaks(pred_diff)
    
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


def _create_derivatives_plot(time_gt_diff: np.ndarray,
                            time_pred_diff: np.ndarray,
                            gt_diff: np.ndarray,
                            pred_diff: np.ndarray,
                            matches: Dict) -> go.Figure:
    """Create plot comparing derivatives with matched peaks."""
    fig = go.Figure()
    
    # Plot derivatives
    fig.add_trace(go.Scatter(
        x=time_gt_diff,
        y=gt_diff,
        mode='lines',
        name='GT Derivative',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_pred_diff,
        y=pred_diff,
        mode='lines',
        name='Pred Derivative',
        line=dict(color='red', width=1)
    ))
    
    # Add matched peaks
    if len(matches['matched_gt']) > 0:
        fig.add_trace(go.Scatter(
            x=time_gt_diff[matches['matched_gt']],
            y=gt_diff[matches['matched_gt']],
            mode='markers',
            name='Matched GT Peaks',
            marker=dict(color='green', size=10, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_pred_diff[matches['matched_pred']],
            y=pred_diff[matches['matched_pred']],
            mode='markers',
            name='Matched Pred Peaks',
            marker=dict(color='lightgreen', size=8, symbol='x')
        ))
    
    # Add unmatched peaks
    if len(matches['unmatched_gt']) > 0:
        fig.add_trace(go.Scatter(
            x=time_gt_diff[matches['unmatched_gt']],
            y=gt_diff[matches['unmatched_gt']],
            mode='markers',
            name='Unmatched GT Peaks',
            marker=dict(color='orange', size=10, symbol='circle')
        ))
    
    if len(matches['unmatched_pred']) > 0:
        fig.add_trace(go.Scatter(
            x=time_pred_diff[matches['unmatched_pred']],
            y=pred_diff[matches['unmatched_pred']],
            mode='markers',
            name='Unmatched Pred Peaks',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title='Derivative Comparison with Peak Matching',
        xaxis_title='Time (s)',
        yaxis_title='Derivative',
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
                       gt_th: float = 0.1,
                       model_th: float = 0.06,
                       max_offset: int = 10) -> Tuple[Dict[str, go.Figure], Dict]:
    """Generate comprehensive blink analysis plots."""
    time_gt = np.arange(len(gt_concat)) / sample_rate
    time_pred = np.arange(len(pred_concat)) / sample_rate
    
    # Filter ground truth signal
    b = np.ones(3) / 3
    gt_filtered = filtfilt(b, 1, gt_concat)
    
    
        
    if gt_diff is not None and pred_diff is not None:
        # Use provided differential signals
        time_gt_diff = time_gt[:-1]
        matches = _match_peaks(gt_diff, pred_diff, max_offset, gt_th=gt_th, model_th=model_th)
        # matches = _match_peaks_from_diff(gt_diff, pred_diff, max_offset)
    else:
        # Compute derivatives
        gt_diff = np.diff(gt_filtered)
        pred_diff = np.diff(pred_concat)
        time_gt_diff = time_gt[:-1]
        matches = _match_peaks(gt_diff, pred_diff, max_offset, gt_th=gt_th, model_th=model_th)
    
    # Generate plots
    plots = {
        'signals': _create_signals_plot(time_gt, time_pred, gt_filtered, pred_concat),
        'derivatives': _create_derivatives_plot(
            time_gt_diff, time_gt_diff, gt_diff, pred_diff, matches
        ),
        'histogram': _create_histogram_plot(gt_diff, pred_diff)
    }
    
    return plots, matches

