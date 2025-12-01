"""
Visualization functions for blink analysis.
"""
import numpy as np
from typing import Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import filtfilt, find_peaks, savgol_filter


def _match_peaks(gt_signal: np.ndarray, 
                pred_signal: np.ndarray, 
                max_offset: int = 10,
                gt_th: float = 0.1,
                model_th: float = 0.06,
                gt_signal_raw: np.ndarray = None,
                pred_signal_raw: np.ndarray = None,
                prominence_gt: float = 1.0,
                prominence_pred: float = 0.8,
                distance: int = 5,
                width_min: int = 2,
                width_max: int = 20,
                use_derivative_validation: bool = True) -> Dict:
    """Match peaks between ground truth and prediction signals.
    
    Args:
        gt_signal: Ground truth signal (z-scored) for peak detection
        pred_signal: Prediction signal (z-scored) for peak detection
        max_offset: Maximum frame offset for matching peaks
        gt_th: Height threshold for GT peaks
        model_th: Height threshold for prediction peaks
        gt_signal_raw: Optional raw GT signal for derivative validation
        pred_signal_raw: Optional raw signal for derivative validation
        prominence_gt: Prominence threshold for GT peaks (how much peak stands out)
        prominence_pred: Prominence threshold for pred peaks
        distance: Minimum frames between peaks
        width_min: Minimum peak width in frames
        width_max: Maximum peak width in frames
        use_derivative_validation: Whether to validate peaks with derivative pattern
    """
    # Compute derivatives for peak validation if raw signals provided
    if gt_signal_raw is not None:
        gt_derivative = np.diff(gt_signal_raw)
    else:
        gt_derivative = np.diff(gt_signal)
    
    if pred_signal_raw is not None:
        pred_derivative = np.diff(pred_signal_raw)
    else:
        pred_derivative = np.diff(pred_signal)
    
    # Apply Savgol filter for smoothing before peak detection
    # window_length must be odd and less than signal length
    window_length = min(9, len(gt_signal) if len(gt_signal) % 2 == 1 else len(gt_signal) - 1)
    if window_length >= 3:  # Need at least 3 points for savgol
        gt_signal_smooth = savgol_filter(gt_signal, window_length, 2, mode='interp')
    else:
        gt_signal_smooth = gt_signal
    
    window_length_pred = min(9, len(pred_signal) if len(pred_signal) % 2 == 1 else len(pred_signal) - 1)
    if window_length_pred >= 3:
        pred_signal_smooth = savgol_filter(pred_signal, window_length_pred, 2, mode='interp')
    else:
        pred_signal_smooth = pred_signal
    
    # Find peaks with configurable filtering
    # - height: minimum peak height (absolute z-score threshold) - can be None to disable
    # - prominence: how much the peak stands out from surrounding baseline (RELATIVE to local DC - this handles moving baseline!)
    # - distance: minimum frames between peaks (prevents clustering)
    # - width: peak width constraints (filters noise) - max width should be less than 10
    
    # For GT: use height threshold only if gt_th is not None and > 0
    gt_peaks, gt_properties = find_peaks(
        gt_signal_smooth, 
        height=gt_th if gt_th is not None and gt_th > 0 else None,
        prominence=prominence_gt if prominence_gt is not None else None,
        distance=distance if distance is not None else None,
        width=(width_min, min(width_max, 10)) if (width_min is not None and width_max is not None) else None
    )
    
    # For pred: use height threshold only if model_th is not None and > 0
    pred_peaks, pred_properties = find_peaks(
        pred_signal_smooth, 
        height=model_th if model_th is not None and model_th > 0 else None,
        prominence=prominence_pred if prominence_pred is not None else None,
        distance=distance if distance is not None else None,
        width=(width_min, min(width_max, 10)) if (width_min is not None and width_max is not None) else None
    )
    
    # Optional derivative validation
    if use_derivative_validation and gt_signal_raw is not None and pred_signal_raw is not None:
        # Additional validation: check derivative sign change (peak should have + before, - after)
        def validate_peak_with_derivative(peak_idx, derivative, signal_len):
            """Validate that peak has proper derivative pattern (rise then fall)."""
            if peak_idx == 0 or peak_idx >= signal_len - 1:
                return False
            
            # Check a window around the peak
            window_before = max(0, peak_idx - 3)
            window_after = min(signal_len - 1, peak_idx + 2)
            
            # Derivative before peak should be mostly positive (rising)
            # Derivative at/after peak should be mostly negative (falling)
            if peak_idx < len(derivative):
                before_positive = np.sum(derivative[window_before:peak_idx] > 0) > 0
                after_negative = np.sum(derivative[peak_idx:window_after] < 0) > 0
                return before_positive and after_negative
            return True
        
        # Filter peaks based on derivative validation
        gt_peaks_valid = [p for p in gt_peaks if validate_peak_with_derivative(p, gt_derivative, len(gt_signal))]
        pred_peaks_valid = [p for p in pred_peaks if validate_peak_with_derivative(p, pred_derivative, len(pred_signal))]
        
        gt_peaks = np.array(gt_peaks_valid)
        pred_peaks = np.array(pred_peaks_valid)
    
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
        'matched_gt': np.array(matched_gt),
        'matched_pred': np.array(matched_pred),
        'unmatched_gt': np.array(unmatched_gt),
        'unmatched_pred': np.array(unmatched_pred),
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
        xaxis_title='Frame Index',
        yaxis_title='Blink Intensity',
        hovermode='x unified',
        height=400
    )
    
    return fig


def _create_derivatives_plot(time_gt_diff: np.ndarray,
                            time_pred_diff: np.ndarray,
                            gt_diff: np.ndarray,
                            pred_diff: np.ndarray,
                            matches: Dict,
                            gt_concat: np.ndarray = None,
                            pred_concat: np.ndarray = None,
                            time_gt: np.ndarray = None,
                            time_pred: np.ndarray = None,
                            gt_zscore: np.ndarray = None,
                            pred_zscore: np.ndarray = None) -> go.Figure:
    """Create plot comparing z-scored signals with matched peaks, and optionally raw signals."""
    
    # Determine if we should create subplots
    has_raw_signals = (gt_concat is not None and pred_concat is not None and 
                       time_gt is not None and time_pred is not None)
    
    # Check if we have z-scored signals for peak plotting
    has_zscore = (gt_zscore is not None and pred_zscore is not None)
    
    if has_raw_signals:
        # Create subplots: z-scored signals on top, raw signals on bottom
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Z-scored Signals with Peak Matching', 'Raw Signals'),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )
        row_zscore = 1
        row_signals = 2
    else:
        # Single plot (backward compatibility)
        fig = go.Figure()
        row_zscore = None
        row_signals = None
    
    # Plot z-scored signals if available, otherwise plot derivatives
    if has_zscore:
        # Plot z-scored signals
        fig.add_trace(go.Scatter(
            x=time_gt,
            y=gt_zscore,
            mode='lines',
            name='GT Z-score',
            line=dict(color='blue', width=1),
            legendgroup='zscore'
        ), row=row_zscore, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_pred,
            y=pred_zscore,
            mode='lines',
            name='Pred Z-score',
            line=dict(color='red', width=1),
            legendgroup='zscore'
        ), row=row_zscore, col=1)
        
        # Add matched peaks on z-scored signals
        if len(matches['matched_gt']) > 0:
            fig.add_trace(go.Scatter(
                x=time_gt[matches['matched_gt']],
                y=gt_zscore[matches['matched_gt']],
                mode='markers',
                name='Matched GT Peaks',
                marker=dict(color='green', size=10, symbol='circle'),
                legendgroup='zscore'
            ), row=row_zscore, col=1)
            
            fig.add_trace(go.Scatter(
                x=time_pred[matches['matched_pred']],
                y=pred_zscore[matches['matched_pred']],
                mode='markers',
                name='Matched Pred Peaks',
                marker=dict(color='lightgreen', size=8, symbol='x'),
                legendgroup='zscore'
            ), row=row_zscore, col=1)
        
        # Add unmatched peaks on z-scored signals
        if len(matches['unmatched_gt']) > 0:
            fig.add_trace(go.Scatter(
                x=time_gt[matches['unmatched_gt']],
                y=gt_zscore[matches['unmatched_gt']],
                mode='markers',
                name='Unmatched GT Peaks',
                marker=dict(color='orange', size=10, symbol='circle'),
                legendgroup='zscore'
            ), row=row_zscore, col=1)
        
        if len(matches['unmatched_pred']) > 0:
            fig.add_trace(go.Scatter(
                x=time_pred[matches['unmatched_pred']],
                y=pred_zscore[matches['unmatched_pred']],
                mode='markers',
                name='Unmatched Pred Peaks',
                marker=dict(color='red', size=8, symbol='x'),
                legendgroup='zscore'
            ), row=row_zscore, col=1)
    else:
        # Fallback to derivatives (backward compatibility)
        fig.add_trace(go.Scatter(
            x=time_gt_diff,
            y=gt_diff,
            mode='lines',
            name='GT Derivative',
            line=dict(color='blue', width=1),
            legendgroup='derivatives'
        ), row=row_zscore, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_pred_diff,
            y=pred_diff,
            mode='lines',
            name='Pred Derivative',
            line=dict(color='red', width=1),
            legendgroup='derivatives'
        ), row=row_zscore, col=1)
    
    # Add raw signals subplot if available
    if has_raw_signals:
        # Trim the last frame from raw signals to match derivative length
        # (derivatives have one less frame due to np.diff)
        gt_concat_trimmed = gt_concat[:-1]
        pred_concat_trimmed = pred_concat[:-1]
        time_gt_trimmed = time_gt[:-1]
        time_pred_trimmed = time_pred[:-1]
        
        # Remove frames where pred is zero
        non_zero_mask = pred_concat_trimmed != 0
        gt_concat_filtered = gt_concat_trimmed[non_zero_mask]
        pred_concat_filtered = pred_concat_trimmed[non_zero_mask]
        time_gt_filtered = time_gt_trimmed[non_zero_mask]
        time_pred_filtered = time_pred_trimmed[non_zero_mask]
        
        # Plot raw GT and Pred signals (with zeros removed)
        fig.add_trace(go.Scatter(
            x=time_gt_filtered,
            y=gt_concat_filtered,
            mode='lines',
            name='GT Signal',
            line=dict(color='blue', width=1),
            legendgroup='signals',
            showlegend=True
        ), row=row_signals, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_pred_filtered,
            y=pred_concat_filtered,
            mode='lines',
            name='Pred Signal',
            line=dict(color='red', width=1),
            legendgroup='signals',
            showlegend=True
        ), row=row_signals, col=1)
        
        # Update layout for subplots with linked x-axes
        fig.update_xaxes(title_text='Frame Index', row=1, col=1)
        fig.update_yaxes(title_text='Z-score' if has_zscore else 'Derivative', row=1, col=1)
        fig.update_xaxes(title_text='Frame Index', row=2, col=1, matches='x')
        fig.update_yaxes(title_text='Signal Value', row=2, col=1)
        
        fig.update_layout(
            hovermode='x unified',
            height=800
        )
    else:
        # Single plot layout (backward compatibility)
        fig.update_layout(
            title='Z-scored Signal Comparison with Peak Matching' if has_zscore else 'Derivative Comparison with Peak Matching',
            xaxis_title='Frame Index',
            yaxis_title='Z-score' if has_zscore else 'Derivative',
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
                       max_offset: int = 10,
                       significant_gt_movenents: np.ndarray = None,
                       prominence_gt: float = 1.0,
                       prominence_pred: float = 0.8,
                       distance: int = 5,
                       width_min: int = 2,
                       width_max: int = 20,
                       use_derivative_validation: bool = True) -> Tuple[Dict[str, go.Figure], Dict]:
    """Generate comprehensive blink analysis plots."""
    # Use frame indices instead of time
    frames_gt = np.arange(len(gt_concat))
    frames_pred = np.arange(len(pred_concat))
    
    # Filter ground truth signal
    b = np.ones(3) / 3
    gt_filtered = filtfilt(b, 1, gt_concat)
    
    # Compute z-scores of the raw signals for peak detection
    from scipy.stats import zscore
    gt_zscore = zscore(gt_filtered)
    pred_zscore = zscore(pred_concat)
    
    # Find peaks on z-scored signals with configurable filtering
    matches = _match_peaks(
        gt_zscore, pred_zscore, max_offset, 
        gt_th=gt_th, model_th=model_th,
        gt_signal_raw=gt_filtered, pred_signal_raw=pred_concat,
        prominence_gt=prominence_gt, prominence_pred=prominence_pred,
        distance=distance, width_min=width_min, width_max=width_max,
        use_derivative_validation=use_derivative_validation
    )
    
    # Still compute derivatives for the derivative plot
    if gt_diff is not None and pred_diff is not None:
        # Use provided differential signals
        frames_gt_diff = frames_gt[:-1]
    else:
        # Compute derivatives
        gt_diff = np.diff(gt_filtered)
        pred_diff = np.diff(pred_concat)
        frames_gt_diff = frames_gt[:-1]
    
    # Generate plots
    plots = {
        'signals': _create_signals_plot(frames_gt, frames_pred, gt_filtered, pred_concat),
        'derivatives': _create_derivatives_plot(
            frames_gt_diff, frames_gt_diff, gt_diff, pred_diff, matches,
            gt_concat=gt_filtered, pred_concat=pred_concat,
            time_gt=frames_gt, time_pred=frames_pred,
            gt_zscore=gt_zscore, pred_zscore=pred_zscore
        ),
        'histogram': _create_histogram_plot(gt_diff, pred_diff)
    }
    
    return plots, matches

