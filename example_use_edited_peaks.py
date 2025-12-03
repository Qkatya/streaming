"""
Example script showing how to use edited GT peaks in your analysis.
This demonstrates loading the edited peaks and using them with your existing code.
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import zscore
from scipy.signal import find_peaks

def load_edited_peaks(final_peaks_file="final_gt_peaks.pkl"):
    """
    Load the final edited GT peaks.
    
    Returns:
    --------
    peaks : np.ndarray
        Array of frame indices for GT peaks
    """
    if not Path(final_peaks_file).exists():
        raise FileNotFoundError(
            f"Edited peaks file not found: {final_peaks_file}\n"
            "Run the peak editor first to create this file."
        )
    
    with open(final_peaks_file, 'rb') as f:
        peaks = pickle.load(f)
    
    print(f"Loaded {len(peaks)} edited GT peaks")
    return peaks

def calculate_metrics_with_edited_peaks(gt_signal, pred_signal, edited_gt_peaks, 
                                       sample_rate=30.0, max_offset=10):
    """
    Calculate blink detection metrics using edited GT peaks.
    
    Parameters:
    -----------
    gt_signal : np.ndarray
        Ground truth blink signal
    pred_signal : np.ndarray
        Predicted blink signal
    edited_gt_peaks : np.ndarray
        Edited GT peak locations (frame indices)
    sample_rate : float
        Sampling rate in Hz
    max_offset : int
        Maximum frame offset for matching peaks
    
    Returns:
    --------
    metrics : dict
        Dictionary with TPR, FPR, FNR, precision, recall, F1
    """
    # Detect prediction peaks
    pred_zscore = zscore(pred_signal)
    pred_peaks, _ = find_peaks(pred_zscore, height=0.5, prominence=1.0)
    
    # Match peaks
    matched_gt = []
    matched_pred = []
    unmatched_pred = list(pred_peaks)
    
    for gt_peak in edited_gt_peaks:
        distances = np.abs(pred_peaks - gt_peak)
        close_peaks = np.where(distances <= max_offset)[0]
        
        if len(close_peaks) > 0:
            closest_idx = close_peaks[np.argmin(distances[close_peaks])]
            closest_pred_peak = pred_peaks[closest_idx]
            
            matched_gt.append(gt_peak)
            matched_pred.append(closest_pred_peak)
            
            if closest_pred_peak in unmatched_pred:
                unmatched_pred.remove(closest_pred_peak)
    
    # Calculate metrics
    TP = len(matched_gt)
    FN = len(edited_gt_peaks) - TP
    FP = len(unmatched_pred)
    
    # Rates
    TPR = (TP / len(edited_gt_peaks)) * 100 if len(edited_gt_peaks) > 0 else 0
    FNR = (FN / len(edited_gt_peaks)) * 100 if len(edited_gt_peaks) > 0 else 0
    
    # False positives per minute
    duration_minutes = len(pred_signal) / sample_rate / 60
    FPR_per_min = FP / duration_minutes if duration_minutes > 0 else 0
    
    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TPR': TPR,
        'FNR': FNR,
        'FPR_per_min': FPR_per_min,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_gt_peaks': len(edited_gt_peaks),
        'total_pred_peaks': len(pred_peaks),
        'matched_peaks': TP
    }
    
    return metrics

def print_metrics(metrics):
    """Print metrics in a nice format."""
    print("\n" + "=" * 80)
    print("BLINK DETECTION METRICS (with Edited GT Peaks)")
    print("=" * 80)
    
    print(f"\nPeak Counts:")
    print(f"  Total GT peaks (edited): {metrics['total_gt_peaks']}")
    print(f"  Total Pred peaks: {metrics['total_pred_peaks']}")
    print(f"  Matched peaks: {metrics['matched_peaks']}")
    print(f"  True Positives (TP): {metrics['TP']}")
    print(f"  False Positives (FP): {metrics['FP']}")
    print(f"  False Negatives (FN): {metrics['FN']}")
    
    print(f"\nDetection Rates:")
    print(f"  True Positive Rate (TPR): {metrics['TPR']:.2f}%")
    print(f"  False Negative Rate (FNR): {metrics['FNR']:.2f}%")
    print(f"  False Positives per minute: {metrics['FPR_per_min']:.2f}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    print("=" * 80)

def compare_original_vs_edited(gt_signal, pred_signal, original_gt_peaks, 
                               edited_gt_peaks, sample_rate=30.0):
    """
    Compare metrics before and after editing.
    
    Parameters:
    -----------
    gt_signal : np.ndarray
        Ground truth blink signal
    pred_signal : np.ndarray
        Predicted blink signal
    original_gt_peaks : np.ndarray
        Original GT peaks (before editing)
    edited_gt_peaks : np.ndarray
        Edited GT peaks (after editing)
    sample_rate : float
        Sampling rate in Hz
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Edited GT Peaks")
    print("=" * 80)
    
    # Calculate metrics for original peaks
    print("\n[1] Calculating metrics with ORIGINAL GT peaks...")
    original_metrics = calculate_metrics_with_edited_peaks(
        gt_signal, pred_signal, original_gt_peaks, sample_rate
    )
    
    # Calculate metrics for edited peaks
    print("[2] Calculating metrics with EDITED GT peaks...")
    edited_metrics = calculate_metrics_with_edited_peaks(
        gt_signal, pred_signal, edited_gt_peaks, sample_rate
    )
    
    # Print comparison
    print("\n" + "-" * 80)
    print("COMPARISON TABLE")
    print("-" * 80)
    print(f"{'Metric':<30} {'Original':<20} {'Edited':<20} {'Change':<20}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('Total GT Peaks', 'total_gt_peaks', ''),
        ('Total Pred Peaks', 'total_pred_peaks', ''),
        ('True Positives', 'TP', ''),
        ('False Positives', 'FP', ''),
        ('False Negatives', 'FN', ''),
        ('TPR (%)', 'TPR', '%'),
        ('FNR (%)', 'FNR', '%'),
        ('FPR (per min)', 'FPR_per_min', ''),
        ('Precision', 'precision', ''),
        ('Recall', 'recall', ''),
        ('F1 Score', 'f1_score', ''),
    ]
    
    for metric_name, metric_key, suffix in metrics_to_compare:
        orig_val = original_metrics[metric_key]
        edit_val = edited_metrics[metric_key]
        
        if suffix == '%':
            change = edit_val - orig_val
            change_str = f"{change:+.2f}%"
            orig_str = f"{orig_val:.2f}{suffix}"
            edit_str = f"{edit_val:.2f}{suffix}"
        elif suffix == '':
            if isinstance(orig_val, int):
                change = edit_val - orig_val
                change_str = f"{change:+d}"
                orig_str = f"{orig_val}"
                edit_str = f"{edit_val}"
            else:
                change = edit_val - orig_val
                change_str = f"{change:+.4f}"
                orig_str = f"{orig_val:.4f}"
                edit_str = f"{edit_val:.4f}"
        
        print(f"{metric_name:<30} {orig_str:<20} {edit_str:<20} {change_str:<20}")
    
    print("-" * 80)
    
    # Summary
    print("\nSUMMARY:")
    peak_change = len(edited_gt_peaks) - len(original_gt_peaks)
    f1_change = edited_metrics['f1_score'] - original_metrics['f1_score']
    
    print(f"  GT peaks changed by: {peak_change:+d}")
    print(f"  F1 score changed by: {f1_change:+.4f}")
    
    if f1_change > 0:
        print(f"  ✓ Editing IMPROVED performance by {f1_change:.4f}")
    elif f1_change < 0:
        print(f"  ✗ Editing DECREASED performance by {abs(f1_change):.4f}")
    else:
        print(f"  = Editing had NO EFFECT on F1 score")
    
    print("=" * 80)

def example_usage():
    """Example of how to use edited peaks."""
    print("=" * 80)
    print("Example: Using Edited GT Peaks in Analysis")
    print("=" * 80)
    
    # Check if edited peaks exist
    if not Path("final_gt_peaks.pkl").exists():
        print("\n✗ No edited peaks found!")
        print("Please run the peak editor first:")
        print("  python launch_peak_editor.py")
        return
    
    # Load edited peaks
    print("\n[1] Loading edited GT peaks...")
    edited_peaks = load_edited_peaks()
    
    print("\n[2] To use these peaks in your analysis:")
    print("\nOption A - Direct usage:")
    print("""
    import pickle
    with open('final_gt_peaks.pkl', 'rb') as f:
        gt_peaks = pickle.load(f)
    
    # Use gt_peaks in your analysis
    # Example: Calculate blink rate
    blink_rate = len(gt_peaks) / (total_frames / sample_rate / 60)  # blinks per minute
    print(f"Blink rate: {blink_rate:.2f} blinks/min")
    """)
    
    print("\nOption B - Replace peaks in BlinkAnalyzer:")
    print("""
    from blink_analyzer import BlinkAnalyzer
    
    # Create analyzer
    analyzer = BlinkAnalyzer()
    
    # Instead of detecting peaks, use your edited peaks
    # Modify the analyze_blinks method to accept pre-detected peaks
    # Or manually create the matches dictionary with your edited peaks
    """)
    
    print("\nOption C - Calculate metrics (see calculate_metrics_with_edited_peaks):")
    print("""
    from example_use_edited_peaks import calculate_metrics_with_edited_peaks
    
    metrics = calculate_metrics_with_edited_peaks(
        gt_signal, pred_signal, edited_peaks
    )
    
    print(f"TPR: {metrics['TPR']:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    """)
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    example_usage()


