"""
Test script for the peak editor app with synthetic data.
Use this to verify the app works before loading real data.
"""
import numpy as np
from peak_editor_app import create_app

def generate_synthetic_blink_signal(duration_seconds=300, sample_rate=30.0, num_blinks=20):
    """
    Generate synthetic blink signals for testing.
    
    Parameters:
    -----------
    duration_seconds : float
        Duration of signal in seconds
    sample_rate : float
        Sampling rate in Hz
    num_blinks : int
        Number of blinks to generate
    
    Returns:
    --------
    gt_signal : np.ndarray
        Synthetic ground truth signal
    pred_signal : np.ndarray
        Synthetic prediction signal (with some noise and missed blinks)
    """
    num_frames = int(duration_seconds * sample_rate)
    
    # Initialize signals with baseline noise
    gt_signal = np.random.normal(0.1, 0.02, num_frames)
    pred_signal = np.random.normal(0.1, 0.03, num_frames)
    
    # Generate random blink times
    blink_times = np.sort(np.random.choice(
        np.arange(100, num_frames - 100), 
        size=num_blinks, 
        replace=False
    ))
    
    print(f"Generating {num_blinks} synthetic blinks...")
    
    for i, blink_time in enumerate(blink_times):
        # Generate a blink waveform (Gaussian-like)
        blink_duration = int(sample_rate * 0.2)  # 200ms blink
        blink_start = blink_time - blink_duration // 2
        blink_end = blink_time + blink_duration // 2
        
        if blink_start < 0 or blink_end >= num_frames:
            continue
        
        # Create blink shape
        t = np.linspace(-2, 2, blink_duration)
        blink_shape = np.exp(-t**2)  # Gaussian
        
        # Add to GT signal
        gt_signal[blink_start:blink_end] += blink_shape * 0.8
        
        # Add to prediction signal with some variations
        if np.random.random() > 0.1:  # 90% of blinks are detected
            # Add some timing offset
            offset = np.random.randint(-3, 4)
            pred_start = max(0, blink_start + offset)
            pred_end = min(num_frames, blink_end + offset)
            pred_duration = pred_end - pred_start
            
            # Add with slightly different amplitude
            amplitude = 0.8 + np.random.normal(0, 0.1)
            if pred_duration == blink_duration:
                pred_signal[pred_start:pred_end] += blink_shape * amplitude
            else:
                # Adjust shape if offset caused length change
                t_pred = np.linspace(-2, 2, pred_duration)
                pred_shape = np.exp(-t_pred**2)
                pred_signal[pred_start:pred_end] += pred_shape * amplitude
    
    # Add some false positives to prediction
    num_false_positives = num_blinks // 5
    false_positive_times = np.random.choice(
        np.arange(100, num_frames - 100),
        size=num_false_positives,
        replace=False
    )
    
    print(f"Adding {num_false_positives} false positive blinks to predictions...")
    
    for fp_time in false_positive_times:
        blink_duration = int(sample_rate * 0.15)
        fp_start = fp_time - blink_duration // 2
        fp_end = fp_time + blink_duration // 2
        
        if fp_start < 0 or fp_end >= num_frames:
            continue
        
        t = np.linspace(-2, 2, blink_duration)
        fp_shape = np.exp(-t**2)
        pred_signal[fp_start:fp_end] += fp_shape * 0.6
    
    # Smooth signals slightly
    from scipy.signal import savgol_filter
    gt_signal = savgol_filter(gt_signal, 5, 2)
    pred_signal = savgol_filter(pred_signal, 5, 2)
    
    # Ensure non-negative
    gt_signal = np.maximum(gt_signal, 0)
    pred_signal = np.maximum(pred_signal, 0)
    
    print(f"\nGenerated synthetic signals:")
    print(f"  Duration: {duration_seconds} seconds")
    print(f"  Frames: {num_frames}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  GT blinks: {num_blinks}")
    print(f"  Pred false positives: {num_false_positives}")
    
    return gt_signal, pred_signal

def main():
    """Main entry point for testing."""
    print("=" * 80)
    print("Peak Editor App - Test with Synthetic Data")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    print("Generating synthetic blink signals...")
    gt_signal, pred_signal = generate_synthetic_blink_signal(
        duration_seconds=300,  # 5 minutes
        sample_rate=30.0,
        num_blinks=30
    )
    
    print("\n" + "=" * 80)
    print("Launching Peak Editor App...")
    print("=" * 80)
    print("\nThe app will open in your browser at: http://127.0.0.1:8050")
    print("\nThis is TEST MODE with synthetic data.")
    print("Use this to familiarize yourself with the interface before editing real data.")
    print("\nInstructions:")
    print("  - Use the Previous/Next buttons to navigate between windows")
    print("  - Click on peaks to remove them")
    print("  - Click elsewhere to add new peaks")
    print("  - Click 'Save Changes' to save (will create test pickle files)")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    # Create and run app
    app = create_app(gt_signal, pred_signal, sample_rate=30.0)
    app.run(debug=True, port=8050)

if __name__ == '__main__':
    main()


