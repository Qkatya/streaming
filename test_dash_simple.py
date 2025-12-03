"""
Simple test to check if Dash works and identify the error.
"""
import sys
import traceback

print("Step 1: Testing imports...")
try:
    from dash import Dash, dcc, html
    print("✓ Dash imports OK")
except Exception as e:
    print(f"✗ Dash import failed: {e}")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    print("✓ Plotly imports OK")
except Exception as e:
    print(f"✗ Plotly import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    from scipy.stats import zscore
    from scipy.signal import find_peaks, filtfilt
    print("✓ NumPy/SciPy imports OK")
except Exception as e:
    print(f"✗ NumPy/SciPy import failed: {e}")
    sys.exit(1)

print("\nStep 2: Testing data loading...")
try:
    import pickle
    with open('concatenated_signals_20251203_151149.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Data loaded: {data['num_frames']} frames")
    gt_concat = data['gt_concat']
    pred_concat = data['pred_concat']
    print(f"  GT shape: {gt_concat.shape}")
    print(f"  Pred shape: {pred_concat.shape}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Testing peak editor initialization...")
try:
    from peak_editor_app import PeakEditor
    print("  Creating PeakEditor...")
    editor = PeakEditor(gt_concat, pred_concat, sample_rate=30.0)
    print(f"✓ PeakEditor created successfully")
    print(f"  Original GT peaks: {len(editor.original_gt_peaks)}")
    print(f"  Number of windows: {editor.get_num_windows()}")
except Exception as e:
    print(f"✗ PeakEditor failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: Testing window data...")
try:
    window_data = editor.get_window_data(0)
    print(f"✓ Window data retrieved")
    print(f"  Window frames: {window_data['start_frame']} - {window_data['end_frame']}")
    print(f"  GT peaks in window: {len(window_data['gt_peaks'])}")
except Exception as e:
    print(f"✗ Window data failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing Dash app creation...")
try:
    from peak_editor_app import create_app
    print("  Creating Dash app...")
    app = create_app(gt_concat, pred_concat, sample_rate=30.0)
    print(f"✓ Dash app created successfully")
except Exception as e:
    print(f"✗ Dash app creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Starting Dash server...")
print("If this hangs or crashes, there's an issue with the server startup.")
try:
    app.run(debug=True, host='0.0.0.0', port=8050)
except Exception as e:
    print(f"✗ Server failed: {e}")
    traceback.print_exc()
    sys.exit(1)

