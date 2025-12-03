"""
Launch peak editor with concatenated GT and prediction data from analysis.
This loads the already processed concatenated signals and detected peaks.
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from peak_editor_app import create_app

def load_concatenated_analysis_data(analysis_results_file="blink_analysis_results.pkl"):
    """
    Load concatenated GT and prediction data from saved analysis results.
    
    Parameters:
    -----------
    analysis_results_file : str
        Path to saved analysis results pickle file
    
    Returns:
    --------
    gt_concat : np.ndarray
        Concatenated ground truth blink signal
    pred_concat : np.ndarray
        Concatenated predicted blink signal
    metadata : dict
        Additional metadata
    """
    if not Path(analysis_results_file).exists():
        raise FileNotFoundError(
            f"Analysis results file not found: {analysis_results_file}\n"
            "Please run your analysis first to generate concatenated data."
        )
    
    with open(analysis_results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"✓ Loaded analysis results from: {analysis_results_file}")
    
    # Extract concatenated signals
    gt_concat = results['gt_concat']
    pred_concat = results['pred_concat']
    
    print(f"  GT signal length: {len(gt_concat)} frames ({len(gt_concat)/30:.1f} seconds)")
    print(f"  Pred signal length: {len(pred_concat)} frames ({len(pred_concat)/30:.1f} seconds)")
    
    metadata = {
        'num_frames': len(gt_concat),
        'duration_seconds': len(gt_concat) / 30.0,
        'source_file': analysis_results_file
    }
    
    if 'matches' in results:
        metadata['original_matches'] = results['matches']
        print(f"  Original GT peaks: {len(results['matches'].get('gt_peaks', []))}")
        print(f"  Original Pred peaks: {len(results['matches'].get('pred_peaks', []))}")
    
    return gt_concat, pred_concat, metadata

def save_concatenated_data_for_editing(gt_concat, pred_concat, output_file="concatenated_signals.pkl"):
    """
    Save concatenated signals to a file for later loading.
    
    Parameters:
    -----------
    gt_concat : np.ndarray
        Concatenated GT signal
    pred_concat : np.ndarray
        Concatenated prediction signal
    output_file : str
        Output file path
    """
    data = {
        'gt_concat': gt_concat,
        'pred_concat': pred_concat,
        'num_frames': len(gt_concat),
        'duration_seconds': len(gt_concat) / 30.0
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved concatenated signals to: {output_file}")

def load_from_saved_concatenated(input_file="concatenated_signals.pkl"):
    """
    Load previously saved concatenated signals.
    
    Parameters:
    -----------
    input_file : str
        Input file path
    
    Returns:
    --------
    gt_concat : np.ndarray
        Concatenated GT signal
    pred_concat : np.ndarray
        Concatenated prediction signal
    metadata : dict
        Metadata
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded concatenated signals from: {input_file}")
    print(f"  GT signal: {data['num_frames']} frames ({data['duration_seconds']:.1f} seconds)")
    
    metadata = {
        'num_frames': data['num_frames'],
        'duration_seconds': data['duration_seconds'],
        'source_file': input_file
    }
    
    return data['gt_concat'], data['pred_concat'], metadata

def main():
    """Main entry point."""
    print("=" * 80)
    print("Ground Truth Peak Editor - Load Concatenated Analysis Data")
    print("=" * 80)
    print()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Loading from specified file: {input_file}")
    else:
        # Try default files
        input_file = None
        
        # Check for common file names
        possible_files = [
            "concatenated_signals.pkl",
            "blink_analysis_results.pkl",
            "analysis_results.pkl"
        ]
        
        for fname in possible_files:
            if Path(fname).exists():
                input_file = fname
                print(f"Found data file: {input_file}")
                break
        
        if input_file is None:
            print("No concatenated data file found!")
            print("\nOptions:")
            print("1. Provide file path as argument:")
            print("   python launch_peak_editor_from_analysis.py your_data.pkl")
            print("\n2. Or save your concatenated data using this format:")
            print("""
import pickle
data = {
    'gt_concat': gt_concat,      # Your concatenated GT signal
    'pred_concat': pred_concat,  # Your concatenated prediction signal
    'num_frames': len(gt_concat),
    'duration_seconds': len(gt_concat) / 30.0
}
with open('concatenated_signals.pkl', 'wb') as f:
    pickle.dump(data, f)
""")
            print("\n3. Or manually provide arrays in Python:")
            print("""
from peak_editor_app import create_app
import numpy as np

# Your concatenated arrays
gt_concat = ...  # Your GT signal
pred_concat = ...  # Your prediction signal

# Launch editor
app = create_app(gt_concat, pred_concat, sample_rate=30.0)
app.run_server(debug=True, port=8050)
""")
            sys.exit(1)
    
    try:
        # Try to load the data
        if 'gt_concat' in str(input_file) or 'concatenated' in str(input_file):
            gt_signal, pred_signal, metadata = load_from_saved_concatenated(input_file)
        else:
            gt_signal, pred_signal, metadata = load_concatenated_analysis_data(input_file)
        
        # Print metadata
        print("\n" + "=" * 80)
        print("Data Loaded Successfully")
        print("=" * 80)
        for key, value in metadata.items():
            if key != 'original_matches':
                print(f"  {key}: {value}")
        
        # Create and launch app
        print("\n" + "=" * 80)
        print("Launching Peak Editor App...")
        print("=" * 80)
        print("\nThe app will open in your browser at: http://127.0.0.1:8050")
        print("\nYou are editing the CONCATENATED GT and prediction signals.")
        print("This includes all examples combined together.")
        print("\nInstructions:")
        print("  - Use Previous/Next buttons to navigate 50-second windows")
        print("  - Click on GT peaks (green/orange circles) to REMOVE them")
        print("  - Click elsewhere to ADD new GT peaks")
        print("  - Click 'Save Changes' to save modifications")
        print("\nOutput files:")
        print("  - removed_peaks.pkl (peaks you removed)")
        print("  - added_peaks.pkl (peaks you added)")
        print("  - final_gt_peaks.pkl (final GT peaks - USE THIS!)")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 80)
        print()
        
        # Create and run app
        app = create_app(gt_signal, pred_signal, sample_rate=30.0)
        app.run(debug=True, host='0.0.0.0', port=8050)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

