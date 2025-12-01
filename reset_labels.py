#!/usr/bin/env python3
"""
Reset all manual labels to None/unlabeled state.
"""

import pandas as pd
from pathlib import Path

# Files
input_file = Path("unmatched_pred_peaks_labeled.pkl")
output_file = Path("unmatched_pred_peaks_labeled.pkl")

if input_file.exists():
    print(f"\n{'='*80}")
    print("RESETTING MANUAL LABELS")
    print(f"{'='*80}")
    
    df = pd.read_pickle(input_file)
    
    # Count current labels
    if 'is_blink' in df.columns:
        labeled_count = df['is_blink'].notna().sum()
        print(f"Current labeled entries: {labeled_count}")
    
    # Reset all labels to None
    df['is_blink'] = None
    
    # Save
    df.to_pickle(output_file)
    
    print(f"All labels reset to None")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}\n")
else:
    print(f"\n{'='*80}")
    print("ERROR: File not found")
    print(f"{'='*80}")
    print(f"Looking for: {input_file.absolute()}")
    print(f"{'='*80}\n")

