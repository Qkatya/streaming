#!/usr/bin/env python3
"""
Quick script to check manual labels in the pickle file.
"""

import pandas as pd
from pathlib import Path

manual_labels_file = Path("unmatched_pred_peaks_labeled.pkl")

if manual_labels_file.exists():
    print(f"\n{'='*80}")
    print("MANUAL LABELS FILE FOUND")
    print(f"{'='*80}")
    
    df = pd.read_pickle(manual_labels_file)
    
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    if 'is_blink' in df.columns:
        labeled_count = df['is_blink'].notna().sum()
        unlabeled_count = df['is_blink'].isna().sum()
        
        if labeled_count > 0:
            blink_count = int((df['is_blink'] == True).sum())
            no_blink_count = int((df['is_blink'] == False).sum())
            dont_know_count = int((df['is_blink'] == 'unknown').sum())
            
            print(f"\n{'='*80}")
            print("LABEL STATISTICS")
            print(f"{'='*80}")
            print(f"Labeled: {labeled_count}")
            print(f"Unlabeled: {unlabeled_count}")
            print(f"  - Labeled as BLINK: {blink_count}")
            print(f"  - Labeled as NO BLINK: {no_blink_count}")
            print(f"  - Labeled as DON'T KNOW: {dont_know_count}")
            print(f"{'='*80}")
            
            print(f"\nFirst 10 labeled entries:")
            print(df[df['is_blink'].notna()][['global_frame', 'file_index', 'local_frame', 'timestamp_seconds', 'is_blink']].head(10).to_string(index=False))
        else:
            print("\nNo labels found yet (all values are None/NaN)")
    else:
        print("\nWarning: 'is_blink' column not found in the dataframe")
else:
    print(f"\n{'='*80}")
    print("MANUAL LABELS FILE NOT FOUND")
    print(f"{'='*80}")
    print(f"Looking for: {manual_labels_file.absolute()}")
    print("Please run the labeling GUI and save your labels first.")

