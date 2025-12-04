#!/usr/bin/env python3
"""
Quick script to display the manual blink labels from the dashboard.
"""

import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("MANUAL BLINK LABELS FROM DASHBOARD")
print("="*80)

# Load CSV
csv_file = "manual_blink_labels.csv"
if Path(csv_file).exists():
    df = pd.read_csv(csv_file)
    
    print(f"\n📊 Total labels: {len(df)}")
    print(f"\n📋 Label distribution:")
    print(df['label'].value_counts())
    
    print(f"\n📝 Sample labels:")
    print(df[['run_path', 'tar_id', 'side', 'peak_frame_25fps', 'label']].head(20))
    
    print(f"\n✅ Blinks (first 10):")
    blinks = df[df['label'] == 'blink']
    print(blinks[['run_path', 'tar_id', 'side', 'peak_frame_25fps']].head(10))
    
    print(f"\n❓ Don't know (first 10):")
    dont_know = df[df['label'] == 'dont_know']
    print(dont_know[['run_path', 'tar_id', 'side', 'peak_frame_25fps']].head(10))
    
    print(f"\n❌ No blink (first 10):")
    no_blink = df[df['label'] == 'no_blink']
    print(no_blink[['run_path', 'tar_id', 'side', 'peak_frame_25fps']].head(10))
    
else:
    print(f"❌ File not found: {csv_file}")

print("\n" + "="*80)
print("MANUAL GT PEAKS PICKLE")
print("="*80)

# Load pickle
pkl_file = "manual_gt_peaks_from_labels.pkl"
if Path(pkl_file).exists():
    with open(pkl_file, 'rb') as f:
        manual_peaks = pickle.load(f)
    
    print(f"\n📦 Type: {type(manual_peaks)}")
    print(f"📊 Number of files: {len(manual_peaks)}")
    
    # Show first few entries
    print(f"\n📝 Sample entries:")
    for i, (key, peaks) in enumerate(list(manual_peaks.items())[:5]):
        run_path, tar_id, side = key
        print(f"\n  File {i+1}:")
        print(f"    Run: {run_path.split('/')[-1]}")
        print(f"    Tar ID: {tar_id[:8]}...")
        print(f"    Side: {side}")
        print(f"    Peaks: {peaks[:10] if len(peaks) > 10 else peaks}")
        print(f"    Total peaks: {len(peaks)}")
    
    # Statistics
    total_peaks = sum(len(peaks) for peaks in manual_peaks.values())
    avg_peaks = total_peaks / len(manual_peaks) if manual_peaks else 0
    
    print(f"\n📈 Statistics:")
    print(f"  Total peaks across all files: {total_peaks}")
    print(f"  Average peaks per file: {avg_peaks:.1f}")
    
else:
    print(f"❌ File not found: {pkl_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
The manual_blink_labels.csv file contains your manual tags from the dashboard:
- 144 peaks labeled as 'blink' (should be added to GT)
- 36 peaks labeled as 'dont_know' (should be removed from analysis)
- 35 peaks labeled as 'no_blink' (no change to GT)

The manual_gt_peaks_from_labels.pkl file contains the adjusted GT peaks per file.

To visualize the differences:
1. Re-run: python blinking_split_analysis.py
2. Update CONCATENATED_SIGNALS_FILE in plot_manual_corrections_comparison.py
3. Run: python plot_manual_corrections_comparison.py
""")


