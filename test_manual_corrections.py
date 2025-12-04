#!/usr/bin/env python3
"""
Quick test to verify manual corrections are loaded and applied correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from blinking_split_analysis import load_manual_corrections_from_labels, apply_manual_corrections_to_peaks

def test_manual_corrections():
    print("="*80)
    print("TESTING MANUAL CORRECTIONS")
    print("="*80)
    
    # Load corrections
    corrections_dict = load_manual_corrections_from_labels("manual_blink_labels.csv")
    
    if not corrections_dict:
        print("\n✗ No corrections loaded!")
        return
    
    print(f"\n✓ Loaded corrections for {len(corrections_dict)} files")
    
    # Test applying corrections
    print("\n" + "="*80)
    print("TESTING CORRECTION APPLICATION")
    print("="*80)
    
    # Example: auto peaks at frames [10, 20, 30]
    auto_peaks = np.array([10, 20, 30])
    
    # Example corrections: add blink at 15, no_blink at 25, dont_know at 35
    test_corrections = [
        (15, 'blink'),      # Should add frame 15
        (25, 'no_blink'),   # Should do nothing
        (35, 'dont_know'),  # Should do nothing
    ]
    
    print(f"\nOriginal auto peaks: {auto_peaks}")
    print(f"Test corrections: {test_corrections}")
    
    corrected_peaks = apply_manual_corrections_to_peaks(auto_peaks, test_corrections)
    
    print(f"Corrected peaks: {corrected_peaks}")
    print(f"\nExpected: [10, 15, 20, 30] (added frame 15)")
    
    if list(corrected_peaks) == [10, 15, 20, 30]:
        print("✓ Correction application works correctly!")
    else:
        print("✗ Correction application failed!")
    
    # Show some real examples
    print("\n" + "="*80)
    print("REAL CORRECTION EXAMPLES")
    print("="*80)
    
    for i, (key, corrections) in enumerate(list(corrections_dict.items())[:5]):
        run_path, tar_id, side = key
        print(f"\n{i+1}. File: {run_path}")
        print(f"   TAR ID: {tar_id}, Side: {side}")
        print(f"   Corrections ({len(corrections)}):")
        for frame, label in corrections:
            action = {
                'blink': '→ ADD peak to GT',
                'no_blink': '→ Keep GT as is (pred FP)',
                'dont_know': '→ Keep GT as is (uncertain)'
            }[label]
            print(f"     Frame {frame} ({label}): {action}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    test_manual_corrections()


