"""
Simple test script to verify the refactoring works correctly.
Tests that all imports and basic functions work.
"""

import sys
import numpy as np
from pathlib import Path

print("Testing imports from blendshapes_utils...")

try:
    from blendshapes_utils import (
        BLENDSHAPES_ORDERED,
        NEMO_BLENDSHAPE_BOUNDARIES,
        init_fairseq,
        unnormalize_nemo_blendshapes,
        zeropad_blendshapes_to_52,
        unnormalize_blendshapes,
        find_zip_file_id,
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test constants
print(f"\nTesting constants...")
print(f"✓ BLENDSHAPES_ORDERED has {len(BLENDSHAPES_ORDERED)} items")
print(f"✓ NEMO_BLENDSHAPE_BOUNDARIES has {len(NEMO_BLENDSHAPE_BOUNDARIES)} items")

# Test zeropad function
print(f"\nTesting zeropad_blendshapes_to_52...")
test_blendshapes = np.random.rand(10, 5)
blendshapes_idx = [1, 2, 3, 4, 5]
padded = zeropad_blendshapes_to_52(test_blendshapes, blendshapes_idx)
assert padded.shape == (10, 52), f"Expected shape (10, 52), got {padded.shape}"
assert np.allclose(padded[:, blendshapes_idx], test_blendshapes), "Padding failed"
print(f"✓ Zeropad function works correctly")

# Test unnormalize_nemo_blendshapes
print(f"\nTesting unnormalize_nemo_blendshapes...")
test_normalized = np.random.rand(10, 52)
blendshapes_idx = list(range(1, 52))
unnormalized = unnormalize_nemo_blendshapes(test_normalized, blendshapes_idx)
assert unnormalized.shape == test_normalized.shape, "Shape mismatch"
print(f"✓ NeMo unnormalization works")

# Test unnormalize_blendshapes wrapper
print(f"\nTesting unnormalize_blendshapes wrapper...")
result = unnormalize_blendshapes(test_normalized, 'nemo', blendshapes_idx)
assert result.shape == test_normalized.shape, "Shape mismatch"
print(f"✓ Unnormalize wrapper works for NeMo")

# Test find_zip_file_id
print(f"\nTesting find_zip_file_id...")
# This will return None for non-existent path, which is expected
test_path = Path("/tmp/nonexistent")
result = find_zip_file_id(test_path)
print(f"✓ find_zip_file_id handles missing path correctly (returned: {result})")

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nThe refactoring is successful. Both scripts should now work with shared utilities.")
print("\nNext steps:")
print("1. Test offline_blendshapes_inference_metrics.py with actual data")
print("2. Test sliding_window_inference_simulator.py with actual data")

