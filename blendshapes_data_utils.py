"""
Lightweight data utilities for blendshapes.

This module contains data loading functions and constants WITHOUT heavy ML dependencies.
Use this for scripts that only need to load/process data, not run inference.
"""

import numpy as np
from pathlib import Path
from typing import Optional

# ============================================================================
# CONSTANTS
# ============================================================================

BLENDSHAPES_ORDERED = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 
    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 
    'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 
    'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 
    'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 
    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 
    'noseSneerLeft', 'noseSneerRight'
]

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ground_truth_blendshapes(
    run_path: Path, 
    downsample: bool = True
) -> np.ndarray:
    """
    Load ground truth blendshapes from landmarks_and_blendshapes.npz.
    
    Args:
        run_path: Full path to run directory
        downsample: Whether to downsample from 200fps (skip every 6th frame)
    
    Returns:
        Ground truth blendshapes array
    """
    gt_file = run_path / "landmarks_and_blendshapes.npz"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    gt_blendshapes = np.load(gt_file)['blendshapes']
    
    if downsample:
        # Downsample from 200fps to match features (skip every 6th frame)
        mask = np.ones(gt_blendshapes.shape[0], dtype=bool)
        mask[5::6] = False
        gt_blendshapes = gt_blendshapes[mask]
    
    return gt_blendshapes

def find_zip_file_id(run_path: Path) -> Optional[str]:
    """
    Find the ID from XXXX.right.zip file in the run_path.
    
    Args:
        run_path: Path to the run directory
        
    Returns:
        The zip file ID (without .right.zip extension) or None if not found
    """
    zip_files = list(run_path.glob("*.right.zip"))
    if not zip_files:
        return None
    
    # Get the first .right.zip file and extract the ID
    zip_file = zip_files[0]
    zip_id = zip_file.stem.replace('.right', '')
    
    return zip_id

