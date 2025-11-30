"""
Shared utilities for blendshapes inference and analysis.

This module contains common functions used across multiple scripts:
- Model loading (Fairseq and NeMo)
- Inference functions
- Blendshapes normalization/unnormalization
- Data loading utilities
- Constants and configurations
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# Fairseq setup
FAIRSEQ_PATH = "/home/katya.ivantsiv/q-fairseq-train_bs_loud-blendshapes_loud-8878cd07-20250718_212401"
sys.path.insert(0, FAIRSEQ_PATH)
sys.path.insert(0, f"{FAIRSEQ_PATH}/examples/data2vec")
from fairseq import checkpoint_utils, utils

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

# NeMo model min/max boundaries for unnormalization
NEMO_BLENDSHAPE_BOUNDARIES = [
    (1.2090332290881634e-08, 3.749652239548596e-06), (2.445738971346145e-07, 0.2749528124928477), 
    (1.978288395321215e-07, 0.2759505271911622), (1.0057362942461623e-06, 0.7707802176475524), 
    (7.059926429064944e-05, 0.7855742245912554), (4.5511307689594105e-05, 0.6722787052392961), 
    (7.704340418968059e-07, 0.00024206499801948667), (4.007033815867089e-09, 1.7510708346435425e-06), 
    (5.4851407860212475e-09, 7.810492093085486e-07), (0.0013135804329067469, 0.6775347143411636), 
    (0.0006478337454609573, 0.6262029528617858), (3.462312452029437e-05, 0.8312738806009295), 
    (0.000131698208861053, 0.826391476392746), (2.216839675384108e-05, 0.2110402494668961), 
    (2.0124221919104457e-05, 0.3471406474709512), (8.35571256629919e-08, 0.350243017077446), 
    (1.0900915185629856e-05, 0.20838836356997492), (8.824750693747774e-06, 0.09804979339241984), 
    (1.4880988601362333e-06, 0.0945791814476252), (0.007547934073954821, 0.5727719873189926), 
    (0.0041250200010836124, 0.5164364755153656), (0.00012323759438004345, 0.011711244843900223), 
    (2.376515476498753e-05, 0.01267738505266607), (4.543169325188501e-07, 0.0015177460212726151), 
    (3.409600140003022e-06, 0.018415350466966636), (3.4944002891279524e-06, 0.2855767607688904), 
    (1.3396369524798502e-07, 0.0006583309383131573), (6.175905582495034e-07, 0.031902600452303885), 
    (1.264479124074569e-05, 0.050973990932107), (2.231232610938605e-05, 0.05713949967175725), 
    (1.1405156818966589e-09, 0.021382492315024138), (2.0549280055348618e-09, 0.024682952091097846), 
    (3.3339190395054175e-06, 0.14517886489629747), (1.7259408124914444e-08, 0.01276048296131194), 
    (1.8087397393173887e-06, 0.08748787157237532), (3.655995328699646e-07, 0.10239365324378022), 
    (2.1377220036811195e-05, 0.22725961357355118), (3.9081238355720416e-05, 0.25498466938734066), 
    (3.082554712818819e-06, 0.6337731003761291), (5.294374361142218e-08, 0.00830749128945172), 
    (1.74538217834197e-05, 0.14103781953454025), (4.109945621166844e-06, 0.1965724654495717), 
    (1.3196885220168042e-06, 0.20960539430379874), (1.560291821078863e-05, 0.12181339934468284), 
    (1.7525019657682606e-08, 0.26781445741653453), (9.214115692657288e-09, 0.2741092413663864), 
    (3.705958562250089e-08, 0.05159267019480475), (1.8201465934453154e-07, 0.07845675386488438), 
    (1.978705341798559e-07, 0.21748517602682124), (1.297534311106574e-07, 0.271655547618866), 
    (1.9878497070635603e-08, 1.1203549274796392e-05), (3.0873905654260625e-09, 3.264578106154661e-06)
]

# ============================================================================
# FAIRSEQ INITIALIZATION
# ============================================================================

def init_fairseq():
    """Initialize fairseq user module. Call this before loading fairseq models."""
    class MyDefaultObj:
        def __init__(self):
            self.user_dir = f"{FAIRSEQ_PATH}/examples/data2vec"
    
    utils.import_user_module(MyDefaultObj())

# ============================================================================
# BLENDSHAPES UTILITY FUNCTIONS
# ============================================================================

def unnormalize_nemo_blendshapes(blendshape_labels: np.ndarray, blendshapes_idx: List[int]) -> np.ndarray:
    """
    Unnormalize NeMo model outputs using min/max boundaries.
    
    Args:
        blendshape_labels: Normalized blendshape values (N, 52)
        blendshapes_idx: Indices of blendshapes to unnormalize
    
    Returns:
        Unnormalized blendshape values
    """
    boundaries = np.stack(NEMO_BLENDSHAPE_BOUNDARIES)[blendshapes_idx, :]
    unnormalized_labels = blendshape_labels.copy()
    unnormalized_labels[:, blendshapes_idx] = (
        (unnormalized_labels[:, blendshapes_idx]) * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
    )
    return unnormalized_labels

def unnormalize_fairseq_blendshapes(
    blendshape_labels: np.ndarray, 
    blendshapes_idx: List[int], 
    normalization_factors
) -> np.ndarray:
    """
    Unnormalize Fairseq model outputs using mean/std normalization.
    
    Args:
        blendshape_labels: Normalized blendshape values (N, 52)
        blendshapes_idx: Indices of blendshapes to unnormalize
        normalization_factors: DataFrame with 'Mean' and 'Std' columns
    
    Returns:
        Unnormalized blendshape values
    """
    unnormalized_labels = blendshape_labels.copy()
    std = normalization_factors["Std"].values[np.array(blendshapes_idx)]
    mean = normalization_factors["Mean"].values[np.array(blendshapes_idx)]
    unnormalized_labels[:, blendshapes_idx] = (
        (blendshape_labels[:, blendshapes_idx] * std) + mean
    )
    return unnormalized_labels

def zeropad_blendshapes_to_52(blendshapes: np.ndarray, blendshapes_idx: List[int]) -> np.ndarray:
    """
    Pad blendshapes array to full 52 dimensions with zeros.
    
    Args:
        blendshapes: Blendshape values (N, K) where K <= 52
        blendshapes_idx: Indices where blendshapes should be placed
    
    Returns:
        Zero-padded blendshapes (N, 52)
    """
    blendshapes_52 = np.zeros((blendshapes.shape[0], 52))
    blendshapes_52[:, blendshapes_idx] = blendshapes
    return blendshapes_52

def unnormalize_blendshapes(
    blendshapes: np.ndarray, 
    model_type: str, 
    blendshapes_idx: List[int],
    normalization_factors=None
) -> np.ndarray:

    blendshapes_unnorm = blendshapes.copy()
    
    if model_type == 'fairseq':
        if normalization_factors is None:
            raise ValueError("normalization_factors required for fairseq models")
        blendshapes_unnorm = unnormalize_fairseq_blendshapes(
            blendshapes, blendshapes_idx, normalization_factors
        )
    elif model_type == 'nemo':
        blendshapes_unnorm = unnormalize_nemo_blendshapes(blendshapes, blendshapes_idx)
    
    return blendshapes_unnorm

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_fairseq_model(model_path: str, device: str, use_fp16: bool = True):
    """
    Load and prepare a Fairseq model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on (e.g., 'cuda:0')
        use_fp16: Whether to use half precision
    
    Returns:
        Tuple of (model, saved_cfg, task)
    """
    print(f"Loading Fairseq model from {model_path}")
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path], suffix='', strict=False
    )
    model = models[0].eval()
    if use_fp16:
        model.half()
    model.to(device)
    print(f"Loaded Fairseq model")
    return model, saved_cfg, task

def load_nemo_model(model_path: str, device: str, use_fp16: bool = True):
    """
    Load and prepare a NeMo model.
    
    Args:
        model_path: Path to .nemo model file
        device: Device to load model on (e.g., 'cuda:0')
        use_fp16: Whether to use half precision
    
    Returns:
        NeMo model
    """
    print(f"Loading NeMo model from {model_path}")
    model = nemo_asr.models.EncDecCTCModel.restore_from(model_path).eval()
    if use_fp16:
        model.half()
    model.to(device)
    print(f"Loaded NeMo model")
    return model

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def infer_fairseq_model(
    model, 
    sample: torch.Tensor, 
    padding_mask: torch.Tensor, 
    blendshapes_idx: List[int]
) -> np.ndarray:
    with torch.inference_mode():
        net_output = model(**{"source": sample, "padding_mask": padding_mask})
        blendshapes = net_output["encoder_blends"].cpu().numpy().squeeze().transpose(1, 0)
        # print(sample.shape)
        # print(blendshapes.shape)
        blendshapes = zeropad_blendshapes_to_52(blendshapes, blendshapes_idx)
    return blendshapes

def infer_nemo_model(
    model, 
    data: np.ndarray, 
    device: str, 
    blendshapes_idx: List[int]
) -> np.ndarray:
    """
    Run inference on NeMo model.
    
    Args:
        model: NeMo model
        data: Input features array (time, features)
        device: Device model is on
        blendshapes_idx: Indices of blendshapes to extract
    
    Returns:
        Blendshapes array (time, 52)
    """
    with torch.no_grad():
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=torch.tensor(data).unsqueeze(0).to(device).half(), 
            length=torch.tensor(data.shape[0]).unsqueeze(0).to(device)
        )
        encoder_output = model.encoder(audio_signal=processed_signal, length=processed_signal_length)
        output_blendshapes = model.decoder.blendshapes_head(encoder_output[0]).cpu().numpy().squeeze().transpose(1, 0)
        output_blendshapes = zeropad_blendshapes_to_52(
            output_blendshapes[:, :len(blendshapes_idx)], blendshapes_idx
        )
    return output_blendshapes

def prepare_fairseq_sample(data: np.ndarray, device: str, start: int = 0, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare input sample for Fairseq model inference.
    
    Args:
        data: Input features array
        device: Device to place tensors on
        start: Start index
        end: End index (None for all)
    
    Returns:
        Tuple of (sample, padding_mask)
    """
    stack = np.array(data[start:end])
    sample = torch.from_numpy(stack).half().float().unsqueeze(0).to(device).half()
    padding_mask = torch.Tensor([False] * sample.shape[1]).unsqueeze(0).to(device)
    return sample, padding_mask

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
        run_path: Path to run directory
    
    Returns:
        File ID (XXXX) or None if not found
    """
    zip_files = list(run_path.glob("*.right.zip"))
    if not zip_files:
        return None
    # Extract the ID from filename (everything before .right.zip)
    return zip_files[0].stem.replace('.right', '')

