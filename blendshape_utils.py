"""
Utility functions for blendshape processing.
"""
import numpy as np


def resample_to_30hz(signal: np.ndarray, original_fps: int = 25) -> np.ndarray:
    """Resample a signal from original_fps to 30Hz."""
    t_orig = np.linspace(0, len(signal)/original_fps, len(signal))
    t_30hz = np.linspace(0, len(signal)/original_fps, int(len(signal) * 30/original_fps))
    
    target_length = int(len(signal) * 30/original_fps)
    t_30hz = np.linspace(0, len(signal)/original_fps, target_length)
    
    if len(signal.shape) == 1:
        resampled = np.interp(t_30hz, t_orig, signal)
        return resampled[:target_length]
    else:
        resampled = np.array([np.interp(t_30hz, t_orig, signal[:,i]) 
                        for i in range(signal.shape[1])]).T
        return resampled[:target_length]

