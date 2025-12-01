import numpy as np
from typing import List, Dict
from pathlib import Path
import pandas as pd
from visualization import plot_blink_analysis
from blendshape_utils import resample_to_30hz
from scipy.signal import savgol_filter
from scipy.stats import zscore

def normalize_blinks(data):
    return (data - np.median(data)) / (1 - np.median(data))

class BlinkAnalyzer:
    def __init__(self, sample_rate: float = 30.0):
        self.sample_rate = sample_rate
        
    def analyze_blinks(self, 
                      blendshapes_list: List[np.ndarray], 
                      pred_blends_list: List[np.ndarray],
                      pred_blends_diff_list: List[np.ndarray] = None,
                      gt_th: float = 0.1,
                      model_th: float = 0.06,
                      max_offset: int = 10,
                      significant_gt_movenents: np.ndarray = None) -> Dict:
        """
        Analyze blink patterns in ground truth and predicted data.
        If pred_blends_diff_list is provided, use it for differential analysis instead of computing derivatives.
        """
        # Extract blink signals
        self.gt_th = gt_th
        self.model_th = model_th
        # blinks_examples = [self._extract_right_blinks(x) for x in blendshapes_list]
        # pred_blinks_examples = [self._extract_right_blinks(x) for x in pred_blends_list]
        blinks_examples = [self._extract_blinks(x) for x in blendshapes_list]
        pred_blinks_examples = [self._extract_blinks(x) for x in pred_blends_list]
        significant_blinks_movenents = [self._extract_right_blinks(x) for x in significant_gt_movenents]
        # gt_diff_examples = [np.diff(savgol_filter(x, 9, 2, mode='interp')) for x in blinks_examples] #!!!!!!!!!!!!!!
        
        gt_diff_examples = [(np.diff(x)) for x in blinks_examples]
        pred_blinks_diff_examples = [(np.diff(savgol_filter(x, 9, 2, mode='interp'))) for x in pred_blinks_examples]
            
        # gt_diff_examples = [np.diff(x) for x in blinks_examples]
        # pred_blinks_diff_examples = [np.diff(x) for x in pred_blinks_examples]
        
        # for i in range(len(gt_diff_examples)): #!!!!!!!!!!!!!!
        #     gt_diff_examples[i] *= significant_blinks_movenents[i]
        #     pred_blinks_diff_examples[i] *= significant_blinks_movenents[i]
        
        resampled_data = self._align_differential(
                blinks_examples, 
                pred_blinks_examples,
                gt_diff_examples,
                pred_blinks_diff_examples
            )
        
        # if pred_blends_diff_list is not None:
        #     # Use provided differential signals
        #     pred_blinks_examples = [self._extract_blinks(x) for x in pred_blends_list]
        #     pred_blinks_diff_examples = [self._extract_blinks(x) for x in pred_blends_diff_list]
            
        #     # Get derivatives of ground truth only
        #     gt_diff_examples = [np.diff(x) for x in blinks_examples]
            
        #     # Resample and align differential signals
        #     resampled_data = self._resample_and_align_differential(
        #         blinks_examples, 
        #         pred_blinks_examples,
        #         gt_diff_examples,
        #         pred_blinks_diff_examples
        #     )
        # else:
        #     # Fallback to original behavior
        #     pred_blinks_examples = [self._extract_blinks(x) for x in pred_blends_list]
        #     resampled_data = self._resample_and_align(blinks_examples, pred_blinks_examples)
        
        # Generate analysis plots
        plots, matches = plot_blink_analysis(
            resampled_data['gt_concat'],
            resampled_data['pred_concat'],
            self.sample_rate,
            gt_diff=resampled_data.get('gt_diff_concat'),
            pred_diff=resampled_data.get('pred_diff_concat'),
            gt_th=self.gt_th,
            model_th=self.model_th,
            max_offset=max_offset,
            significant_gt_movenents=significant_gt_movenents
        )
        
        # Calculate metrics using differential data if available
        metrics = self._calculate_metrics(resampled_data)
        
        return {
            'plots': plots,
            'metrics': metrics,
            'processed_data': resampled_data,
            'matches': matches,
            'pred_concat': resampled_data['pred_concat'],
            'gt_concat': resampled_data['gt_concat']
        }

    @staticmethod
    def _extract_blinks(data: np.ndarray) -> np.ndarray:
        """Extract blink signal from blendshape data."""
        if data.shape[1]==51:
            return (data[:,8] + data[:,9])/2  # Combine left and right blink
        return (data[:,9] + data[:,10])/2  # Combine left and right blink
    
    def _extract_right_blinks(self, data: np.ndarray) -> np.ndarray:
        """Extract blink signal from blendshape data."""
        if data.shape[1]==51:
            return (data[:,9])  # Combine left and right blink
        return (data[:,10])  # Combine left and right blink
    
    def _extract_blinks_mask(self, data: np.ndarray) -> np.ndarray:
        """Extract blink signal from blendshape data."""
        if data.shape[1]==51:
            return (data[:,8] * data[:,9])  # Combine left and right blink
        
        return (data[:,9] + data[:,10])/2  # Combine left and right blink
    
    def _resample_and_align(self, 
                           gt_sequences: List[np.ndarray], 
                           pred_sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Resample predictions to match ground truth sampling rate and align sequences."""
        resampled_pred_sequences = []
        matched_gt_sequences = []

        for gt_seq, pred_seq in zip(gt_sequences, pred_sequences):
            # Resample prediction from 25Hz to 30Hz
            pred_resampled = resample_to_30hz(pred_seq)
            
            # Trim both sequences to same length
            min_len = min(len(pred_resampled), len(gt_seq))
            pred_resampled = pred_resampled[:min_len]
            gt_seq = gt_seq[:min_len]
            
            resampled_pred_sequences.append(pred_resampled)
            matched_gt_sequences.append(gt_seq)

        # Concatenate all examples
        return {
            'gt_concat': np.concatenate(matched_gt_sequences),
            'pred_concat': np.concatenate(resampled_pred_sequences),
            'gt_sequences': matched_gt_sequences,
            'pred_sequences': resampled_pred_sequences
        }

    def _resample_and_align_differential(self,
                                       gt_sequences: List[np.ndarray],
                                       pred_sequences: List[np.ndarray],
                                       gt_diff_sequences: List[np.ndarray],
                                       pred_diff_sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Resample and align both regular and differential signals."""
        resampled_pred_sequences = []
        resampled_pred_diff_sequences = []
        matched_gt_sequences = []
        matched_gt_diff_sequences = []

        for gt_seq, pred_seq, gt_diff, pred_diff in zip(
            gt_sequences, pred_sequences, gt_diff_sequences, pred_diff_sequences):
            
            # Resample predictions from 25Hz to 30Hz
            pred_resampled = resample_to_30hz(pred_seq)
            pred_diff_resampled = resample_to_30hz(pred_diff)
            
            # Trim all sequences to same length
            min_len = min(len(pred_resampled), len(gt_seq))
            min_len_diff = min_len - 1  # Differential signals are one sample shorter
            
            pred_resampled = pred_resampled[:min_len]
            gt_seq = gt_seq[:min_len]
            pred_diff_resampled = pred_diff_resampled[:min_len_diff]
            gt_diff = gt_diff[:min_len_diff]
            
            resampled_pred_sequences.append(pred_resampled)
            resampled_pred_diff_sequences.append(pred_diff_resampled)
            matched_gt_sequences.append(gt_seq)
            matched_gt_diff_sequences.append(gt_diff)

        return {
            'gt_concat': np.concatenate(matched_gt_sequences),
            'pred_concat': np.concatenate(resampled_pred_sequences),
            'gt_diff_concat': np.concatenate(matched_gt_diff_sequences),
            'pred_diff_concat': np.concatenate(resampled_pred_diff_sequences),
            'gt_sequences': matched_gt_sequences,
            'pred_sequences': resampled_pred_sequences,
            'gt_diff_sequences': matched_gt_diff_sequences,
            'pred_diff_sequences': resampled_pred_diff_sequences
        }
        
    def _align_differential(self,
                                       gt_sequences: List[np.ndarray],
                                       pred_sequences: List[np.ndarray],
                                       gt_diff_sequences: List[np.ndarray],
                                       pred_diff_sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Resample and align both regular and differential signals."""
        resampled_pred_sequences = []
        resampled_pred_diff_sequences = []
        matched_gt_sequences = []
        matched_gt_diff_sequences = []

        for gt_seq, pred_seq, gt_diff, pred_diff in zip(
            gt_sequences, pred_sequences, gt_diff_sequences, pred_diff_sequences):
            
            # Resample predictions from 25Hz to 30Hz
            pred_resampled = pred_seq#resample_to_30hz(pred_seq)
            pred_diff_resampled = pred_diff#resample_to_30hz(pred_diff)
            
            # Trim all sequences to same length
            min_len = min(len(pred_resampled), len(gt_seq))
            min_len_diff = min_len - 1  # Differential signals are one sample shorter
            
            pred_resampled = pred_resampled[:min_len]
            gt_seq = gt_seq[:min_len]
            pred_diff_resampled = pred_diff_resampled[:min_len_diff]
            gt_diff = gt_diff[:min_len_diff]
            
            resampled_pred_sequences.append(pred_resampled)
            resampled_pred_diff_sequences.append(pred_diff_resampled)
            matched_gt_sequences.append(gt_seq)
            matched_gt_diff_sequences.append(gt_diff)

        return {
            'gt_concat': np.concatenate(matched_gt_sequences),
            'pred_concat': np.concatenate(resampled_pred_sequences),
            'gt_diff_concat': np.concatenate(matched_gt_diff_sequences),
            'pred_diff_concat': np.concatenate(resampled_pred_diff_sequences),
            'gt_sequences': matched_gt_sequences,
            'pred_sequences': resampled_pred_sequences,
            'gt_diff_sequences': matched_gt_diff_sequences,
            'pred_diff_sequences': resampled_pred_diff_sequences
        }

    def _calculate_metrics(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate various metrics for blink analysis."""
        from scipy.signal import find_peaks
        from scipy.stats import pearsonr
        
        gt_filtered = savgol_filter(data['gt_concat'], 9, 2, mode='interp')
        pred_filtered = savgol_filter(data['pred_concat'], 9, 2, mode='interp')
        # gt_filtered = np.convolve(data['gt_concat'], np.ones(3)/3, mode='same')
        # pred_filtered = np.convolve(data['pred_concat'], np.ones(3)/3, mode='same')
        
        # Find peaks
        gt_peaks, _ = find_peaks(np.diff(gt_filtered), height=self.gt_th)
        pred_peaks, _ = find_peaks(np.diff(pred_filtered), height=self.model_th)
        
        # gt_peaks, _ = find_peaks(np.diff(gt_filtered), height=self.gt_th)
        # pred_peaks, _ = find_peaks(np.diff(pred_filtered), height=self.model_th)
        
        # # Find peaks
        # gt_peaks, _ = find_peaks(np.diff(gt_filtered), height=self.gt_th)
        # pred_peaks, _ = find_peaks(np.diff(pred_filtered), height=self.model_th)
        
        # Calculate correlation
        corr, _ = pearsonr(np.diff(gt_filtered), np.diff(pred_filtered))
        
        return {
            'num_gt_peaks': len(gt_peaks),
            'num_pred_peaks': len(pred_peaks),
            'correlation': corr,
            'mean_error': np.mean(np.abs(gt_filtered - pred_filtered)),
            'std_error': np.std(np.abs(gt_filtered - pred_filtered))
        }
