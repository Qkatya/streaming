#!/usr/bin/env python3
"""
Plot comparison of GT peaks before and after manual corrections.

Shows:
1. Top subplot: Original automatic GT peak detection
2. Bottom subplot: GT peaks with manual label corrections applied
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import zscore
import pickle
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))
from blendshapes_data_utils import load_ground_truth_blendshapes
import plotly.io as pio
pio.renderers.default = "browser"

# Configuration
LABELS_CSV = "manual_blink_labels.csv"
MANUAL_GT_PEAKS_FILE = "manual_gt_peaks_from_labels.pkl"  # Per-file manual peaks dictionary
CONCATENATED_SIGNALS_FILE = "concatenated_signals_20251203_214121.pkl"  # NEW file with file order!

# Peak detection parameters (same as in analysis)
HEIGHT_THRESHOLD = 0.0
GT_PROMINENCE = 1.0
PRED_PROMINENCE = 0.5
DISTANCE = 5
MAX_OFFSET = 8

print("="*80)
print("PLOTTING GT PEAKS COMPARISON: BEFORE vs AFTER MANUAL CORRECTIONS")
print("="*80)

# Load concatenated signals
print(f"\nLoading concatenated signals from: {CONCATENATED_SIGNALS_FILE}")
with open(CONCATENATED_SIGNALS_FILE, 'rb') as f:
    data = pickle.load(f)

gt_concat = data['gt_concat']
pred_concat = data['pred_concat']

print(f"GT signal shape: {gt_concat.shape}")
print(f"Pred signal shape: {pred_concat.shape}")

# Detect original GT peaks (automatic detection)
print("\nDetecting original GT peaks (automatic)...")
gt_smooth = savgol_filter(gt_concat, 9, 2, mode='interp')
gt_zscore = zscore(gt_smooth)
original_gt_peaks, _ = find_peaks(gt_zscore, height=HEIGHT_THRESHOLD, prominence=GT_PROMINENCE, distance=DISTANCE)
print(f"  Found {len(original_gt_peaks)} automatic GT peaks")

# Detect pred peaks
print("\nDetecting prediction peaks...")
pred_smooth = savgol_filter(pred_concat, 9, 2, mode='interp')
pred_zscore = zscore(pred_smooth)
pred_peaks, _ = find_peaks(pred_zscore, height=HEIGHT_THRESHOLD, prominence=PRED_PROMINENCE, distance=DISTANCE)
print(f"  Found {len(pred_peaks)} prediction peaks")

# Load manual GT peaks
print(f"\nLoading manual GT peaks from: {MANUAL_GT_PEAKS_FILE}")
with open(MANUAL_GT_PEAKS_FILE, 'rb') as f:
    manual_gt_peaks_dict = pickle.load(f)

print(f"  Loaded manual GT peaks dictionary with {len(manual_gt_peaks_dict)} files")

# Check if concatenated file has the file order information
if 'file_order' in data and 'file_offsets' in data:
    # Convert manual peaks dict to concatenated format using file order
    print("  Converting manual GT peaks to concatenated format...")
    manual_gt_peaks_list = []
    files_with_manual_peaks = 0
    
    for (run_path, tar_id, side), offset in zip(data['file_order'], data['file_offsets']):
        key = (run_path, tar_id, side)
        if key in manual_gt_peaks_dict:
            # Add offset to convert from per-file peaks to concatenated peaks
            file_peaks = np.array(manual_gt_peaks_dict[key])
            concat_peaks = file_peaks + offset
            manual_gt_peaks_list.extend(concat_peaks)
            files_with_manual_peaks += 1
    
    manual_gt_peaks = np.array(sorted(manual_gt_peaks_list), dtype=np.int64)
    print(f"  Converted to {len(manual_gt_peaks)} concatenated manual GT peaks")
    print(f"  Found manual peaks in {files_with_manual_peaks}/{len(data['file_order'])} files")
else:
    # No file order info - need to use the labels CSV to map peaks
    print("  WARNING: No file order in concatenated file!")
    print("  Cannot convert manual peaks without file order information.")
    print("  Please re-run blinking_split_analysis.py to regenerate concatenated_signals file with file order.")
    print("  Using original automatic peaks as fallback for now.")
    manual_gt_peaks = original_gt_peaks

# Load labels to show statistics
labels_df = pd.read_csv(LABELS_CSV)
label_counts = labels_df['label'].value_counts()

print(f"\nManual label statistics:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

# Match peaks for both scenarios
def match_peaks(gt_peaks, pred_peaks, max_offset=MAX_OFFSET):
    """Match GT and prediction peaks."""
    matched_gt = []
    matched_pred = []
    unmatched_gt = []
    unmatched_pred = list(pred_peaks)
    
    for gt_peak in gt_peaks:
        distances = np.abs(pred_peaks - gt_peak)
        close_peaks = np.where(distances <= max_offset)[0]
        
        if len(close_peaks) > 0:
            closest_idx = close_peaks[np.argmin(distances[close_peaks])]
            closest_pred_peak = pred_peaks[closest_idx]
            
            matched_gt.append(gt_peak)
            matched_pred.append(closest_pred_peak)
            
            if closest_pred_peak in unmatched_pred:
                unmatched_pred.remove(closest_pred_peak)
        else:
            unmatched_gt.append(gt_peak)
    
    return {
        'matched_gt': matched_gt,
        'matched_pred': matched_pred,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred
    }

# Match peaks for original
print("\nMatching peaks (original automatic GT)...")
matches_original = match_peaks(original_gt_peaks, pred_peaks)
print(f"  Matched: {len(matches_original['matched_gt'])} GT, {len(matches_original['matched_pred'])} Pred")
print(f"  Unmatched: {len(matches_original['unmatched_gt'])} GT, {len(matches_original['unmatched_pred'])} Pred")

# Match peaks for manual
print("\nMatching peaks (manual corrected GT)...")
matches_manual = match_peaks(manual_gt_peaks, pred_peaks)
print(f"  Matched: {len(matches_manual['matched_gt'])} GT, {len(matches_manual['matched_pred'])} Pred")
print(f"  Unmatched: {len(matches_manual['unmatched_gt'])} GT, {len(matches_manual['unmatched_pred'])} Pred")

# Calculate metrics
def calculate_metrics(matches, total_frames, fps=25):
    """Calculate TPR and FPR."""
    matched_gt = len(matches['matched_gt'])
    unmatched_gt = len(matches['unmatched_gt'])
    unmatched_pred = len(matches['unmatched_pred'])
    
    total_gt = matched_gt + unmatched_gt
    tpr = (matched_gt / total_gt * 100) if total_gt > 0 else 0
    
    duration_minutes = total_frames / fps / 60
    fpr = unmatched_pred / duration_minutes if duration_minutes > 0 else 0
    
    return tpr, fpr

tpr_original, fpr_original = calculate_metrics(matches_original, len(gt_concat))
tpr_manual, fpr_manual = calculate_metrics(matches_manual, len(gt_concat))

print(f"\nMetrics comparison:")
print(f"  Original - TPR: {tpr_original:.2f}%, FPR: {fpr_original:.2f} per minute")
print(f"  Manual   - TPR: {tpr_manual:.2f}%, FPR: {fpr_manual:.2f} per minute")
print(f"  Improvement - TPR: {tpr_manual - tpr_original:+.2f}%, FPR: {fpr_manual - fpr_original:+.2f} per minute")

# Identify added and removed peaks
print("\nIdentifying added and removed peaks...")
original_gt_set = set(original_gt_peaks)
manual_gt_set = set(manual_gt_peaks)

added_peaks = sorted(manual_gt_set - original_gt_set)
removed_peaks = sorted(original_gt_set - manual_gt_set)

print(f"  Added peaks: {len(added_peaks)}")
print(f"  Removed peaks: {len(removed_peaks)}")

# Create visualization
print("\nCreating visualization...")

# Use ALL frames
plot_frames = len(gt_concat)
frames = np.arange(plot_frames)

# Create two subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        f"ORIGINAL: Automatic GT Detection (TPR: {tpr_original:.1f}%, FPR: {fpr_original:.2f}/min)",
        f"AFTER MANUAL CORRECTIONS (TPR: {tpr_manual:.1f}%, FPR: {fpr_manual:.2f}/min) - Added: {len(added_peaks)}, Removed: {len(removed_peaks)}"
    ),
    vertical_spacing=0.08,
    row_heights=[0.5, 0.5]
)

# ============================================================================
# SUBPLOT 1: ORIGINAL AUTOMATIC DETECTION
# ============================================================================

# GT signal
fig.add_trace(
    go.Scatter(
        x=frames,
        y=gt_zscore[:plot_frames],
        mode='lines',
        name='GT Z-Score',
        line=dict(color='#B0C4DE', width=1.0),
        showlegend=True
    ),
    row=1, col=1
)

# Pred signal
fig.add_trace(
    go.Scatter(
        x=frames,
        y=pred_zscore[:plot_frames],
        mode='lines',
        name='Pred Z-Score',
        line=dict(color='#FFB6C1', width=1.0),
        showlegend=True
    ),
    row=1, col=1
)

# Matched GT peaks (original) - GREEN
matched_gt_orig = [p for p in matches_original['matched_gt'] if p < plot_frames]
if matched_gt_orig:
    fig.add_trace(
        go.Scatter(
            x=matched_gt_orig,
            y=gt_zscore[matched_gt_orig],
            mode='markers',
            name=f'Matched GT ({len(matches_original["matched_gt"])})',
            marker=dict(
                color='green',
                size=10,
                symbol='circle',
                line=dict(color='darkgreen', width=1.5)
            ),
            showlegend=True
        ),
        row=1, col=1
    )

# Unmatched GT peaks (original) - BLUE
unmatched_gt_orig = [p for p in matches_original['unmatched_gt'] if p < plot_frames]
if unmatched_gt_orig:
    fig.add_trace(
        go.Scatter(
            x=unmatched_gt_orig,
            y=gt_zscore[unmatched_gt_orig],
            mode='markers',
            name=f'Unmatched GT ({len(matches_original["unmatched_gt"])})',
            marker=dict(
                color='blue',
                size=11,
                symbol='diamond',
                line=dict(color='darkblue', width=1.5)
            ),
            showlegend=True
        ),
        row=1, col=1
    )

# Matched Pred peaks (original) - YELLOW
matched_pred_orig = [p for p in matches_original['matched_pred'] if p < plot_frames]
if matched_pred_orig:
    fig.add_trace(
        go.Scatter(
            x=matched_pred_orig,
            y=pred_zscore[matched_pred_orig],
            mode='markers',
            name=f'Matched Pred ({len(matches_original["matched_pred"])})',
            marker=dict(
                color='yellow',
                size=9,
                symbol='square',
                line=dict(color='orange', width=1.5)
            ),
            showlegend=True
        ),
        row=1, col=1
    )

# Unmatched Pred peaks (original) - RED
unmatched_pred_orig = [p for p in matches_original['unmatched_pred'] if p < plot_frames]
if unmatched_pred_orig:
    fig.add_trace(
        go.Scatter(
            x=unmatched_pred_orig,
            y=pred_zscore[unmatched_pred_orig],
            mode='markers',
            name=f'Unmatched Pred ({len(matches_original["unmatched_pred"])})',
            marker=dict(
                color='red',
                size=12,
                symbol='x',
                line=dict(width=2.5)
            ),
            showlegend=True
        ),
        row=1, col=1
    )

# ============================================================================
# SUBPLOT 2: AFTER MANUAL CORRECTIONS
# ============================================================================

# GT signal
fig.add_trace(
    go.Scatter(
        x=frames,
        y=gt_zscore[:plot_frames],
        mode='lines',
        name='GT Z-Score',
        line=dict(color='#B0C4DE', width=1.0),
        showlegend=False
    ),
    row=2, col=1
)

# Pred signal
fig.add_trace(
    go.Scatter(
        x=frames,
        y=pred_zscore[:plot_frames],
        mode='lines',
        name='Pred Z-Score',
        line=dict(color='#FFB6C1', width=1.0),
        showlegend=False
    ),
    row=2, col=1
)

# Matched GT peaks (manual) - GREEN
matched_gt_manual = [p for p in matches_manual['matched_gt'] if p < plot_frames]
if matched_gt_manual:
    fig.add_trace(
        go.Scatter(
            x=matched_gt_manual,
            y=gt_zscore[matched_gt_manual],
            mode='markers',
            name=f'Matched GT ({len(matches_manual["matched_gt"])})',
            marker=dict(
                color='green',
                size=10,
                symbol='circle',
                line=dict(color='darkgreen', width=1.5)
            ),
            showlegend=False
        ),
        row=2, col=1
    )

# Unmatched GT peaks (manual) - BLUE
unmatched_gt_manual = [p for p in matches_manual['unmatched_gt'] if p < plot_frames]
if unmatched_gt_manual:
    fig.add_trace(
        go.Scatter(
            x=unmatched_gt_manual,
            y=gt_zscore[unmatched_gt_manual],
            mode='markers',
            name=f'Unmatched GT ({len(matches_manual["unmatched_gt"])})',
            marker=dict(
                color='blue',
                size=11,
                symbol='diamond',
                line=dict(color='darkblue', width=1.5)
            ),
            showlegend=False
        ),
        row=2, col=1
    )

# Matched Pred peaks (manual) - YELLOW
matched_pred_manual = [p for p in matches_manual['matched_pred'] if p < plot_frames]
if matched_pred_manual:
    fig.add_trace(
        go.Scatter(
            x=matched_pred_manual,
            y=pred_zscore[matched_pred_manual],
            mode='markers',
            name=f'Matched Pred ({len(matches_manual["matched_pred"])})',
            marker=dict(
                color='yellow',
                size=9,
                symbol='square',
                line=dict(color='orange', width=1.5)
            ),
            showlegend=False
        ),
        row=2, col=1
    )

# Unmatched Pred peaks (manual) - RED
unmatched_pred_manual = [p for p in matches_manual['unmatched_pred'] if p < plot_frames]
if unmatched_pred_manual:
    fig.add_trace(
        go.Scatter(
            x=unmatched_pred_manual,
            y=pred_zscore[unmatched_pred_manual],
            mode='markers',
            name=f'Unmatched Pred ({len(matches_manual["unmatched_pred"])})',
            marker=dict(
                color='red',
                size=12,
                symbol='x',
                line=dict(width=2.5)
            ),
            showlegend=False
        ),
        row=2, col=1
    )

# ADDED PEAKS - Large MAGENTA stars (only in bottom plot)
added_peaks_in_range = [p for p in added_peaks if p < plot_frames]
if added_peaks_in_range:
    fig.add_trace(
        go.Scatter(
            x=added_peaks_in_range,
            y=gt_zscore[added_peaks_in_range],
            mode='markers',
            name=f'✨ ADDED ({len(added_peaks)})',
            marker=dict(
                color='magenta',
                size=18,
                symbol='star',
                line=dict(color='purple', width=3)
            ),
            showlegend=True
        ),
        row=2, col=1
    )

# REMOVED PEAKS - Large CYAN X's (only in bottom plot)
removed_peaks_in_range = [p for p in removed_peaks if p < plot_frames]
if removed_peaks_in_range:
    fig.add_trace(
        go.Scatter(
            x=removed_peaks_in_range,
            y=gt_zscore[removed_peaks_in_range],
            mode='markers',
            name=f'❌ REMOVED ({len(removed_peaks)})',
            marker=dict(
                color='cyan',
                size=18,
                symbol='x',
                line=dict(color='teal', width=4)
            ),
            showlegend=True
        ),
        row=2, col=1
    )

# Update layout
fig.update_xaxes(title_text="Frame (25 fps)", row=1, col=1)
fig.update_xaxes(title_text="Frame (25 fps)", row=2, col=1, matches='x')  # Link x-axis
fig.update_yaxes(title_text="Z-Score", row=1, col=1)
fig.update_yaxes(title_text="Z-Score", row=2, col=1)

fig.update_layout(
    title=dict(
        text=f"GT Peak Detection: Before vs After Manual Corrections<br>" +
             f"<sub>Total frames: {plot_frames:,} | GT peaks: {len(original_gt_peaks)} → {len(manual_gt_peaks)} | " +
             f"Added: {len(added_peaks)} | Removed: {len(removed_peaks)} | " +
             f"TPR: {tpr_original:.1f}% → {tpr_manual:.1f}% | FPR: {fpr_original:.2f} → {fpr_manual:.2f}/min</sub>",
        font=dict(size=18)
    ),
    hovermode='x unified',
    height=1000,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.01,
        font=dict(size=10),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1
    ),
    template='plotly_white'
)

print(f"\nShowing ALL {plot_frames:,} frames")
print("\nColor Legend:")
print("  🟢 GREEN circles = Matched GT peaks (correctly detected by model)")
print("  🔵 BLUE diamonds = Unmatched GT peaks (missed by model)")
print("  🟨 YELLOW squares = Matched Pred peaks (correct predictions)")
print("  🔴 RED X's = Unmatched Pred peaks (false positives)")
print("  💜 MAGENTA stars = ADDED peaks (from manual labeling)")
print("  🩵 CYAN X's = REMOVED peaks (from manual labeling)")

print("\nOpening plot in browser...")
fig.show()

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nSummary:")
print(f"  Original GT peaks: {len(original_gt_peaks)}")
print(f"  Manual GT peaks: {len(manual_gt_peaks)}")
print(f"  Added: {len(added_peaks)} peaks")
print(f"  Removed: {len(removed_peaks)} peaks")
print(f"  Net change: {len(manual_gt_peaks) - len(original_gt_peaks):+d} peaks")
print(f"\nMetrics:")
print(f"  TPR: {tpr_original:.2f}% → {tpr_manual:.2f}% (change: {tpr_manual - tpr_original:+.2f}%)")
print(f"  FPR: {fpr_original:.2f}/min → {fpr_manual:.2f}/min (change: {fpr_manual - fpr_original:+.2f}/min)")
print(f"  Matched predictions: {len(matches_original['matched_pred'])} → {len(matches_manual['matched_pred'])} (change: {len(matches_manual['matched_pred']) - len(matches_original['matched_pred']):+d})")

