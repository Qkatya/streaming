"""
Utility script to analyze and visualize peak modifications made in the editor.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_modification_files(
    removed_file="removed_peaks.pkl",
    added_file="added_peaks.pkl", 
    final_file="final_gt_peaks.pkl"
):
    """Load all modification files."""
    results = {}
    
    if Path(removed_file).exists():
        with open(removed_file, 'rb') as f:
            results['removed'] = pickle.load(f)
        print(f"✓ Loaded {len(results['removed'])} removed peaks")
    else:
        results['removed'] = set()
        print(f"✗ No removed peaks file found")
    
    if Path(added_file).exists():
        with open(added_file, 'rb') as f:
            results['added'] = pickle.load(f)
        print(f"✓ Loaded {len(results['added'])} added peaks")
    else:
        results['added'] = set()
        print(f"✗ No added peaks file found")
    
    if Path(final_file).exists():
        with open(final_file, 'rb') as f:
            results['final'] = pickle.load(f)
        print(f"✓ Loaded {len(results['final'])} final GT peaks")
    else:
        results['final'] = np.array([])
        print(f"✗ No final peaks file found")
    
    return results

def print_statistics(modifications):
    """Print statistics about modifications."""
    print("\n" + "=" * 80)
    print("MODIFICATION STATISTICS")
    print("=" * 80)
    
    removed = modifications['removed']
    added = modifications['added']
    final = modifications['final']
    
    print(f"\nTotal Modifications:")
    print(f"  Removed peaks: {len(removed)}")
    print(f"  Added peaks: {len(added)}")
    print(f"  Net change: {len(added) - len(removed):+d}")
    print(f"  Final GT peaks: {len(final)}")
    
    if len(removed) > 0:
        print(f"\nRemoved Peak Locations (frames):")
        removed_sorted = sorted(removed)
        print(f"  First 10: {removed_sorted[:10]}")
        if len(removed) > 10:
            print(f"  Last 10: {removed_sorted[-10:]}")
    
    if len(added) > 0:
        print(f"\nAdded Peak Locations (frames):")
        added_sorted = sorted(added)
        print(f"  First 10: {added_sorted[:10]}")
        if len(added) > 10:
            print(f"  Last 10: {added_sorted[-10:]}")
    
    if len(final) > 0:
        print(f"\nFinal Peak Statistics:")
        print(f"  Total peaks: {len(final)}")
        print(f"  First peak: frame {final[0]}")
        print(f"  Last peak: frame {final[-1]}")
        
        # Calculate inter-peak intervals
        if len(final) > 1:
            intervals = np.diff(final)
            print(f"  Mean interval: {np.mean(intervals):.1f} frames ({np.mean(intervals)/30:.2f} sec)")
            print(f"  Median interval: {np.median(intervals):.1f} frames ({np.median(intervals)/30:.2f} sec)")
            print(f"  Min interval: {np.min(intervals)} frames ({np.min(intervals)/30:.2f} sec)")
            print(f"  Max interval: {np.max(intervals)} frames ({np.max(intervals)/30:.2f} sec)")

def plot_modifications_timeline(modifications, sample_rate=30.0):
    """Plot timeline of modifications."""
    removed = modifications['removed']
    added = modifications['added']
    final = modifications['final']
    
    if len(final) == 0:
        print("\nNo peaks to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    
    # Convert frames to time
    final_times = final / sample_rate
    
    # Plot 1: Final peaks timeline
    ax = axes[0]
    ax.scatter(final_times, np.ones_like(final_times), alpha=0.6, s=50)
    ax.set_ylabel('Final GT Peaks')
    ax.set_yticks([])
    ax.set_xlim(0, final_times[-1] * 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Final GT Peaks Timeline ({len(final)} peaks)')
    
    # Plot 2: Modifications
    ax = axes[1]
    if len(removed) > 0:
        removed_times = np.array(sorted(removed)) / sample_rate
        ax.scatter(removed_times, np.ones_like(removed_times), 
                  color='red', alpha=0.7, s=100, marker='x', label='Removed')
    if len(added) > 0:
        added_times = np.array(sorted(added)) / sample_rate
        ax.scatter(added_times, np.ones_like(added_times), 
                  color='green', alpha=0.7, s=100, marker='+', label='Added')
    ax.set_ylabel('Modifications')
    ax.set_yticks([])
    ax.set_xlim(0, final_times[-1] * 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Modifications ({len(removed)} removed, {len(added)} added)')
    
    # Plot 3: Inter-peak intervals
    ax = axes[2]
    if len(final) > 1:
        intervals = np.diff(final) / sample_rate
        ax.plot(final_times[1:], intervals, 'o-', alpha=0.6)
        ax.axhline(np.mean(intervals), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(intervals):.2f}s')
        ax.set_ylabel('Interval (s)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Inter-Peak Intervals')
    
    plt.tight_layout()
    plt.savefig('peak_modifications_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot to: peak_modifications_analysis.png")
    plt.show()

def plot_modification_histogram(modifications, sample_rate=30.0):
    """Plot histogram of peak intervals."""
    final = modifications['final']
    
    if len(final) < 2:
        print("\nNot enough peaks for histogram")
        return
    
    intervals = np.diff(final) / sample_rate
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(intervals, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(intervals), color='r', linestyle='--', 
              label=f'Mean: {np.mean(intervals):.2f}s')
    ax.axvline(np.median(intervals), color='g', linestyle='--', 
              label=f'Median: {np.median(intervals):.2f}s')
    ax.set_xlabel('Inter-Peak Interval (s)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Inter-Peak Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1]
    sorted_intervals = np.sort(intervals)
    cumulative = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
    ax.plot(sorted_intervals, cumulative, linewidth=2)
    ax.set_xlabel('Inter-Peak Interval (s)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Intervals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('peak_intervals_histogram.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to: peak_intervals_histogram.png")
    plt.show()

def export_to_csv(modifications, output_file="peak_modifications.csv"):
    """Export modifications to CSV for further analysis."""
    import pandas as pd
    
    removed = modifications['removed']
    added = modifications['added']
    final = modifications['final']
    
    # Create dataframe for final peaks
    df_final = pd.DataFrame({
        'frame': final,
        'time_seconds': final / 30.0,
        'is_added': [f in added for f in final],
        'peak_type': 'final'
    })
    
    # Create dataframe for removed peaks
    if len(removed) > 0:
        df_removed = pd.DataFrame({
            'frame': sorted(removed),
            'time_seconds': np.array(sorted(removed)) / 30.0,
            'is_added': False,
            'peak_type': 'removed'
        })
        df_combined = pd.concat([df_final, df_removed], ignore_index=True)
    else:
        df_combined = df_final
    
    df_combined = df_combined.sort_values('frame').reset_index(drop=True)
    df_combined.to_csv(output_file, index=False)
    print(f"\n✓ Exported to: {output_file}")
    print(f"  Rows: {len(df_combined)}")

def main():
    """Main entry point."""
    print("=" * 80)
    print("Peak Modification Analyzer")
    print("=" * 80)
    print()
    
    # Load modifications
    print("Loading modification files...")
    modifications = load_modification_files()
    
    if len(modifications['final']) == 0:
        print("\n✗ No modifications found. Run the peak editor first.")
        return
    
    # Print statistics
    print_statistics(modifications)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    try:
        plot_modifications_timeline(modifications)
        plot_modification_histogram(modifications)
    except Exception as e:
        print(f"\n✗ Error creating plots: {e}")
    
    # Export to CSV
    print("\n" + "=" * 80)
    print("Exporting data...")
    print("=" * 80)
    
    try:
        export_to_csv(modifications)
    except Exception as e:
        print(f"\n✗ Error exporting to CSV: {e}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()


