#!/usr/bin/env python3
"""
Extract video snippets around unmatched prediction timestamps.
Takes 2 seconds around each timestamp and saves them to video_snippets folder.
"""

import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import os

# Configuration
PICKLE_FILE = "unmatched_pred_peaks_th0.007071.pkl"
VIDEO_BASE_PATH = Path("/mnt/A3000/Recordings/v2_data")
OUTPUT_DIR = Path("video_snippets")
SNIPPET_DURATION = 2.0  # seconds (1 second before and 1 second after)

def extract_video_snippet(video_path, timestamp_seconds, output_path, duration=2.0):
    """
    Extract a video snippet around the given timestamp.
    
    Args:
        video_path: Path to the source video
        timestamp_seconds: Center timestamp in seconds
        output_path: Path to save the output snippet
        duration: Total duration of the snippet (centered on timestamp)
    
    Returns:
        True if successful, False otherwise
    """
    # Calculate start time (1 second before timestamp, but not negative)
    start_time = max(0, timestamp_seconds - duration / 2)
    
    # Build ffmpeg command
    # -ss: start time
    # -i: input file
    # -t: duration
    # -c copy: copy codec (fast, no re-encoding)
    # -avoid_negative_ts make_zero: handle negative timestamps
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', str(video_path),
        '-t', str(duration),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output file if exists
        str(output_path)
    ]
    
    try:
        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting snippet: {e}")
        print(f"stderr: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False

def sanitize_filename(text):
    """Remove or replace characters that are problematic in filenames."""
    # Replace slashes with underscores
    text = text.replace('/', '_')
    text = text.replace('\\', '_')
    return text

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("VIDEO SNIPPET EXTRACTION")
    print("="*80)
    
    # Load the pickle file
    print(f"\nLoading data from {PICKLE_FILE}...")
    df = pd.read_pickle(PICKLE_FILE)
    print(f"Loaded {len(df)} unmatched predictions")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Process each row
    successful = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*80)
    print("EXTRACTING VIDEO SNIPPETS")
    print("="*80)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        global_frame = row['global_frame']
        file_index = row['file_index']
        local_frame = row['local_frame']
        timestamp_seconds = row['timestamp_seconds']
        run_path = row['run_path']
        
        # Construct video path
        video_path = VIDEO_BASE_PATH / run_path / "video_full.mp4"
        
        # Check if video exists
        if not video_path.exists():
            print(f"\nWarning: Video not found: {video_path}")
            skipped += 1
            continue
        
        # Create output filename
        # Format: globalframe_fileindex_localframe_timestamp_runpath.mp4
        sanitized_run_path = sanitize_filename(run_path)
        output_filename = f"{global_frame}_{file_index}_{local_frame}_{timestamp_seconds:.2f}_{sanitized_run_path}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Extract snippet
        success = extract_video_snippet(
            video_path,
            timestamp_seconds,
            output_path,
            duration=SNIPPET_DURATION
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"\nFailed to extract: {output_filename}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total unmatched predictions: {len(df)}")
    print(f"Successfully extracted: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (video not found): {skipped}")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()


