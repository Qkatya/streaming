#!/usr/bin/env python3
"""
Process video snippets for unmatched predicted peaks.

This script:
1. Loads unmatched peaks from a pickle file
2. Accesses corresponding videos from local filesystem
3. Extracts 2-second snippets around each peak
4. Marks frames ±10 from the peak with red borders
5. Saves snippets to video_snippets directory
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import tempfile
import shutil
import subprocess
from typing import Optional, Tuple


# Configuration
OUTPUT_DIR = Path("/home/katya.ivantsiv/streaming/video_snippets")
VIDEO_BASE_PATH = Path("/mnt/A3000/Recordings/v2_data")
VIDEO_FPS = 30  # Video frame rate
SNIPPET_DURATION = 2.0  # seconds
RED_BORDER_FRAMES = 10  # Number of frames before and after peak to mark
BORDER_WIDTH = 10  # Width of red border in pixels


def get_video_path(run_path: str) -> Optional[Path]:
    """
    Get the local path to the video file.
    
    Args:
        run_path: Path to the run (e.g., '2024/11/24/VialApricot-164244/158_0_..._loud/')
        
    Returns:
        Path to video file if it exists, None otherwise
    """
    # Remove trailing slash if present
    run_path = run_path.rstrip('/')
    
    # Construct local path using the full run_path
    video_path = VIDEO_BASE_PATH / run_path / "video_full.mp4"
    
    if video_path.exists():
        return video_path
    else:
        print(f"  ✗ Video not found: {video_path}")
        return None


def add_red_border(frame: np.ndarray, border_width: int = BORDER_WIDTH) -> np.ndarray:
    """
    Add a red border around the frame.
    
    Args:
        frame: Input frame (BGR format)
        border_width: Width of the border in pixels
        
    Returns:
        Frame with red border
    """
    bordered_frame = frame.copy()
    h, w = bordered_frame.shape[:2]
    
    # Top border
    bordered_frame[:border_width, :] = [0, 0, 255]  # BGR: Red
    # Bottom border
    bordered_frame[h-border_width:, :] = [0, 0, 255]
    # Left border
    bordered_frame[:, :border_width] = [0, 0, 255]
    # Right border
    bordered_frame[:, w-border_width:] = [0, 0, 255]
    
    return bordered_frame


def extract_snippet(video_path: Path, 
                   peak_frame: int, 
                   output_path: Path,
                   fps: float = VIDEO_FPS,
                   duration: float = SNIPPET_DURATION,
                   red_border_frames: int = RED_BORDER_FRAMES) -> bool:
    """
    Extract a video snippet around a peak frame and mark frames near the peak.
    
    Args:
        video_path: Path to the full video
        peak_frame: Frame number of the peak
        output_path: Where to save the snippet
        fps: Video frame rate
        duration: Duration of snippet in seconds
        red_border_frames: Number of frames before/after peak to mark with red border
        
    Returns:
        True if successful, False otherwise
    """
    temp_path = None
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"  ✗ Could not open video: {video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use video's actual FPS if available
        if video_fps > 0:
            fps = video_fps
        
        # Calculate frame range for snippet (centered on peak)
        frames_in_snippet = int(duration * fps)
        start_frame = max(0, peak_frame - frames_in_snippet // 2)
        end_frame = min(total_frames, start_frame + frames_in_snippet)
        
        # Adjust start if we hit the end
        if end_frame - start_frame < frames_in_snippet:
            start_frame = max(0, end_frame - frames_in_snippet)
        
        # Define red border range
        red_start = peak_frame - red_border_frames
        red_end = peak_frame + red_border_frames
        
        # Create temporary file for raw video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Set up video writer for temporary file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"  ✗ Could not create temporary video")
            cap.release()
            return False
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add red border if within range
            if red_start <= frame_idx <= red_end:
                frame = add_red_border(frame, BORDER_WIDTH)
            
            out.write(frame)
        
        # Clean up video resources
        cap.release()
        out.release()
        
        # Re-encode with ffmpeg for browser compatibility (H.264 + AAC)
        try:
            cmd = [
                'ffmpeg',
                '-i', temp_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-y',
                str(output_path)
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Remove temporary file
            Path(temp_path).unlink()
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error re-encoding video: {e}")
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()
            return False
        
    except Exception as e:
        print(f"  ✗ Error extracting snippet: {e}")
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
        return False


def process_unmatched_peaks(pkl_file: str, 
                            max_videos: Optional[int] = None,
                            skip_existing: bool = True) -> None:
    """
    Process all unmatched peaks: download videos and create snippets.
    
    Args:
        pkl_file: Path to the pickle file with unmatched peaks
        max_videos: Maximum number of videos to process (None for all)
        skip_existing: If True, skip videos that already exist
    """
    # Load unmatched peaks
    print("="*80)
    print("LOADING UNMATCHED PEAKS")
    print("="*80)
    
    df = pd.read_pickle(pkl_file)
    print(f"Loaded {len(df)} unmatched peaks")
    print(f"Threshold used: {df['threshold'].iloc[0]:.4f}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Group by file (run_path, tar_id, side)
    grouped = df.groupby(['run_path', 'tar_id', 'side'])
    print(f"\nNumber of unique files: {len(grouped)}")
    
    # Process each file
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    print("\n" + "="*80)
    print("PROCESSING VIDEOS")
    print("="*80)
    
    for (run_path, tar_id, side), group in tqdm(grouped, desc="Processing files"):
        if max_videos is not None and processed_count >= max_videos:
            print(f"\nReached maximum of {max_videos} videos. Stopping.")
            break
        
        run_name = Path(run_path).name
        print(f"\n{'='*80}")
        print(f"File: {run_name} (tar_id: {tar_id}, side: {side})")
        print(f"Number of peaks in this file: {len(group)}")
        print(f"{'='*80}")
        
        # Get local video path
        video_path = get_video_path(run_path)
        if video_path is None:
            print(f"  ✗ Skipping file - video not found")
            failed_count += 1
            continue
        
        print(f"  Using video: {video_path}")
        
        # Process each peak in this file
        for idx, row in group.iterrows():
            # Use video_frame_in_file (30 fps) instead of peak_frame_in_file (25 fps)
            peak_frame_25fps = row['peak_frame_in_file']
            peak_frame_30fps = row['video_frame_in_file']
            peak_value = row['peak_value']
            
            # Create output filename (show both frame numbers for reference)
            output_filename = f"{run_name}_tar{tar_id}_{side}_frame{peak_frame_30fps}_25fps{peak_frame_25fps}_peak{peak_value:.3f}.mp4"
            output_path = OUTPUT_DIR / output_filename
            
            # Skip if already exists
            if skip_existing and output_path.exists():
                print(f"  ⊙ Skipping existing: {output_filename}")
                skipped_count += 1
                continue
            
            print(f"  Processing peak at frame {peak_frame_30fps} (30fps) / {peak_frame_25fps} (25fps) (value: {peak_value:.3f})")
            
            # Extract snippet using 30 fps frame number
            if extract_snippet(video_path, peak_frame_30fps, output_path):
                print(f"  ✓ Saved: {output_filename}")
                processed_count += 1
            else:
                print(f"  ✗ Failed to create snippet")
                failed_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successfully processed: {processed_count} snippets")
    print(f"Skipped (already exist): {skipped_count} snippets")
    print(f"Failed: {failed_count} snippets")
    print(f"Total peaks in dataset: {len(df)}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process video snippets for unmatched predicted peaks from local filesystem"
    )
    parser.add_argument(
        "pkl_file",
        type=str,
        help="Path to pickle file with unmatched peaks"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process existing videos (default: skip existing)"
    )
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if not Path(args.pkl_file).exists():
        print(f"Error: Pickle file not found: {args.pkl_file}")
        sys.exit(1)
    
    # Process videos
    process_unmatched_peaks(
        args.pkl_file,
        max_videos=args.max_videos,
        skip_existing=not args.no_skip_existing
    )


if __name__ == "__main__":
    main()

