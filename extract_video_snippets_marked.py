#!/usr/bin/env python3
"""
Extract video snippets around unmatched prediction timestamps with visual markers.
Takes 2 seconds around each timestamp and adds a red border around frames near the peak.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import subprocess
import tempfile

# Configuration
# NOTE: Update this to match the prominence value from your analysis
# The file is now named with prominence instead of threshold (e.g., "unmatched_pred_peaks_prom0.5000.pkl")
PICKLE_FILE = "unmatched_pred_peaks_prom0.5000.pkl"  # Update with your prominence value
VIDEO_BASE_PATH = Path("/mnt/A3000/Recordings/v2_data")
OUTPUT_DIR = Path("video_snippets_marked")
SNIPPET_DURATION = 2.0  # seconds (1 second before and 1 second after)
FRAMES_TO_MARK = 20  # Mark 20 frames around the peak (±10 frames)
BORDER_THICKNESS = 15  # Thickness of red border in pixels
FPS = 30  # Frames per second

def add_red_border(frame, thickness=15):
    """Add a red border to a frame."""
    bordered_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw red border (BGR format, so red is (0, 0, 255))
    cv2.rectangle(bordered_frame, (0, 0), (w-1, h-1), (0, 0, 255), thickness)
    
    return bordered_frame

def extract_and_mark_video_snippet(video_path, timestamp_seconds, local_frame, output_path, 
                                   duration=2.0, frames_to_mark=20, fps=30):
    """
    Extract a video snippet and add red border to frames around the peak.
    
    Args:
        video_path: Path to the source video
        timestamp_seconds: Center timestamp in seconds
        local_frame: The local frame number where the peak occurs (in BLENDSHAPE space at 25fps)
        output_path: Path to save the output snippet
        duration: Total duration of the snippet (centered on timestamp)
        frames_to_mark: Number of frames to mark around the peak
        fps: Frames per second (video fps, typically 30)
    
    Returns:
        True if successful, False otherwise
    """
    # Calculate start time and frame
    start_time = max(0, timestamp_seconds - duration / 2)
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    # IMPORTANT: local_frame is in blendshape space (25 fps), but we need video space (30 fps)
    # Convert: video_frame = timestamp * video_fps
    peak_frame_video = int(timestamp_seconds * fps)
    
    # Calculate which frames to mark (±frames_to_mark/2 around peak)
    peak_frame_in_snippet = peak_frame_video - start_frame
    mark_start = peak_frame_in_snippet - frames_to_mark // 2
    mark_end = peak_frame_in_snippet + frames_to_mark // 2
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = fps
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary file for raw video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, original_fps, (width, height))
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    frame_count = 0
    while frame_count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add red border if this frame is within the marking range
        if mark_start <= frame_count <= mark_end:
            frame = add_red_border(frame, BORDER_THICKNESS)
        
        out.write(frame)
        frame_count += 1
    
    # Release resources
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
        print(f"Error re-encoding video: {e}")
        Path(temp_path).unlink(missing_ok=True)
        return False

def sanitize_filename(text):
    """Remove or replace characters that are problematic in filenames."""
    text = text.replace('/', '_')
    text = text.replace('\\', '_')
    return text

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("VIDEO SNIPPET EXTRACTION WITH MARKERS")
    print("="*80)
    
    # Load the pickle file
    print(f"\nLoading data from {PICKLE_FILE}...")
    df = pd.read_pickle(PICKLE_FILE)
    print(f"Loaded {len(df)} unmatched predictions")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Marking {FRAMES_TO_MARK} frames around each peak with red border")
    
    # Process each row
    successful = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*80)
    print("EXTRACTING AND MARKING VIDEO SNIPPETS")
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
        sanitized_run_path = sanitize_filename(run_path)
        output_filename = f"{global_frame}_{file_index}_{local_frame}_{timestamp_seconds:.2f}_{sanitized_run_path}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Extract and mark snippet
        success = extract_and_mark_video_snippet(
            video_path,
            timestamp_seconds,
            local_frame,
            output_path,
            duration=SNIPPET_DURATION,
            frames_to_mark=FRAMES_TO_MARK,
            fps=FPS
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
    print(f"Frames with red border: {FRAMES_TO_MARK} frames around each peak")
    print("="*80)

if __name__ == "__main__":
    main()

