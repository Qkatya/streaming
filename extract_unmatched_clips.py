#!/usr/bin/env python3
"""
Script to extract video clips around unmatched prediction peaks.
Loads timestamps from pickle file, downloads videos, and creates clips
with marked frames (red squares) 10 frames before and after the peak.
"""

import pickle
import subprocess
import os
import sys
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd

# Global variable for video base path
VIDEO_BASE_PATH = "/mnt/A3000/Recordings/v2_data"


def load_pickle_data(pickle_path):
    """Load the unmatched peaks data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_video_path(run_path):
    """Construct the full video path from run_path."""
    video_path = Path(VIDEO_BASE_PATH) / run_path / "video_full.mp4"
    return str(video_path)


def copy_video_if_needed(video_path, output_path):
    """Copy video from local path (or just return path if same location)."""
    print(f"Using video from: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False, None
    
    # Check if file has content
    if os.path.getsize(video_path) < 1000:
        print(f"Error: Video file is too small: {video_path}")
        return False, None
    
    # Return the video path directly (no need to copy)
    return True, video_path


def mark_frames_with_red_border(video_path, output_path, frame_indices, fps=30):
    """
    Mark specific frames with red border using OpenCV.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        frame_indices: List of frame indices to mark with red border
        fps: Frames per second of the video
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use original fps if available, otherwise use provided fps
    if original_fps > 0:
        fps = original_fps
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    frame_indices_set = set(frame_indices)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add red border to marked frames
        if frame_idx in frame_indices_set:
            border_thickness = 10
            cv2.rectangle(
                frame,
                (border_thickness, border_thickness),
                (width - border_thickness, height - border_thickness),
                (0, 0, 255),  # Red color in BGR
                border_thickness
            )
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return True


def extract_clip_with_marked_frames(video_path, output_path, center_frame, fps=30, 
                                    frames_before=10, frames_after=10,
                                    window_seconds_before=1, window_seconds_after=1):
    """
    Extract a clip around center_frame with marked frames before and after.
    
    Args:
        video_path: Path to input video
        output_path: Path to output clip
        center_frame: The center frame number (peak timestamp)
        fps: Frames per second
        frames_before: Number of frames before center to mark with red
        frames_after: Number of frames after center to mark with red
        window_seconds_before: Seconds before center frame to include in clip
        window_seconds_after: Seconds after center frame to include in clip
    """
    # Calculate the clip window
    clip_start_frame = max(0, center_frame - int(window_seconds_before * fps))
    clip_end_frame = center_frame + int(window_seconds_after * fps)
    
    # Calculate which frames to mark (relative to clip start)
    mark_start_frame = max(clip_start_frame, center_frame - frames_before)
    mark_end_frame = center_frame + frames_after
    
    # Frames to mark (absolute frame numbers)
    frames_to_mark = list(range(mark_start_frame, center_frame)) + \
                     list(range(center_frame + 1, mark_end_frame + 1))
    
    # First, extract the clip without marking - use re-encoding for robustness
    start_time = clip_start_frame / fps
    duration = (clip_end_frame - clip_start_frame) / fps
    
    temp_clip = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    try:
        # Extract clip using ffmpeg with re-encoding (more robust than stream copy)
        # Use -accurate_seek for better seeking accuracy
        cmd = [
            'ffmpeg', '-y',
            '-accurate_seek',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            temp_clip
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, stderr=subprocess.PIPE)
        
        # Check if temp clip was created and has content
        if not os.path.exists(temp_clip) or os.path.getsize(temp_clip) == 0:
            print(f"Error: Extracted clip is empty or doesn't exist")
            return False
        
        # Now mark the frames in the extracted clip
        # Adjust frame indices to be relative to the clip
        frames_to_mark_relative = [f - clip_start_frame for f in frames_to_mark]
        
        success = mark_frames_with_red_border(temp_clip, output_path, frames_to_mark_relative, fps)
        
        return success
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clip: {e}")
        if e.stderr:
            stderr_text = e.stderr.decode()
            # Only print last few lines of stderr to avoid clutter
            stderr_lines = stderr_text.strip().split('\n')
            print(f"stderr (last 5 lines): {chr(10).join(stderr_lines[-5:])}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(temp_clip):
            os.remove(temp_clip)


def process_unmatched_peaks(pickle_path, output_dir='unmatched_clips', fps=30):
    """
    Main processing function.
    
    Args:
        pickle_path: Path to the pickle file with unmatched peaks
        output_dir: Directory to save output clips
        fps: Frames per second (default 30)
    """
    # Load data
    print(f"Loading data from {pickle_path}...")
    data = load_pickle_data(pickle_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Data structure: {type(data)}")
    
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        print(f"DataFrame with {len(data)} rows")
        print(f"Columns: {data.columns.tolist()}")
        
        # Process each row
        for idx, row in data.iterrows():
            print(f"\n{'='*60}")
            print(f"Processing row {idx + 1}/{len(data)}")
            
            # Extract run_path and frame
            run_path = row.get('run_path')
            frame = row.get('global_frame') or row.get('frame') or row.get('timestamp')
            
            if pd.isna(run_path) or pd.isna(frame):
                print(f"Skipping row {idx} - missing run_path or frame")
                continue
            
            # Convert frame to int if needed
            frame = int(frame)
            
            print(f"Run path: {run_path}")
            print(f"Frame: {frame}")
            
            # Get local video path
            video_path = get_video_path(run_path)
            
            try:
                # Check if video exists
                success, actual_video_path = copy_video_if_needed(video_path, None)
                if not success:
                    print(f"Failed to access video for {run_path}")
                    continue
                
                # Generate output filename
                safe_run_path = run_path.replace('/', '_').replace('\\', '_')
                output_filename = f"{safe_run_path}_frame{frame}.mp4"
                output_file = output_path / output_filename
                
                print(f"Extracting clip to: {output_file}")
                
                # Extract clip with marked frames
                success = extract_clip_with_marked_frames(
                    actual_video_path,
                    str(output_file),
                    center_frame=frame,
                    fps=fps,
                    frames_before=10,
                    frames_after=10,
                    window_seconds_before=1,
                    window_seconds_after=1
                )
                
                if success:
                    print(f"Successfully created clip: {output_file}")
                else:
                    print(f"Failed to create clip for {run_path}")
            
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Handle other data structures (dict, list)
    elif isinstance(data, dict):
        print(f"Dictionary keys: {data.keys()}")
        items = data.items()
        
        for idx, item in items:
            print(f"\n{'='*60}")
            print(f"Processing item {idx}")
            print(f"Item structure: {item}")
            
            if isinstance(item, dict):
                run_path = item.get('run_path')
                frame = item.get('frame') or item.get('timestamp') or item.get('peak_frame')
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                run_path = item[0]
                frame = item[1]
            else:
                print(f"Skipping item with unexpected structure: {item}")
                continue
            
            if run_path is None or frame is None:
                print(f"Skipping item - missing run_path or frame: {item}")
                continue
            
            print(f"Run path: {run_path}")
            print(f"Frame: {frame}")
            
            process_single_item(run_path, frame, output_path, fps)
    
    elif isinstance(data, list):
        print(f"List with {len(data)} items")
        
        for idx, item in enumerate(data):
            print(f"\n{'='*60}")
            print(f"Processing item {idx + 1}/{len(data)}")
            
            if isinstance(item, dict):
                run_path = item.get('run_path')
                frame = item.get('frame') or item.get('timestamp') or item.get('peak_frame')
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                run_path = item[0]
                frame = item[1]
            else:
                print(f"Skipping item with unexpected structure: {item}")
                continue
            
            if run_path is None or frame is None:
                print(f"Skipping item - missing run_path or frame: {item}")
                continue
            
            print(f"Run path: {run_path}")
            print(f"Frame: {frame}")
            
            process_single_item(run_path, frame, output_path, fps)
    
    else:
        print(f"Unsupported data type: {type(data)}")
        print(f"Data: {data}")
        return


def process_single_item(run_path, frame, output_path, fps):
    """Process a single item (run_path and frame)."""
    # Get local video path
    video_path = get_video_path(run_path)
    
    try:
        # Check if video exists
        success, actual_video_path = copy_video_if_needed(video_path, None)
        if not success:
            print(f"Failed to access video for {run_path}")
            return
        
        # Generate output filename
        safe_run_path = run_path.replace('/', '_').replace('\\', '_')
        output_filename = f"{safe_run_path}_frame{frame}.mp4"
        output_file = output_path / output_filename
        
        print(f"Extracting clip to: {output_file}")
        
        # Extract clip with marked frames
        success = extract_clip_with_marked_frames(
            actual_video_path,
            str(output_file),
            center_frame=frame,
            fps=fps,
            frames_before=10,
            frames_after=10,
            window_seconds_before=1,
            window_seconds_after=1
        )
        
        if success:
            print(f"Successfully created clip: {output_file}")
        else:
            print(f"Failed to create clip for {run_path}")
    
    except Exception as e:
        print(f"Error processing {run_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract video clips around unmatched prediction peaks'
    )
    parser.add_argument(
        'pickle_file',
        help='Path to pickle file with unmatched peaks'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='unmatched_clips',
        help='Output directory for clips (default: unmatched_clips)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    parser.add_argument(
        '--video-base-path',
        default='/mnt/A3000/Recordings/v2_data',
        help='Base path for video files (default: /mnt/A3000/Recordings/v2_data)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pickle_file):
        print(f"Error: Pickle file not found: {args.pickle_file}")
        sys.exit(1)
    
    # Update global base path if provided
    global VIDEO_BASE_PATH
    VIDEO_BASE_PATH = args.video_base_path
    
    process_unmatched_peaks(args.pickle_file, args.output_dir, args.fps)
    print("\nDone!")


if __name__ == '__main__':
    main()

