#!/usr/bin/env python3
"""
Simple GUI application for labeling video snippets as blink or no-blink using OpenCV.
Loads videos from video_snippets folder and allows user to classify them.
"""

import pandas as pd
from pathlib import Path
import cv2
import numpy as np

# Configuration
PICKLE_FILE = "unmatched_pred_peaks_th0.007071.pkl"
VIDEO_SNIPPETS_DIR = Path("video_snippets")
OUTPUT_PICKLE = "unmatched_pred_peaks_labeled.pkl"

class BlinkLabelingApp:
    def __init__(self):
        # Load data
        self.df = pd.read_pickle(PICKLE_FILE)
        
        # Add 'is_blink' column if it doesn't exist
        if 'is_blink' not in self.df.columns:
            self.df['is_blink'] = None
        
        # Get list of video files
        self.video_files = self.get_video_files()
        self.current_index = 0
        
        # Window name
        self.window_name = "Blink Labeling Tool"
        
        print("\n" + "="*80)
        print("BLINK LABELING TOOL")
        print("="*80)
        print(f"Total videos to label: {len(self.video_files)}")
        print("\nControls:")
        print("  B key = WAS BLINK (mark as true blink)")
        print("  N key = NO BLINK (mark as false positive)")
        print("  <- key = Previous video")
        print("  -> key = Next video (skip)")
        print("  S key = Save progress")
        print("  Q key = Quit (will prompt to save)")
        print("="*80 + "\n")
    
    def get_video_files(self):
        """Get list of video files and match them to dataframe rows."""
        video_files = []
        
        for idx, row in self.df.iterrows():
            # Construct expected filename
            global_frame = row['global_frame']
            file_index = row['file_index']
            local_frame = row['local_frame']
            timestamp_seconds = row['timestamp_seconds']
            run_path = row['run_path'].replace('/', '_').replace('\\', '_')
            
            filename = f"{global_frame}_{file_index}_{local_frame}_{timestamp_seconds:.2f}_{run_path}.mp4"
            video_path = VIDEO_SNIPPETS_DIR / filename
            
            if video_path.exists():
                video_files.append({
                    'path': video_path,
                    'df_index': idx,
                    'info': row.to_dict()
                })
        
        return video_files
    
    def create_info_frame(self, video_info, frame_shape):
        """Create an info overlay frame."""
        height, width = frame_shape[:2]
        info_height = 200
        info_frame = np.zeros((info_height, width, 3), dtype=np.uint8)
        
        # Get current label
        current_label = self.df.loc[video_info['df_index'], 'is_blink']
        if current_label is None:
            label_text = "Not labeled"
            label_color = (128, 128, 128)
        elif current_label:
            label_text = "BLINK"
            label_color = (0, 255, 0)
        else:
            label_text = "NO BLINK"
            label_color = (0, 0, 255)
        
        # Progress
        progress_text = f"Video {self.current_index + 1} / {len(self.video_files)}"
        cv2.putText(info_frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current label
        cv2.putText(info_frame, f"Current: {label_text}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        
        # Info
        info = video_info['info']
        cv2.putText(info_frame, f"Global Frame: {info['global_frame']}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_frame, f"Local Frame: {info['local_frame']} | Time: {info['timestamp_seconds']:.2f}s", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Run path (truncated if too long)
        run_path = info['run_path']
        if len(run_path) > 70:
            run_path = "..." + run_path[-67:]
        cv2.putText(info_frame, f"Path: {run_path}", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(info_frame, "Press: [B]=Blink [N]=No Blink [<-]=Prev [->]=Next [S]=Save [Q]=Quit", 
                   (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
        
        return info_frame
    
    def play_video(self, video_path, video_info):
        """Play video in a loop until user makes a choice."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        delay = int(1000 / fps)  # milliseconds
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            print(f"Error: No frames in video {video_path}")
            return None
        
        # Play video in loop
        frame_idx = 0
        action = None
        
        while action is None:
            frame = frames[frame_idx]
            
            # Create info overlay
            info_frame = self.create_info_frame(video_info, frame.shape)
            
            # Combine video frame and info frame
            display_frame = np.vstack([frame, info_frame])
            
            # Resize if too large
            max_height = 900
            if display_frame.shape[0] > max_height:
                scale = max_height / display_frame.shape[0]
                new_width = int(display_frame.shape[1] * scale)
                new_height = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            cv2.imshow(self.window_name, display_frame)
            
            # Wait for key press
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('b') or key == ord('B'):
                action = 'blink'
            elif key == ord('n') or key == ord('N'):
                action = 'no_blink'
            elif key == 81 or key == 2:  # Left arrow
                action = 'prev'
            elif key == 83 or key == 3:  # Right arrow
                action = 'next'
            elif key == ord('s') or key == ord('S'):
                action = 'save'
            elif key == ord('q') or key == ord('Q'):
                action = 'quit'
            
            # Loop video
            frame_idx = (frame_idx + 1) % len(frames)
        
        return action
    
    def label_video(self, is_blink):
        """Label current video."""
        if self.current_index >= len(self.video_files):
            return
        
        video_info = self.video_files[self.current_index]
        df_index = video_info['df_index']
        self.df.loc[df_index, 'is_blink'] = is_blink
        
        label_text = "BLINK" if is_blink else "NO BLINK"
        print(f"Video {self.current_index + 1}: Labeled as {label_text}")
    
    def save_progress(self):
        """Save current progress to pickle file."""
        try:
            self.df.to_pickle(OUTPUT_PICKLE)
            labeled_count = self.df['is_blink'].notna().sum()
            print(f"\n{'='*80}")
            print(f"Progress saved to {OUTPUT_PICKLE}")
            print(f"Labeled: {labeled_count} / {len(self.df)}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error saving: {e}")
    
    def run(self):
        """Main loop."""
        if len(self.video_files) == 0:
            print("Error: No video files found!")
            return
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while self.current_index < len(self.video_files):
            video_info = self.video_files[self.current_index]
            
            print(f"\nShowing video {self.current_index + 1} / {len(self.video_files)}")
            
            action = self.play_video(video_info['path'], video_info)
            
            if action == 'blink':
                self.label_video(True)
                self.current_index += 1
            elif action == 'no_blink':
                self.label_video(False)
                self.current_index += 1
            elif action == 'prev':
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("Already at first video")
            elif action == 'next':
                self.current_index += 1
            elif action == 'save':
                self.save_progress()
            elif action == 'quit':
                break
        
        # End of videos or quit
        cv2.destroyAllWindows()
        
        labeled_count = self.df['is_blink'].notna().sum()
        
        if labeled_count > 0:
            print(f"\n{'='*80}")
            response = input(f"You have {labeled_count} labeled videos. Save progress? (y/n): ")
            if response.lower() == 'y':
                self.save_progress()
        
        print("\n" + "="*80)
        print("LABELING SESSION COMPLETE")
        print(f"Total labeled: {labeled_count} / {len(self.df)}")
        print("="*80)

def main():
    """Main execution function."""
    app = BlinkLabelingApp()
    app.run()

if __name__ == "__main__":
    main()


