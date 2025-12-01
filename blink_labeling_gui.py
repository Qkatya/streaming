#!/usr/bin/env python3
"""
GUI application for labeling video snippets as blink or no-blink.
Loads videos from video_snippets folder and allows user to classify them.
"""

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import threading
import time

# Configuration
PICKLE_FILE = "unmatched_pred_peaks_th0.007071.pkl"
VIDEO_SNIPPETS_DIR = Path("video_snippets")
OUTPUT_PICKLE = "unmatched_pred_peaks_labeled.pkl"

class BlinkLabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blink Labeling Tool")
        self.root.geometry("900x700")
        
        # Load data
        self.df = pd.read_pickle(PICKLE_FILE)
        
        # Add 'is_blink' column if it doesn't exist
        if 'is_blink' not in self.df.columns:
            self.df['is_blink'] = None
        
        # Get list of video files
        self.video_files = self.get_video_files()
        self.current_index = 0
        
        # Video playback variables
        self.cap = None
        self.playing = False
        self.video_thread = None
        self.stop_playback = False
        
        # Create GUI elements
        self.create_widgets()
        
        # Load first video
        if len(self.video_files) > 0:
            self.load_video(self.current_index)
        else:
            messagebox.showerror("Error", "No video files found in video_snippets folder!")
            self.root.quit()
    
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
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Top frame for info
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(fill=tk.X)
        
        # Progress label
        self.progress_label = ttk.Label(
            info_frame, 
            text=f"Video 1 of {len(self.video_files)}",
            font=("Arial", 14, "bold")
        )
        self.progress_label.pack()
        
        # Info labels
        self.info_label = ttk.Label(
            info_frame,
            text="",
            font=("Arial", 10),
            justify=tk.LEFT
        )
        self.info_label.pack(pady=5)
        
        # Video display frame
        video_frame = ttk.Frame(self.root, padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for video
        self.canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)
        self.canvas.pack()
        
        # Control buttons frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Play/Replay button
        self.play_button = ttk.Button(
            control_frame,
            text="▶ Play / Replay",
            command=self.play_video,
            width=20
        )
        self.play_button.pack(pady=10)
        
        # Labeling buttons frame
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        # Create two large buttons side by side
        self.blink_button = tk.Button(
            button_frame,
            text="✓ WAS BLINK",
            command=lambda: self.label_video(True),
            bg="#4CAF50",
            fg="white",
            font=("Arial", 16, "bold"),
            height=2,
            width=20
        )
        self.blink_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        self.no_blink_button = tk.Button(
            button_frame,
            text="✗ NO BLINK",
            command=lambda: self.label_video(False),
            bg="#f44336",
            fg="white",
            font=("Arial", 16, "bold"),
            height=2,
            width=20
        )
        self.no_blink_button.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        # Navigation frame
        nav_frame = ttk.Frame(self.root, padding="10")
        nav_frame.pack(fill=tk.X)
        
        # Previous button
        self.prev_button = ttk.Button(
            nav_frame,
            text="← Previous",
            command=self.previous_video,
            width=15
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        # Skip button
        self.skip_button = ttk.Button(
            nav_frame,
            text="Skip →",
            command=self.skip_video,
            width=15
        )
        self.skip_button.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_button = ttk.Button(
            nav_frame,
            text="💾 Save Progress",
            command=self.save_progress,
            width=15
        )
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Keyboard bindings
        self.root.bind('b', lambda e: self.label_video(True))
        self.root.bind('n', lambda e: self.label_video(False))
        self.root.bind('<Left>', lambda e: self.previous_video())
        self.root.bind('<Right>', lambda e: self.skip_video())
        self.root.bind('<space>', lambda e: self.play_video())
    
    def load_video(self, index):
        """Load video at given index."""
        if index < 0 or index >= len(self.video_files):
            return
        
        # Stop current playback
        self.stop_current_playback()
        
        self.current_index = index
        video_info = self.video_files[index]
        
        # Update progress label
        self.progress_label.config(
            text=f"Video {index + 1} of {len(self.video_files)}"
        )
        
        # Update info label
        info = video_info['info']
        current_label = self.df.loc[video_info['df_index'], 'is_blink']
        label_text = "Not labeled yet"
        if current_label is not None:
            label_text = "BLINK" if current_label else "NO BLINK"
        
        info_text = (
            f"Global Frame: {info['global_frame']} | "
            f"File Index: {info['file_index']} | "
            f"Local Frame: {info['local_frame']} | "
            f"Timestamp: {info['timestamp_seconds']:.2f}s\n"
            f"Run Path: {info['run_path']}\n"
            f"Current Label: {label_text}"
        )
        self.info_label.config(text=info_text)
        
        # Load video
        self.cap = cv2.VideoCapture(str(video_info['path']))
        
        # Display first frame
        self.display_first_frame()
        
        # Auto-play the video
        self.play_video()
    
    def display_first_frame(self):
        """Display the first frame of the video."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
    
    def display_frame(self, frame):
        """Display a frame on the canvas."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling to fit canvas while maintaining aspect ratio
            frame_height, frame_width = frame_rgb.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        else:
            frame_resized = frame_rgb
        
        # Convert to PIL Image and then to ImageTk
        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo,
            anchor=tk.CENTER
        )
    
    def play_video(self):
        """Play the current video in a loop."""
        if self.playing:
            return
        
        self.stop_playback = False
        self.playing = True
        self.video_thread = threading.Thread(target=self._play_video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def _play_video_loop(self):
        """Internal method to play video in a loop."""
        if self.cap is None:
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        delay = 1.0 / fps
        
        while not self.stop_playback:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            while not self.stop_playback:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.display_frame(frame)
                time.sleep(delay)
        
        self.playing = False
    
    def stop_current_playback(self):
        """Stop current video playback."""
        self.stop_playback = True
        if self.video_thread is not None:
            self.video_thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def label_video(self, is_blink):
        """Label current video and move to next."""
        if self.current_index >= len(self.video_files):
            return
        
        # Save label
        video_info = self.video_files[self.current_index]
        df_index = video_info['df_index']
        self.df.loc[df_index, 'is_blink'] = is_blink
        
        # Move to next video
        self.next_video()
    
    def next_video(self):
        """Load next video."""
        if self.current_index < len(self.video_files) - 1:
            self.load_video(self.current_index + 1)
        else:
            # All videos labeled
            messagebox.showinfo(
                "Complete",
                f"All {len(self.video_files)} videos have been reviewed!\n"
                "Don't forget to save your progress."
            )
    
    def previous_video(self):
        """Load previous video."""
        if self.current_index > 0:
            self.load_video(self.current_index - 1)
    
    def skip_video(self):
        """Skip current video without labeling."""
        self.next_video()
    
    def save_progress(self):
        """Save current progress to pickle file."""
        try:
            self.df.to_pickle(OUTPUT_PICKLE)
            
            # Count labeled videos
            labeled_count = self.df['is_blink'].notna().sum()
            
            messagebox.showinfo(
                "Saved",
                f"Progress saved to {OUTPUT_PICKLE}\n"
                f"Labeled: {labeled_count} / {len(self.df)}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def on_closing(self):
        """Handle window closing."""
        # Ask to save before closing
        labeled_count = self.df['is_blink'].notna().sum()
        if labeled_count > 0:
            if messagebox.askyesno("Save Progress", "Do you want to save your progress before closing?"):
                self.save_progress()
        
        self.stop_current_playback()
        self.root.destroy()

def main():
    """Main execution function."""
    root = tk.Tk()
    app = BlinkLabelingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()


