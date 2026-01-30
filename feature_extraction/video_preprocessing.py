"""
Video Preprocessing Module
Handles video loading, frame extraction, and preprocessing for I3D feature extraction.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Generator
import warnings


def check_opencv():
    """Check if OpenCV is available."""
    try:
        import cv2
        return True
    except ImportError:
        return False


def load_video_frames(video_path: str, 
                      target_fps: int = 16,
                      resize: Tuple[int, int] = (224, 224),
                      max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load video and extract frames.
    
    Args:
        video_path: Path to video file
        target_fps: Target frames per second (default 16 for I3D)
        resize: Target frame size (H, W)
        max_frames: Maximum number of frames to extract (None for all)
    
    Returns:
        frames: numpy array of shape (N, H, W, C) with values in [0, 255]
    """
    import cv2
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if original_fps <= 0:
        original_fps = 30  # Default assumption
        warnings.warn(f"Could not get FPS from video, assuming {original_fps}")
    
    # Calculate frame sampling rate
    frame_interval = max(1, int(original_fps / target_fps))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at target FPS
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            if resize is not None:
                frame = cv2.resize(frame, (resize[1], resize[0]))
            
            frames.append(frame)
            
            if max_frames is not None and len(frames) >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    return np.array(frames, dtype=np.uint8)


def preprocess_frames_for_i3d(frames: np.ndarray, 
                               normalize: bool = True) -> np.ndarray:
    """
    Preprocess frames for I3D model.
    
    Args:
        frames: numpy array of shape (N, H, W, C) with values in [0, 255]
        normalize: Whether to normalize to [-1, 1]
    
    Returns:
        preprocessed: numpy array of shape (N, H, W, C) with values in [-1, 1]
    """
    frames = frames.astype(np.float32)
    
    if normalize:
        # Normalize to [-1, 1] (I3D expects this range)
        frames = (frames / 255.0) * 2 - 1
    
    return frames


def create_clips(frames: np.ndarray, 
                 clip_length: int = 16,
                 stride: int = 16) -> np.ndarray:
    """
    Create video clips from frames.
    
    Args:
        frames: numpy array of shape (N, H, W, C)
        clip_length: Number of frames per clip (16 for I3D)
        stride: Stride between clips
    
    Returns:
        clips: numpy array of shape (num_clips, clip_length, H, W, C)
    """
    num_frames = len(frames)
    
    if num_frames < clip_length:
        # Pad with last frame if video is too short
        padding = clip_length - num_frames
        pad_frames = np.tile(frames[-1:], (padding, 1, 1, 1))
        frames = np.concatenate([frames, pad_frames], axis=0)
        num_frames = clip_length
    
    clips = []
    for start in range(0, num_frames - clip_length + 1, stride):
        clip = frames[start:start + clip_length]
        clips.append(clip)
    
    # Ensure at least one clip
    if len(clips) == 0:
        clips.append(frames[:clip_length])
    
    return np.array(clips)


def segment_features(features: np.ndarray, 
                     num_segments: int = 32) -> np.ndarray:
    """
    Segment features into fixed number of segments using average pooling.
    
    Args:
        features: numpy array of shape (N, D) where N is number of clips
        num_segments: Target number of segments
    
    Returns:
        segmented: numpy array of shape (num_segments, D)
    """
    num_clips, feature_dim = features.shape
    
    if num_clips == num_segments:
        return features
    
    if num_clips < num_segments:
        # Upsample by repeating
        indices = np.linspace(0, num_clips - 1, num_segments).astype(int)
        return features[indices]
    
    # Downsample by averaging
    segment_size = num_clips / num_segments
    segmented = np.zeros((num_segments, feature_dim), dtype=np.float32)
    
    for i in range(num_segments):
        start = int(i * segment_size)
        end = int((i + 1) * segment_size)
        segmented[i] = features[start:end].mean(axis=0)
    
    return segmented


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
    
    Returns:
        info: Dictionary with video properties
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


class VideoIterator:
    """
    Memory-efficient video iterator that yields clips.
    """
    
    def __init__(self, video_path: str, 
                 clip_length: int = 16,
                 stride: int = 16,
                 resize: Tuple[int, int] = (224, 224),
                 target_fps: int = 16):
        """
        Initialize video iterator.
        
        Args:
            video_path: Path to video file
            clip_length: Frames per clip
            stride: Stride between clips
            resize: Target frame size
            target_fps: Target FPS for sampling
        """
        import cv2
        
        self.video_path = video_path
        self.clip_length = clip_length
        self.stride = stride
        self.resize = resize
        self.target_fps = target_fps
        
        # Open video to get properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval = max(1, int(self.fps / target_fps))
        
        # Estimate number of clips
        sampled_frames = self.total_frames // self.frame_interval
        self.num_clips = max(1, (sampled_frames - clip_length) // stride + 1)
    
    def __len__(self):
        return self.num_clips
    
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Yield clips one at a time."""
        import cv2
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_buffer = []
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.resize:
                    frame = cv2.resize(frame, (self.resize[1], self.resize[0]))
                
                # Normalize to [-1, 1]
                frame = (frame.astype(np.float32) / 255.0) * 2 - 1
                frame_buffer.append(frame)
                
                # Yield clip when buffer is full
                while len(frame_buffer) >= self.clip_length:
                    clip = np.array(frame_buffer[:self.clip_length])
                    yield clip
                    
                    # Remove frames based on stride
                    frame_buffer = frame_buffer[self.stride:]
            
            frame_idx += 1
        
        # Handle remaining frames (pad if necessary)
        if len(frame_buffer) > 0 and len(frame_buffer) < self.clip_length:
            padding = self.clip_length - len(frame_buffer)
            pad_frames = [frame_buffer[-1]] * padding
            frame_buffer.extend(pad_frames)
            yield np.array(frame_buffer[:self.clip_length])
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
