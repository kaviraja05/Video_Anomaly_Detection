"""
I3D Feature Extractor
Extracts I3D features from raw videos for anomaly detection.
"""

import os
import sys
import numpy as np
import torch
from typing import Optional, Union, List
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.i3d_model import InceptionI3d, load_i3d_model
from feature_extraction.video_preprocessing import (
    load_video_frames,
    preprocess_frames_for_i3d,
    create_clips,
    segment_features,
    get_video_info,
    VideoIterator,
    check_opencv
)


class I3DFeatureExtractor:
    """
    Extract I3D features from raw video files.
    
    This class handles the complete pipeline from raw video to 
    2048-D I3D features ready for the anomaly detection model.
    """
    
    # URL for pretrained I3D weights
    WEIGHTS_URL = "https://github.com/piergiaj/pytorch-i3d/releases/download/v1.0/rgb_imagenet.pt"
    
    def __init__(self, 
                 weights_path: Optional[str] = None,
                 device: Optional[str] = None,
                 num_segments: int = 32,
                 clip_length: int = 16,
                 stride: int = 16,
                 input_size: tuple = (224, 224),
                 target_fps: int = 16,
                 batch_size: int = 8):
        """
        Initialize the I3D feature extractor.
        
        Args:
            weights_path: Path to pretrained I3D weights. If None, will try to
                         download or use random initialization.
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            num_segments: Number of segments to divide video into (default 32)
            clip_length: Number of frames per clip (default 16)
            stride: Stride between clips (default 16, no overlap)
            input_size: Input frame size (H, W)
            target_fps: Target FPS for frame sampling
            batch_size: Batch size for processing clips
        """
        # Check OpenCV availability
        if not check_opencv():
            raise ImportError(
                "OpenCV is required for video processing. "
                "Install with: pip install opencv-python"
            )
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Store parameters
        self.num_segments = num_segments
        self.clip_length = clip_length
        self.stride = stride
        self.input_size = input_size
        self.target_fps = target_fps
        self.batch_size = batch_size
        
        # Load model
        self.model = self._load_model(weights_path)
        
        # Feature dimension (I3D Mixed_5c output)
        self.feature_dim = 1024
    
    def _load_model(self, weights_path: Optional[str]) -> InceptionI3d:
        """Load I3D model with weights."""
        model = InceptionI3d(
            num_classes=400,
            in_channels=3,
            extract_features=True
        )
        
        if weights_path is not None and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded I3D weights from: {weights_path}")
        else:
            print("Warning: No pretrained weights loaded. Using random initialization.")
            print("For best results, download pretrained weights from:")
            print(f"  {self.WEIGHTS_URL}")
            print("And specify the path with weights_path parameter.")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def extract_features(self, 
                         video_path: str,
                         return_raw: bool = False,
                         verbose: bool = True) -> np.ndarray:
        """
        Extract I3D features from a video file.
        
        Args:
            video_path: Path to video file (.mp4, .avi, etc.)
            return_raw: If True, return raw clip features without segmentation
            verbose: Whether to print progress
        
        Returns:
            features: numpy array of shape (num_segments, feature_dim) or
                     (num_clips, feature_dim) if return_raw=True
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if verbose:
            info = get_video_info(video_path)
            print(f"Processing: {os.path.basename(video_path)}")
            print(f"  Duration: {info['duration']:.1f}s, FPS: {info['fps']:.1f}, "
                  f"Resolution: {info['width']}x{info['height']}")
        
        # Load and preprocess frames
        if verbose:
            print("  Loading frames...")
        
        frames = load_video_frames(
            video_path,
            target_fps=self.target_fps,
            resize=self.input_size
        )
        
        if verbose:
            print(f"  Extracted {len(frames)} frames")
        
        # Preprocess for I3D
        frames = preprocess_frames_for_i3d(frames, normalize=True)
        
        # Create clips
        clips = create_clips(frames, self.clip_length, self.stride)
        
        if verbose:
            print(f"  Created {len(clips)} clips")
        
        # Extract features from clips
        clip_features = self._extract_clip_features(clips, verbose)
        
        if return_raw:
            return clip_features
        
        # Segment features to fixed number
        features = segment_features(clip_features, self.num_segments)
        
        if verbose:
            print(f"  Output shape: {features.shape}")
        
        return features
    
    def extract_features_streaming(self, 
                                   video_path: str,
                                   verbose: bool = True) -> np.ndarray:
        """
        Extract features using memory-efficient streaming.
        
        Useful for long videos that don't fit in memory.
        
        Args:
            video_path: Path to video file
            verbose: Whether to print progress
        
        Returns:
            features: numpy array of shape (num_segments, feature_dim)
        """
        if verbose:
            print(f"Processing (streaming): {os.path.basename(video_path)}")
        
        # Create video iterator
        video_iter = VideoIterator(
            video_path,
            clip_length=self.clip_length,
            stride=self.stride,
            resize=self.input_size,
            target_fps=self.target_fps
        )
        
        if verbose:
            print(f"  Estimated clips: {len(video_iter)}")
        
        # Process clips in batches
        all_features = []
        batch = []
        
        for clip in video_iter:
            batch.append(clip)
            
            if len(batch) >= self.batch_size:
                features = self._process_batch(np.array(batch))
                all_features.append(features)
                batch = []
        
        # Process remaining clips
        if len(batch) > 0:
            features = self._process_batch(np.array(batch))
            all_features.append(features)
        
        # Concatenate all features
        clip_features = np.concatenate(all_features, axis=0)
        
        if verbose:
            print(f"  Extracted {len(clip_features)} clip features")
        
        # Segment to fixed number
        features = segment_features(clip_features, self.num_segments)
        
        if verbose:
            print(f"  Output shape: {features.shape}")
        
        return features
    
    def _extract_clip_features(self, 
                               clips: np.ndarray,
                               verbose: bool = True) -> np.ndarray:
        """
        Extract features from video clips.
        
        Args:
            clips: numpy array of shape (num_clips, T, H, W, C)
            verbose: Whether to print progress
        
        Returns:
            features: numpy array of shape (num_clips, feature_dim)
        """
        num_clips = len(clips)
        all_features = []
        
        for i in range(0, num_clips, self.batch_size):
            batch_clips = clips[i:i + self.batch_size]
            features = self._process_batch(batch_clips)
            all_features.append(features)
            
            if verbose and (i + self.batch_size) % (self.batch_size * 5) == 0:
                print(f"  Processed {min(i + self.batch_size, num_clips)}/{num_clips} clips")
        
        return np.concatenate(all_features, axis=0)
    
    def _process_batch(self, clips: np.ndarray) -> np.ndarray:
        """
        Process a batch of clips through I3D.
        
        Args:
            clips: numpy array of shape (B, T, H, W, C)
        
        Returns:
            features: numpy array of shape (B, feature_dim)
        """
        # Convert to tensor: (B, T, H, W, C) -> (B, C, T, H, W)
        clips = np.transpose(clips, (0, 4, 1, 2, 3))
        clips_tensor = torch.from_numpy(clips).float().to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features_from_video(clips_tensor)
        
        return features.cpu().numpy()
    
    def process_video_to_npy(self,
                            video_path: str,
                            output_path: Optional[str] = None,
                            verbose: bool = True) -> str:
        """
        Process video and save features to .npy file.
        
        Args:
            video_path: Path to input video
            output_path: Path for output .npy file. If None, uses video name.
            verbose: Whether to print progress
        
        Returns:
            output_path: Path to saved .npy file
        """
        # Extract features
        features = self.extract_features(video_path, verbose=verbose)
        
        # Determine output path
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"{video_name}_i3d.npy"
        
        # Save features
        np.save(output_path, features)
        
        if verbose:
            print(f"  Saved to: {output_path}")
        
        return output_path
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         video_extensions: tuple = ('.mp4', '.avi', '.mkv', '.mov'),
                         verbose: bool = True) -> List[str]:
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Input directory containing videos
            output_dir: Output directory for .npy files
            video_extensions: Video file extensions to process
            verbose: Whether to print progress
        
        Returns:
            output_files: List of output .npy file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(input_dir).glob(f"*{ext}"))
            video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if verbose:
            print(f"Found {len(video_files)} videos in {input_dir}")
        
        output_files = []
        for i, video_path in enumerate(video_files):
            if verbose:
                print(f"\n[{i+1}/{len(video_files)}]")
            
            output_name = video_path.stem + "_i3d.npy"
            output_path = os.path.join(output_dir, output_name)
            
            try:
                self.process_video_to_npy(
                    str(video_path),
                    output_path,
                    verbose=verbose
                )
                output_files.append(output_path)
            except Exception as e:
                print(f"  Error processing {video_path}: {e}")
        
        if verbose:
            print(f"\nProcessed {len(output_files)}/{len(video_files)} videos successfully")
        
        return output_files


def download_i3d_weights(output_path: str = "pretrained/rgb_imagenet.pt") -> str:
    """
    Download pretrained I3D weights from available sources.
    
    Note: Due to storage limitations on public hosts, the automatic download
    may not always work. In that case, please download manually.
    
    Args:
        output_path: Path to save weights
    
    Returns:
        output_path: Path to downloaded weights
    """
    import urllib.request
    import urllib.error
    
    # Multiple mirror sources for I3D weights
    # These may not always be available - manual download recommended
    urls = [
        # Alternative mirrors (availability varies)
        "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_rgb_imagenet.pt?dl=1",
    ]
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    print("Attempting to download I3D pretrained weights...")
    print("This may take a few minutes (~400MB)...")
    print("If automatic download fails, see instructions below.\n")
    
    for i, url in enumerate(urls):
        try:
            print(f"Trying source {i+1}/{len(urls)}: {url[:50]}...")
            urllib.request.urlretrieve(url, output_path)
            
            # Verify file was downloaded and has correct size (I3D is ~400MB)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000_000:  # > 100MB
                print(f"Successfully downloaded to: {output_path}")
                return output_path
            else:
                print(f"  Downloaded file too small, may be corrupted")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  Source {i+1} failed: {e}")
            continue
    
    # If all URLs fail, provide manual instructions
    print("\n" + "="*60)
    print("AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print("""
Please download I3D weights manually:

OPTION 1 - Original Repository:
  Go to: https://github.com/piergiaj/pytorch-i3d
  Look for pretrained weights in README or releases

OPTION 2 - Use pre-extracted features (Recommended for demo):
  Your project already has pre-extracted I3D features in:
  data/i3d_features/test/ and data/i3d_features/train/
  
  Use the "Upload Features" or "Demo Video" tabs in the Streamlit demo
  to analyze videos without needing to download I3D weights.

OPTION 3 - Run the setup script:
  python feature_extraction/setup_weights.py
  This will provide more detailed download options.
""")
    print(f"Expected file location: {os.path.abspath(output_path)}")
    print("="*60)
    
    raise RuntimeError("Could not download I3D weights - see instructions above")


# Convenience function for quick feature extraction
def extract_i3d_features(video_path: str,
                         weights_path: Optional[str] = None,
                         num_segments: int = 32,
                         device: Optional[str] = None) -> np.ndarray:
    """
    Quick function to extract I3D features from a video.
    
    Args:
        video_path: Path to video file
        weights_path: Path to pretrained I3D weights
        num_segments: Number of segments (default 32)
        device: Device to use
    
    Returns:
        features: numpy array of shape (num_segments, 1024)
    """
    extractor = I3DFeatureExtractor(
        weights_path=weights_path,
        device=device,
        num_segments=num_segments
    )
    
    return extractor.extract_features(video_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract I3D features from videos")
    parser.add_argument("input", help="Video file or directory")
    parser.add_argument("-o", "--output", help="Output path or directory")
    parser.add_argument("-w", "--weights", help="Path to I3D weights")
    parser.add_argument("-s", "--segments", type=int, default=32,
                       help="Number of segments (default: 32)")
    parser.add_argument("-d", "--device", choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--download-weights", action="store_true",
                       help="Download pretrained weights")
    
    args = parser.parse_args()
    
    # Download weights if requested
    if args.download_weights:
        args.weights = download_i3d_weights()
    
    # Create extractor
    extractor = I3DFeatureExtractor(
        weights_path=args.weights,
        device=args.device,
        num_segments=args.segments
    )
    
    # Process input
    if os.path.isdir(args.input):
        output_dir = args.output or os.path.join(args.input, "i3d_features")
        extractor.process_directory(args.input, output_dir)
    else:
        output_path = args.output
        if output_path is None:
            output_path = Path(args.input).stem + "_i3d.npy"
        extractor.process_video_to_npy(args.input, output_path)
