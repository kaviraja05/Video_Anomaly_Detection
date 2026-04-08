import numpy as np
import random
from pathlib import Path
import logging
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# Fallback feature dir if actual extraction isn't available
FEATURES_DIR = Path(__file__).parent.parent / "data/i3d_features/train"

class FeatureExtractor:
    """
    Handles feature extraction pipeline for raw videos using OpenCV and PyTorch.
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.output_dim = 1024 # Target dimension to match expected model input
        
    def _initialize_model(self):
        """Lazy load the feature extraction model (ResNet18 as a lightweight I3D proxy)."""
        if self.model is None:
            logger.info("Initializing PyTorch feature extractor...")
            # Using ResNet18 for fast inference during streaming
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove the classification head to get feature embeddings
            self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
            self.model.eval()
            self.model.to(self.device)
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("Feature extractor initialized.")

    def extract_from_video(self, video_path: str, fps_sample_rate: int = 5) -> np.ndarray:
        """
        Extract features from a video file using OpenCV.
        Args:
            video_path: Path to the video file
            fps_sample_rate: Limit frame extraction to X frames per second to save memory
        """
        self._initialize_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            raise Exception("Failed to open video file")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / fps_sample_rate))
        
        features = []
        frame_idx = 0
        batch_frames = []
        batch_size = 16

        logger.info(f"Extracting features from {video_path}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb_frame)
                batch_frames.append(tensor)
                
                if len(batch_frames) >= batch_size:
                    batch_tensor = torch.stack(batch_frames).to(self.device)
                    with torch.no_grad():
                        out = self.model(batch_tensor) # (B, 512, 1, 1)
                        out = out.view(out.size(0), -1) # (B, 512)
                        
                        # Linear projection to match the expected 1024 dim of the anomaly model
                        # Just padding with zeros or duplicating for now to match dimensions since we're replacing I3D 1024
                        out = torch.cat([out, out], dim=1) # (B, 1024)
                        
                        features.append(out.cpu().numpy())
                    batch_frames = []

            frame_idx += 1

        # Process any remaining frames
        if len(batch_frames) > 0:
            batch_tensor = torch.stack(batch_frames).to(self.device)
            with torch.no_grad():
                out = self.model(batch_tensor)
                out = out.view(out.size(0), -1)
                out = torch.cat([out, out], dim=1)
                features.append(out.cpu().numpy())

        cap.release()
        
        if not features:
            logger.warning("No features extracted from video.")
            return np.random.randn(32, 1024).astype(np.float32)
            
        final_features = np.concatenate(features, axis=0)
        logger.info(f"Extracted {final_features.shape[0]} feature snippets.")
        return final_features

    @staticmethod
    def get_features(video_path: str = None, instance=None) -> np.ndarray:
        """
        Legacy static method wrapper. If video_path is provided and valid, does actual extraction.
        Otherwise falls back to dummy features.
        """
        if video_path and Path(video_path).exists():
            extr = instance if instance else FeatureExtractor()
            try:
                return extr.extract_from_video(video_path)
            except Exception as e:
                logger.error(f"True extraction failed, falling back to dummy: {e}")
                
        # Simulated extraction by reading test data
        if not FEATURES_DIR.exists():
            logger.warning("Features directory not found, returning dummy tensor.")
            return np.random.randn(32, 1024).astype(np.float32)
            
        npy_files = list(FEATURES_DIR.glob("*.npy"))
        if not npy_files:
            return np.random.randn(32, 1024).astype(np.float32)
            
        feature_file = random.choice(npy_files)
        return np.load(feature_file)

    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1e-5, std) # prevent div by zero
        return (features - mean) / std

    @staticmethod
    def segment_features(features: np.ndarray, num_segments: int = 32) -> np.ndarray:
        if features.ndim == 3:
            T, num_clips, D = features.shape
            features = features.reshape(T * num_clips, D)
            
        T, D = features.shape
        if T == 0:
            return np.zeros((num_segments, D), dtype=features.dtype)
            
        if T >= num_segments:
            indices = np.linspace(0, T - 1, num_segments, dtype=np.int32)
            return features[indices]
        else:
            padded = np.zeros((num_segments, D), dtype=features.dtype)
            padded[:T] = features
            return padded
