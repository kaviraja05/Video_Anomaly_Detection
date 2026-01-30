"""
Feature Extraction Package
Provides I3D feature extraction from raw videos.
"""

from feature_extraction.extract_features import (
    I3DFeatureExtractor,
    extract_i3d_features,
    download_i3d_weights
)

from feature_extraction.i3d_model import InceptionI3d, load_i3d_model

from feature_extraction.video_preprocessing import (
    load_video_frames,
    preprocess_frames_for_i3d,
    create_clips,
    segment_features,
    get_video_info,
    check_opencv
)

__all__ = [
    'I3DFeatureExtractor',
    'extract_i3d_features',
    'download_i3d_weights',
    'InceptionI3d',
    'load_i3d_model',
    'load_video_frames',
    'preprocess_frames_for_i3d',
    'create_clips',
    'segment_features',
    'get_video_info',
    'check_opencv'
]
