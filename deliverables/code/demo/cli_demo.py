"""
Command-Line Interface Demo for Video Anomaly Detection.

Provides a simple CLI for running inference on video features
without requiring a web interface.
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from utils.config import get_config
from models.proposed_model import build_model


def load_model(checkpoint_path: str = None):
    """Load the trained model."""
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(config)
    model = model.to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print("⚠️  Using randomly initialized model (no checkpoint found)")
    
    model.eval()
    return model, device, config


def segment_features(features: np.ndarray, num_segments: int = 32) -> np.ndarray:
    """Segment features to fixed length."""
    if len(features.shape) == 3:
        features = features.mean(axis=1)
    
    T, D = features.shape
    
    if T >= num_segments:
        indices = np.linspace(0, T - 1, num_segments, dtype=np.int32)
        return features[indices]
    else:
        padded = np.zeros((num_segments, D), dtype=features.dtype)
        padded[:T] = features
        return padded


@torch.no_grad()
def run_inference(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    config
) -> Dict:
    """Run inference on video features."""
    # Segment features
    features = segment_features(features, config.num_segments)
    
    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Forward pass
    output = model(features_tensor)
    scores = output['scores'].cpu().numpy()[0]
    
    # Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Analysis
    max_score = float(scores.max())
    mean_score = float(scores.mean())
    prediction = "ABNORMAL" if max_score > 0.5 else "NORMAL"
    confidence = max_score if max_score > 0.5 else 1 - max_score
    
    # Find anomalous regions
    anomalous_segments = np.where(scores > 0.5)[0].tolist()
    
    return {
        'scores': scores.tolist(),
        'max_score': max_score,
        'mean_score': mean_score,
        'prediction': prediction,
        'confidence': confidence,
        'anomalous_segments': anomalous_segments,
        'num_segments': len(scores)
    }


def print_result(result: Dict, video_name: str):
    """Print result in a formatted way."""
    print("\n" + "=" * 60)
    print(f"ANALYSIS RESULT: {video_name}")
    print("=" * 60)
    
    # Prediction
    pred_symbol = "🚨" if result['prediction'] == "ABNORMAL" else "✅"
    print(f"\n{pred_symbol} Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    
    # Scores
    print(f"\n📊 Score Statistics:")
    print(f"   Max Score: {result['max_score']:.4f}")
    print(f"   Mean Score: {result['mean_score']:.4f}")
    print(f"   Segments Analyzed: {result['num_segments']}")
    
    # Anomalous segments
    if result['anomalous_segments']:
        print(f"\n⚠️  Anomalous Segments: {result['anomalous_segments']}")
    else:
        print(f"\n✅ No anomalous segments detected")
    
    # Score visualization (ASCII)
    print(f"\n📈 Score Timeline:")
    scores = result['scores']
    max_width = 40
    for i, score in enumerate(scores):
        bar_width = int(score * max_width)
        bar = "█" * bar_width + "░" * (max_width - bar_width)
        status = "🔴" if score > 0.5 else "🟡" if score > 0.3 else "🟢"
        print(f"   [{i:2d}] {bar} {score:.3f} {status}")
    
    print("\n" + "=" * 60)


def analyze_video(feature_path: str, checkpoint_path: str = None, save_json: bool = False):
    """Analyze a single video."""
    if not os.path.exists(feature_path):
        print(f"❌ Feature file not found: {feature_path}")
        return None
    
    # Load features
    print(f"📁 Loading features from: {feature_path}")
    features = np.load(feature_path)
    print(f"   Feature shape: {features.shape}")
    
    # Load model
    model, device, config = load_model(checkpoint_path)
    
    # Run inference
    print("🔍 Running inference...")
    result = run_inference(model, features, device, config)
    
    # Print result
    video_name = os.path.basename(feature_path).replace('.npy', '')
    print_result(result, video_name)
    
    # Save JSON if requested
    if save_json:
        result['video_name'] = video_name
        result['timestamp'] = datetime.now().isoformat()
        
        json_path = feature_path.replace('.npy', '_result.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Result saved to: {json_path}")
    
    return result


def analyze_directory(dir_path: str, checkpoint_path: str = None, limit: int = None):
    """Analyze all videos in a directory."""
    if not os.path.isdir(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        return
    
    # Find all .npy files
    feature_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    
    if limit:
        feature_files = feature_files[:limit]
    
    print(f"📁 Found {len(feature_files)} feature files")
    
    # Load model once
    model, device, config = load_model(checkpoint_path)
    
    results = []
    abnormal_count = 0
    
    for i, filename in enumerate(feature_files, 1):
        feature_path = os.path.join(dir_path, filename)
        print(f"\n[{i}/{len(feature_files)}] Analyzing: {filename}")
        
        features = np.load(feature_path)
        result = run_inference(model, features, device, config)
        result['video_name'] = filename.replace('.npy', '')
        
        results.append(result)
        
        if result['prediction'] == 'ABNORMAL':
            abnormal_count += 1
        
        status = "🚨 ABNORMAL" if result['prediction'] == 'ABNORMAL' else "✅ NORMAL"
        print(f"   {status} (Max Score: {result['max_score']:.3f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total Videos: {len(results)}")
    print(f"  Abnormal: {abnormal_count} ({abnormal_count/len(results)*100:.1f}%)")
    print(f"  Normal: {len(results) - abnormal_count} ({(len(results)-abnormal_count)/len(results)*100:.1f}%)")
    print("=" * 60)
    
    # Save summary
    summary_path = os.path.join(dir_path, 'analysis_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(results),
        'abnormal_count': abnormal_count,
        'normal_count': len(results) - abnormal_count,
        'results': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Video Anomaly Detection CLI Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_demo.py --video path/to/features.npy
  python cli_demo.py --directory path/to/features/
  python cli_demo.py --video features.npy --checkpoint model.pth --save
        """
    )
    
    parser.add_argument('--video', '-v', type=str,
                       help='Path to single video feature file (.npy)')
    parser.add_argument('--directory', '-d', type=str,
                       help='Path to directory containing feature files')
    parser.add_argument('--checkpoint', '-c', type=str,
                       default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save results as JSON')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of videos for batch analysis')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     VIDEO ANOMALY DETECTION - CLI DEMO                   ║
    ║     Using GNN, DSM, RA²R, and MIL                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Default checkpoint path
    if args.checkpoint is None:
        args.checkpoint = os.path.join(project_root, 'experiments', 
                                       'checkpoints', 'best_model.pth')
    
    if args.video:
        analyze_video(args.video, args.checkpoint, args.save)
    elif args.directory:
        analyze_directory(args.directory, args.checkpoint, args.limit)
    else:
        # Demo mode with sample data
        print("No input specified. Running demo with test data...\n")
        
        test_dir = os.path.join(project_root, 'data', 'i3d_features', 'test')
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')][:3]
            if test_files:
                for filename in test_files:
                    feature_path = os.path.join(test_dir, filename)
                    analyze_video(feature_path, args.checkpoint, args.save)
            else:
                print("No test files found. Please specify --video or --directory.")
        else:
            print(f"Test directory not found: {test_dir}")
            print("Please specify --video or --directory option.")


if __name__ == '__main__':
    main()
