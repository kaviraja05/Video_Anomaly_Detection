"""
Video Anomaly Detection - Professional Demo Application
Supports raw video upload with frame-level anomaly detection.
"""

import os
import sys
import json
import numpy as np
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn as nn
import cv2

from utils.config import get_config
from models.proposed_model import build_model


# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(
    page_title="Video Anomaly Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 600; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.95rem; }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-normal {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
    }
    
    .result-abnormal {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
    }
    
    .result-card h2 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .result-card .confidence { font-size: 1rem; opacity: 0.9; margin-top: 0.3rem; }
    
    .frame-card {
        background: #fee2e2;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        text-align: center;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained anomaly detection model."""
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config)
    model = model.to(device)
    
    checkpoint_path = os.path.join(project_root, 'experiments', 'checkpoints', 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
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


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def get_video_info(video_path: str) -> Dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    }
    cap.release()
    return info


def extract_frame_features(frame: np.ndarray) -> np.ndarray:
    """Extract 2048-D feature vector from a single frame (matching I3D output dimension)."""
    frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    features = []
    
    # Channel statistics (18 features)
    for c in range(3):
        ch = frame[:, :, c]
        features.extend([ch.mean(), ch.std(), ch.max(), ch.min(),
                        np.percentile(ch, 25), np.percentile(ch, 75)])
    
    # Grid features 8x8 (384 features)
    h, w = frame.shape[:2]
    gh, gw = h // 8, w // 8
    for i in range(8):
        for j in range(8):
            patch = frame[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            for c in range(3):
                features.extend([patch[:,:,c].mean(), patch[:,:,c].std()])
    
    # Edge features (8 features)
    gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([edge_mag.mean()/255, edge_mag.std()/255, 
                    edge_mag.max()/255, np.percentile(edge_mag, 90)/255,
                    np.percentile(edge_mag, 10)/255, np.percentile(edge_mag, 50)/255,
                    (edge_mag > 50).sum() / edge_mag.size, (edge_mag > 100).sum() / edge_mag.size])
    
    # Histogram features - more bins (192 features)
    for c in range(3):
        hist = cv2.calcHist([frame], [c], None, [64], [0, 1])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        features.extend(hist.tolist())
    
    # Texture features using Laplacian (4 features)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([laplacian.mean()/255, laplacian.std()/255, 
                    laplacian.max()/255, laplacian.min()/255])
    
    # Multi-scale pooling (more spatial features)
    for scale in [2, 4, 8, 16]:
        pooled = cv2.resize(frame, (scale, scale))
        for c in range(3):
            features.extend(pooled[:,:,c].flatten().tolist())
    
    # Pad to 2048-D
    current_len = len(features)
    if current_len < 2048:
        np.random.seed(int(abs(np.sum(features[:min(10, len(features))])) * 1000) % 2**31)
        features.extend((np.random.randn(2048 - current_len) * 0.01).tolist())
    
    return np.array(features[:2048], dtype=np.float32)


def process_video(video_path: str, num_segments: int = 32, 
                  progress_callback=None) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Process video and extract features.
    Returns: features, frame_indices for each segment, video_info
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'duration': total_frames / max(fps, 1)
    }
    
    # Sample frames for each segment
    samples_per_segment = 4
    total_samples = num_segments * samples_per_segment
    sample_indices = np.linspace(0, total_frames - 1, total_samples, dtype=int)
    
    # Track which frame corresponds to each segment
    segment_frame_mapping = []
    for seg in range(num_segments):
        start_idx = seg * samples_per_segment
        end_idx = start_idx + samples_per_segment
        seg_frames = sample_indices[start_idx:end_idx]
        segment_frame_mapping.append({
            'segment': seg,
            'start_frame': int(seg_frames[0]),
            'end_frame': int(seg_frames[-1]),
            'time_start': seg_frames[0] / max(fps, 1),
            'time_end': seg_frames[-1] / max(fps, 1)
        })
    
    # Extract features
    features_list = []
    
    for i, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            feat = extract_frame_features(frame)
            features_list.append(feat)
        
        if progress_callback and i % 10 == 0:
            progress_callback(i / len(sample_indices))
    
    cap.release()
    
    if len(features_list) == 0:
        raise ValueError("No frames could be extracted")
    
    features = np.array(features_list)
    
    # Average features within each segment
    segment_features = []
    for seg in range(num_segments):
        start_idx = seg * samples_per_segment
        end_idx = min(start_idx + samples_per_segment, len(features))
        if start_idx < len(features):
            seg_feat = features[start_idx:end_idx].mean(axis=0)
            segment_features.append(seg_feat)
        else:
            segment_features.append(np.zeros(2048, dtype=np.float32))
    
    return np.array(segment_features), segment_frame_mapping, video_info


@torch.no_grad()
def analyze_features(features: np.ndarray) -> Dict:
    """Run anomaly detection on features."""
    model, device, config = load_model()
    
    features = segment_features(features, config.num_segments)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    output = model(features_tensor, return_features=True)
    scores = output['scores'].cpu().numpy()[0]
    
    # Normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    max_score = float(scores.max())
    mean_score = float(scores.mean())
    is_abnormal = max_score > 0.5
    
    return {
        'scores': scores,
        'max_score': max_score,
        'mean_score': mean_score,
        'is_abnormal': is_abnormal,
        'confidence': max_score if is_abnormal else 1 - max_score,
        'anomalous_segments': np.where(scores > 0.5)[0].tolist()
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_timeline_chart(scores: np.ndarray, segment_mapping: List[Dict] = None) -> go.Figure:
    """Create anomaly timeline with frame information."""
    n = len(scores)
    
    colors = [
        f'rgba({int(220*s + 46*(1-s))}, {int(53*s + 204*(1-s))}, {int(69*s + 113*(1-s))}, 0.85)'
        for s in scores
    ]
    
    # Hover text with frame info
    if segment_mapping:
        hover_text = [
            f"Segment {i}<br>Score: {scores[i]:.3f}<br>"
            f"Frames: {m['start_frame']}-{m['end_frame']}<br>"
            f"Time: {m['time_start']:.1f}s - {m['time_end']:.1f}s"
            for i, m in enumerate(segment_mapping)
        ]
    else:
        hover_text = [f"Segment {i}<br>Score: {scores[i]:.3f}" for i in range(n)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(n)),
        y=scores,
        marker_color=colors,
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c",
                  annotation_text="Threshold", annotation_position="right")
    
    fig.add_trace(go.Scatter(
        x=list(range(n)),
        y=scores,
        mode='lines',
        line=dict(color='#3498db', width=2.5),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text="Anomaly Score Timeline", font=dict(size=16), x=0.5),
        xaxis=dict(title="Segment", showgrid=False, dtick=5),
        yaxis=dict(title="Anomaly Score", range=[0, 1.05], gridcolor='#ecf0f1'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=350,
        margin=dict(l=50, r=30, t=50, b=40),
        showlegend=False
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Video Anomaly Detection</h1>
        <p>Upload any video to detect anomalies with frame-level analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col_input, col_result = st.columns([4, 6])
    
    with col_input:
        st.markdown("### 📤 Upload Video")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mkv', 'mov', 'webm'],
            help="Upload MP4, AVI, MKV, MOV or WebM video"
        )
        
        if uploaded_video:
            # Show video preview
            st.video(uploaded_video)
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
            
            # Get video info
            video_info = get_video_info(tmp_path)
            st.info(f"📹 **{video_info['total_frames']}** frames | "
                   f"**{video_info['fps']:.1f}** FPS | "
                   f"**{video_info['duration']:.1f}s** duration")
            
            # Analyze button
            if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
                
                progress_bar = st.progress(0, text="Extracting features...")
                
                def update_progress(p):
                    progress_bar.progress(p, text=f"Processing frames... {int(p*100)}%")
                
                try:
                    # Process video
                    features, segment_mapping, v_info = process_video(
                        tmp_path, 
                        num_segments=32,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(0.9, text="Running anomaly detection...")
                    
                    # Analyze
                    result = analyze_features(features)
                    result['segment_mapping'] = segment_mapping
                    result['video_info'] = v_info
                    
                    progress_bar.progress(1.0, text="Complete!")
                    
                    st.session_state['result'] = result
                    st.session_state['video_name'] = uploaded_video.name
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        else:
            st.caption("Drag & drop or click to upload a video file")
    
    with col_result:
        if 'result' in st.session_state:
            result = st.session_state['result']
            video_name = st.session_state.get('video_name', 'Video')
            
            # Result card
            if result['is_abnormal']:
                st.markdown(f"""
                <div class="result-card result-abnormal">
                    <h2>⚠️ ANOMALY DETECTED</h2>
                    <div class="confidence">Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-normal">
                    <h2>✓ NORMAL VIDEO</h2>
                    <div class="confidence">Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Peak Score", f"{result['max_score']:.3f}")
            m2.metric("Average Score", f"{result['mean_score']:.3f}")
            m3.metric("Anomalous Segments", f"{len(result['anomalous_segments'])}/32")
            
            # Timeline chart
            segment_mapping = result.get('segment_mapping', None)
            st.plotly_chart(
                create_timeline_chart(result['scores'], segment_mapping),
                use_container_width=True
            )
            
            # Show anomalous frames
            if result['anomalous_segments'] and segment_mapping:
                st.markdown("### 🚨 Anomalous Frames Detected")
                
                video_info = result.get('video_info', {})
                fps = video_info.get('fps', 30)
                
                # Create a table of anomalous segments
                anomaly_data = []
                for seg_idx in result['anomalous_segments']:
                    if seg_idx < len(segment_mapping):
                        seg = segment_mapping[seg_idx]
                        anomaly_data.append({
                            'Segment': seg_idx,
                            'Frame Range': f"{seg['start_frame']} - {seg['end_frame']}",
                            'Time': f"{seg['time_start']:.1f}s - {seg['time_end']:.1f}s",
                            'Score': f"{result['scores'][seg_idx]:.3f}"
                        })
                
                if anomaly_data:
                    st.table(anomaly_data)
                    
                    # Summary
                    total_anomalous_frames = sum(
                        segment_mapping[idx]['end_frame'] - segment_mapping[idx]['start_frame']
                        for idx in result['anomalous_segments']
                        if idx < len(segment_mapping)
                    )
                    
                    st.warning(f"⚠️ **{len(result['anomalous_segments'])}** segments with anomalies "
                              f"covering approximately **{total_anomalous_frames}** frames")
            
            # Export
            with st.expander("📥 Export Report"):
                report = {
                    'video': video_name,
                    'result': 'ABNORMAL' if result['is_abnormal'] else 'NORMAL',
                    'confidence': round(result['confidence'], 4),
                    'max_score': round(result['max_score'], 4),
                    'mean_score': round(result['mean_score'], 4),
                    'anomalous_segments': [
                        {
                            'segment': idx,
                            'score': round(float(result['scores'][idx]), 4),
                            'frame_range': f"{segment_mapping[idx]['start_frame']}-{segment_mapping[idx]['end_frame']}",
                            'time_range': f"{segment_mapping[idx]['time_start']:.2f}s-{segment_mapping[idx]['time_end']:.2f}s"
                        }
                        for idx in result['anomalous_segments']
                        if segment_mapping and idx < len(segment_mapping)
                    ],
                    'all_scores': [round(float(s), 4) for s in result['scores']],
                    'timestamp': datetime.now().isoformat()
                }
                
                st.download_button(
                    "Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"anomaly_report_{video_name}.json",
                    mime="application/json"
                )
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #95a5a6;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <h3 style="color: #7f8c8d; font-weight: 500;">Upload a Video</h3>
                <p>Upload an MP4, AVI, or other video file to detect anomalies</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
