"""
Streamlit Demo Application for Video Anomaly Detection.

Provides an interactive web interface for:
- Video/Feature upload
- Model inference
- Anomaly score visualization
- Explainability reports
"""

import os
import sys
import json
import numpy as np
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Run: pip install streamlit plotly")

import torch
import torch.nn as nn

from utils.config import get_config
from models.proposed_model import build_model


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

if STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="Video Anomaly Detection - GNN",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(checkpoint_path: str = None):
    """Load the trained model."""
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(config)
    model = model.to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success(f"✅ Model loaded from {checkpoint_path}")
    else:
        st.warning("⚠️ Using randomly initialized model (no checkpoint found)")
    
    model.eval()
    return model, device, config


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    config
) -> Dict:
    """
    Run inference on video features.
    
    Args:
        model: Trained model
        features: Video features of shape (T, D)
        device: Computation device
        config: Model configuration
        
    Returns:
        Dictionary with predictions and analysis
    """
    # Segment features
    features = segment_features(features, config.num_segments)
    
    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Forward pass
    output = model(features_tensor, return_features=True)
    
    scores = output['scores'].cpu().numpy()[0]  # (T,)
    
    # Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Analysis
    max_score = float(scores.max())
    mean_score = float(scores.mean())
    prediction = "🚨 ABNORMAL" if max_score > 0.5 else "✅ NORMAL"
    confidence = max_score if max_score > 0.5 else 1 - max_score
    
    # Find anomalous regions
    anomalous_segments = np.where(scores > 0.5)[0]
    
    return {
        'scores': scores,
        'max_score': max_score,
        'mean_score': mean_score,
        'prediction': prediction,
        'confidence': confidence,
        'anomalous_segments': anomalous_segments.tolist(),
        'num_segments': len(scores)
    }


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
# VISUALIZATION
# ============================================================================

def create_anomaly_timeline(scores: np.ndarray, title: str = "Anomaly Score Timeline") -> go.Figure:
    """Create interactive anomaly timeline plot."""
    n_segments = len(scores)
    
    # Create color scale based on scores
    colors = [f'rgb({int(255*s)}, {int(255*(1-s))}, 0)' for s in scores]
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        subplot_titles=(title, "Segment Status"),
        vertical_spacing=0.1
    )
    
    # Main anomaly score plot
    fig.add_trace(
        go.Bar(
            x=list(range(n_segments)),
            y=scores,
            marker_color=colors,
            name='Anomaly Score',
            hovertemplate='Segment %{x}<br>Score: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Threshold", row=1, col=1)
    
    # Line plot overlay
    fig.add_trace(
        go.Scatter(
            x=list(range(n_segments)),
            y=scores,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            name='Score Trend'
        ),
        row=1, col=1
    )
    
    # Status bar
    status_colors = ['green' if s < 0.3 else 'orange' if s < 0.5 else 'red' for s in scores]
    fig.add_trace(
        go.Bar(
            x=list(range(n_segments)),
            y=[1] * n_segments,
            marker_color=status_colors,
            showlegend=False,
            hovertemplate='Segment %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Segment Index", row=2, col=1)
    fig.update_yaxes(title_text="Anomaly Score", range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    
    return fig


def create_score_distribution(scores: np.ndarray) -> go.Figure:
    """Create score distribution plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        name='Score Distribution',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Threshold")
    
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300
    )
    
    return fig


def create_segment_analysis(result: Dict) -> go.Figure:
    """Create segment analysis pie chart."""
    scores = result['scores']
    
    normal = sum(1 for s in scores if s < 0.3)
    suspicious = sum(1 for s in scores if 0.3 <= s < 0.5)
    anomalous = sum(1 for s in scores if s >= 0.5)
    
    fig = go.Figure(data=[go.Pie(
        labels=['Normal', 'Suspicious', 'Anomalous'],
        values=[normal, suspicious, anomalous],
        marker_colors=['#2ECC71', '#F39C12', '#E74C3C'],
        hole=0.4
    )])
    
    fig.update_layout(
        title="Segment Classification",
        height=300,
        template="plotly_white"
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    if not STREAMLIT_AVAILABLE:
        print("Please install streamlit: pip install streamlit plotly")
        return
    
    # Header
    st.title("🔍 Video Anomaly Detection with GNN")
    st.markdown("""
    **Weakly Supervised Video Anomaly Detection** using I3D features, 
    Dynamic Similarity Module (DSM), Relation-Aware Reasoning (RA²R), 
    and Graph Neural Networks (GNN).
    """)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value=os.path.join(project_root, 'experiments', 'checkpoints', 'best_model.pth')
        )
        
        # Threshold
        threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
        
        # Number of segments
        num_segments = st.selectbox("Number of Segments", [16, 32, 64], index=1)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This demo showcases a state-of-the-art 
        video anomaly detection system trained 
        on the UCF-Crime dataset.
        
        **Components:**
        - 🎬 I3D Feature Extraction
        - 🔗 Dynamic Similarity Module
        - 🧠 Graph Neural Network
        - 🎯 Relation-Aware Reasoning
        - 📊 MIL-based Training
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        # Tab for different input methods
        tab1, tab2, tab3 = st.tabs(["Upload Features", "Demo Video", "Batch Analysis"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload I3D Features (.npy file)",
                type=['npy'],
                help="Upload pre-extracted I3D features in .npy format"
            )
            
            if uploaded_file is not None:
                try:
                    features = np.load(uploaded_file)
                    st.success(f"✅ Features loaded: Shape {features.shape}")
                    
                    # Display feature info
                    st.info(f"""
                    **Feature Statistics:**
                    - Shape: {features.shape}
                    - Mean: {features.mean():.4f}
                    - Std: {features.std():.4f}
                    """)
                    
                    if st.button("🚀 Run Analysis", key="analyze_upload"):
                        with st.spinner("Running inference..."):
                            # Load model
                            model, device, config = load_model(checkpoint_path)
                            
                            # Run inference
                            result = run_inference(model, features, device, config)
                            
                            st.session_state['result'] = result
                            st.session_state['video_name'] = uploaded_file.name
                
                except Exception as e:
                    st.error(f"Error loading features: {e}")
        
        with tab2:
            st.markdown("### Demo Analysis")
            
            # List available test features
            test_dir = os.path.join(project_root, 'data', 'i3d_features', 'test')
            if os.path.exists(test_dir):
                test_files = [f.replace('.npy', '') for f in os.listdir(test_dir) if f.endswith('.npy')][:20]
                
                selected_video = st.selectbox("Select Test Video", test_files)
                
                if st.button("🚀 Analyze Selected Video", key="analyze_demo"):
                    feature_path = os.path.join(test_dir, f"{selected_video}.npy")
                    if os.path.exists(feature_path):
                        with st.spinner("Running inference..."):
                            features = np.load(feature_path)
                            model, device, config = load_model(checkpoint_path)
                            result = run_inference(model, features, device, config)
                            
                            st.session_state['result'] = result
                            st.session_state['video_name'] = selected_video
            else:
                st.warning("Test features directory not found.")
        
        with tab3:
            st.markdown("### Batch Analysis")
            st.info("Upload multiple .npy files for batch processing.")
            
            batch_files = st.file_uploader(
                "Upload Multiple Features",
                type=['npy'],
                accept_multiple_files=True
            )
            
            if batch_files and st.button("🚀 Run Batch Analysis"):
                model, device, config = load_model(checkpoint_path)
                
                batch_results = []
                progress_bar = st.progress(0)
                
                for i, file in enumerate(batch_files):
                    features = np.load(file)
                    result = run_inference(model, features, device, config)
                    result['filename'] = file.name
                    batch_results.append(result)
                    progress_bar.progress((i + 1) / len(batch_files))
                
                st.session_state['batch_results'] = batch_results
    
    with col2:
        st.header("📊 Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            video_name = st.session_state.get('video_name', 'Unknown')
            
            # Prediction card
            prediction_color = "red" if "ABNORMAL" in result['prediction'] else "green"
            st.markdown(f"""
            <div style="background-color: {prediction_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{result['prediction']}</h2>
                <p style="color: white; margin: 5px 0;">Confidence: {result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Max Score", f"{result['max_score']:.3f}")
            col_m2.metric("Mean Score", f"{result['mean_score']:.3f}")
            col_m3.metric("Anomalous Segments", len(result['anomalous_segments']))
            
            # Timeline plot
            st.plotly_chart(
                create_anomaly_timeline(result['scores'], f"Timeline: {video_name}"),
                use_container_width=True
            )
            
            # Additional analysis
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.plotly_chart(create_score_distribution(result['scores']), 
                               use_container_width=True)
            
            with col_a2:
                st.plotly_chart(create_segment_analysis(result),
                               use_container_width=True)
            
            # Detailed segment info
            with st.expander("📋 Detailed Segment Analysis"):
                st.markdown("**Top Anomalous Segments:**")
                top_indices = np.argsort(result['scores'])[-5:][::-1]
                for idx in top_indices:
                    score = result['scores'][idx]
                    status = "🔴 Anomalous" if score > 0.5 else "🟡 Suspicious" if score > 0.3 else "🟢 Normal"
                    st.write(f"Segment {idx}: {score:.4f} {status}")
            
            # Export options
            with st.expander("📥 Export Results"):
                result_json = {
                    'video_name': video_name,
                    'prediction': result['prediction'],
                    'max_score': result['max_score'],
                    'mean_score': result['mean_score'],
                    'scores': result['scores'].tolist(),
                    'anomalous_segments': result['anomalous_segments'],
                    'timestamp': datetime.now().isoformat()
                }
                
                st.download_button(
                    "Download JSON Report",
                    data=json.dumps(result_json, indent=2),
                    file_name=f"anomaly_report_{video_name}.json",
                    mime="application/json"
                )
        
        elif 'batch_results' in st.session_state:
            st.subheader("Batch Results")
            
            batch_results = st.session_state['batch_results']
            
            # Summary table
            data = []
            for r in batch_results:
                data.append({
                    'Video': r['filename'],
                    'Prediction': '🚨 Abnormal' if r['max_score'] > 0.5 else '✅ Normal',
                    'Max Score': f"{r['max_score']:.3f}",
                    'Mean Score': f"{r['mean_score']:.3f}"
                })
            
            st.table(data)
            
            # Summary statistics
            abnormal_count = sum(1 for r in batch_results if r['max_score'] > 0.5)
            st.info(f"**Summary:** {abnormal_count}/{len(batch_results)} videos classified as abnormal")
        
        else:
            st.info("👈 Upload features or select a demo video to start analysis.")
            
            # Show sample visualization
            st.markdown("### Sample Visualization")
            np.random.seed(42)
            sample_scores = np.random.rand(32) * 0.4
            sample_scores[15:20] = 0.6 + np.random.rand(5) * 0.3
            st.plotly_chart(create_anomaly_timeline(sample_scores, "Sample: Anomaly Detection"),
                           use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 10px;">
        <p>Video Anomaly Detection with GNN | UCF-Crime Dataset | Research Project</p>
        <p>Components: I3D • DSM • RA²R • GNN • MIL</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
