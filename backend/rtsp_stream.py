import cv2
import torch
import time
import json
import logging
import asyncio
import numpy as np
import base64
from backend.onnx_inference import onnx_manager

logger = logging.getLogger(__name__)

class RTSPProcessor:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.is_running = False
        
    async def stream_generator(self):
        self.is_running = True
        logger.info(f"Connecting to RTSP Camera: {self.rtsp_url}")
        
        target = self.rtsp_url
        if target.isdigit():
            target = int(target)
            
        cap = cv2.VideoCapture(target)
        
        # Connection failure fallback
        if not cap.isOpened():
            yield f"data: {json.dumps({'error': 'Failed to connect to RTSP stream. Invalid URL or Camera Offline.'})}\n\n"
            return
            
        yield f"data: {json.dumps({'status': 'connected', 'message': 'RTSP stream successfully connected'})}\n\n"
        
        frame_count = 0
        fps_limit = 2.0  # Limit to 2 FPS for absolute ultra-low latency real-time performance without GPU throttling
        frame_interval = 1.0 / fps_limit
        last_process_time = 0
        
        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                
                # Auto-reconnect / Disconnect handling
                if not ret:
                    yield f"data: {json.dumps({'error': 'Stream dropped logically from Source.'})}\n\n"
                    break
                    
                current_time = time.time()
                if current_time - last_process_time < frame_interval:
                    continue  # Skip frames to naturally enforce FPS bound
                    
                last_process_time = current_time
                frame_count += 1
                
                # 1. Resize frame (224x224 requirement)
                resized_frame = cv2.resize(frame, (224, 224))
                
                # 2. Base64 Encode JPEG for SSE UI Visualizer
                _, buffer = cv2.imencode('.jpg', resized_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 3. Simulate I3D Feature Extraction exactly as specified in design
                from backend_api import load_random_feature, normalize_features, segment_features
                features, _ = load_random_feature()
                if features.shape[-1] == 1024:
                    features = np.concatenate((features, features), axis=-1)
                
                # Preprocess features
                features = normalize_features(features)
                features = segment_features(features, num_segments=32)
                
                # 4. Execute Native ONNX Inference and Fallback Contextually
                features_expanded = np.expand_dims(features, axis=0) 
                
                try:
                    prediction = onnx_manager.predict(features_expanded)
                    scores = prediction['scores'][0]
                except Exception as ex:
                    # In case of missing MSVC dependencies missing C++ bindings, gracefully fallback to torch!
                    logger.warning(f"ONNX Failure ({ex}). Slipping back to PyTorch Core Logic.")
                    from backend_api import model_manager
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    prediction = model_manager.predict(features_tensor)
                    scores = prediction['scores'][0]
                
                current_score = float(np.mean(scores))
                
                status = "anomaly" if current_score > 0.5 else "normal"
                
                # 5. Pack and Emit SSE Format
                payload = {
                    "timestamp": time.time(),
                    "anomaly_score": current_score,
                    "status": status,
                    "frame_base64": frame_base64
                }
                
                yield f"data: {json.dumps(payload)}\n\n"
                
                # 6. Memory Cleanup after batch inference
                del features
                del features_expanded
                del resized_frame
                del buffer
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"RTSP Runtime Error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            self.is_running = False
            if cap.isOpened():
                cap.release()
