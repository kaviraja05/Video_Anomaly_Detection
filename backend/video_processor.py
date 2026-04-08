import asyncio
import json
import time

class VideoProcessor:
    """
    Handles the orchestration of processing a video sequentially
    and streaming the results through Server-Sent Events (SSE).
    """
    
    @staticmethod
    def extract_anomaly_segments(scores, threshold=0.5, fps=30.0):
        anomaly_mask = scores > threshold
        segments = []
        i = 0
        while i < len(anomaly_mask):
            if anomaly_mask[i]:
                start = i
                while i < len(anomaly_mask) and anomaly_mask[i]:
                    i += 1
                end = i - 1
                
                confidence = float(sum(scores[start:end+1])) / (end - start + 1)
                segments.append({
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "timestamp_start": float(start / fps),
                    "timestamp_end": float(end / fps),
                    "confidence": confidence,
                    "severity": "high" if confidence > 0.8 else "medium"
                })
            i += 1
        return segments
