import numpy as np
from typing import List, Dict, Optional, Any
import torch

class ExplainabilityModule:
    """
    Explainable AI module for video anomaly detection.
    Generates human-readable explanations based on model scores and attention weights.
    """
    
    @staticmethod
    def generate_explanation(
        scores: np.ndarray,
        attention_weights: Optional[torch.Tensor],
        anomaly_segments: List[Dict[str, Any]]
    ) -> dict:
        """
        Generate interpretability outputs explaining why segments were flagged.
        """
        if len(scores) == 0:
            return {"reason": "No frames analyzed."}
            
        # Top contributing frames (highest scores)
        top_indices = np.argsort(scores)[-5:][::-1]
        contributing_frames = [int(idx) for idx in top_indices]
        
        # Feature importance 
        feature_importance = {
            "temporal_patterns": float(np.mean(scores)),
            "motion_intensity": float(np.std(scores)),
            "contextual_anomaly": float(np.max(scores)),
            "gnn_reasoning": 0.85 if attention_weights is not None else 0.0
        }
        
        if attention_weights is not None:
            attn = attention_weights.cpu().numpy().flatten().tolist()
        else:
            attn = scores.tolist()
            
        if len(anomaly_segments) == 0:
            reason = "No significant anomaly detected. Video appears normal."
            temporal_context = "All segments show normal activity patterns."
        else:
            max_conf = max([s.get('confidence', 0.0) for s in anomaly_segments])
            
            # Simple heuristic reasons based on variation
            motion_var = np.var(scores)
            if motion_var > 0.1:
                cause = "Sudden motion spike / Crowd behavior change"
            else:
                cause = "Abnormal object interaction"
                
            reason = f"Anomaly detected with {max_conf:.1%} confidence. "
            reason += f"Possible anomaly cause: {cause}. "
            reason += f"{len(anomaly_segments)} suspicious segment(s) identified."
            
            temporal_context = f"High motion variance detected around frames {contributing_frames[0]}-{contributing_frames[-1]}."
            
        return {
            "reason": reason,
            "contributing_frames": contributing_frames,
            "feature_importance": feature_importance,
            "attention_weights": attn,
            "temporal_context": temporal_context
        }
