import torch
import os
import sys
from pathlib import Path

# Now living dynamically in project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from models.proposed_model import build_model
    from utils.config import get_config
except ImportError as e:
    print(f"Failed to import models. Ensure you are running this from D:/Video_Anomaly_Detection/Video_Anomaly_Detection: {e}")
    sys.exit(1)

class ONNXWrapper(torch.nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core_model = core_model
        
    def forward(self, x):
        out = self.core_model(x, return_features=True)
        return out.get('scores', torch.empty(1)), out.get('features', torch.empty(1))

def export_to_onnx():
    config = get_config()
    model = build_model(config)
    
    model_path = project_root / "experiments/checkpoints/best_model.pth"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {model_path}")
    else:
        print("WARNING: No checkpoint found, exporting UNTRAINED model!")
        
    model.eval()
    
    # Initialize Wrapper
    wrapper = ONNXWrapper(model)
    
    dummy_input = torch.randn(1, getattr(config, 'num_segments', 32), getattr(config, 'output_dim', 2048))
    
    onnx_path = project_root / "experiments/checkpoints/model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['scores', 'features'],
        dynamic_axes={
            'input_features': {0: 'batch_size', 1: 'num_segments'},
            'scores': {0: 'batch_size', 1: 'num_segments'},
            'features': {0: 'batch_size', 1: 'num_segments'}
        }
    )
    print(f"✅ ONNX model successfully exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
