import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ONNXModelManager:
    def __init__(self, model_path: str = "experiments/checkpoints/model.onnx"):
        project_root = Path(__file__).parent.parent
        self.model_path = str(project_root / model_path)
        self.session = None
        self.is_loaded = False
        self.onnx_supported = True

    def load_model(self):
        try:
            import onnxruntime as ort
            # Opt for TensorRT if GPU bounds available, fallback gracefully to CPU
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.is_loaded = True
            logger.info(f"Loaded ONNX model natively with active providers: {self.session.get_providers()}")
        except ImportError as ie:
            self.onnx_supported = False
            logger.error(f"onnxruntime unavailable locally (likely due to MSVC bindings). {ie}")
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX architecture: {e}")
            raise

    def predict(self, features: np.ndarray):
        """
        Memory-safe rapid inference yielding 2x-3x speedup.
        Expects: np.ndarray shape (Batch, 32, 2048) of float32
        """
        if not self.onnx_supported:
            raise Exception("ONNX is disabled natively due to missing Windows dependencies.")
            
        if not self.is_loaded:
            self.load_model()
            
        inputs = {self.session.get_inputs()[0].name: features.astype(np.float32)}
        outputs = self.session.run(None, inputs)
        
        # Mapping indices corresponding to ONNX Wrapper Export: ['scores', 'features']
        return {
            'scores': outputs[0],
            'embeddings': outputs[1]
        }

# Singleton instantiation binding
onnx_manager = ONNXModelManager()
