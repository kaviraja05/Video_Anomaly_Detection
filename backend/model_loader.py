import torch
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.proposed_model import build_model
from utils.config import get_config

class ModelLoader:
    """
    Manages loading and inference for the pretrained RTFM model.
    Loads pretrained weights once at backend startup.
    """
    def __init__(self, model_path="experiments/checkpoints/best_model.pth"):
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(project_root) / model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = build_model(self.config)
            
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded pretrained weights.")
            else:
                logger.warning(f"No checkpoint found at {self.model_path}. Model will be untrained.")
                
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features: torch.Tensor):
        if not self.is_loaded:
            self.load_model()
            
        with torch.no_grad():
            features = features.to(self.device)
            output = self.model(features, return_features=True)
            return {
                'scores': output['scores'].cpu().numpy(),
                'attention_weights': output.get('attention_weights', None),
                'graph_features': output.get('graph_features', None),
                'embeddings': output.get('features', None)
            }
