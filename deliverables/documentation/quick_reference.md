# Video Anomaly Detection using GNN - Quick Reference Card

## Project Overview
| Item | Value |
|------|-------|
| **Title** | Video Anomaly Detection using Graph Neural Networks |
| **Dataset** | UCF-Crime (1,900 videos, 13 anomaly types) |
| **Features** | I3D (2048-D pretrained on Kinetics) |
| **Framework** | PyTorch 2.0+ |
| **Best AUC-ROC** | **86.7%** |

---

## Architecture Summary

```
Input Video → I3D Features (2048-D)
    ↓
Feature Embedding (2048 → 512 → 128)
    ↓
Temporal Module (Multi-scale Conv1D)
    ↓
Dynamic Similarity Module (DSM) → Adjacency Matrix
    ↓
Graph Neural Network (2 layers, 4 heads)
    ↓
Relation-Aware Reasoning (RA²R)
    ↓
Anomaly Scorer (MLP → Sigmoid)
    ↓
Output: Per-segment anomaly scores [0,1]
```

---

## Key Components

### 1. Dynamic Similarity Module (DSM)
- **Purpose**: Learn content-aware graph structure
- **Method**: Attention-based adjacency with context gates
- **Innovation**: Adaptive edge weights based on segment similarity

### 2. Graph Neural Network
- **Type**: Multi-head graph attention
- **Layers**: 2
- **Heads**: 4
- **Features**: Message passing with residual connections

### 3. Relation-Aware Reasoning (RA²R)
- **Purpose**: Capture high-order relationships
- **Components**: Pairwise encoding + Transformer
- **Relation Types**: Concat, Difference, Product

### 4. MIL Loss
- **Base**: Ranking loss (max abnormal > max normal)
- **Regularization**: Smoothness + Sparsity terms

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_segments` | 32 | Video segments |
| `hidden_dim` | 128 | Feature dimension |
| `gnn_layers` | 2 | GNN depth |
| `num_heads` | 4 | Attention heads |
| `dropout` | 0.6 | Dropout rate |
| `learning_rate` | 1e-4 | AdamW LR |
| `batch_size` | 30 | 15 normal + 15 abnormal |
| `epochs` | 100 | Training epochs |

---

## Results Comparison

| Method | AUC-ROC (%) |
|--------|-------------|
| SVM | 50.0 |
| C3D + MIL | 75.4 |
| Sultani et al. | 77.9 |
| RTFM | 84.3 |
| **Ours (Proposed)** | **86.7** |

---

## Ablation Study

| Configuration | AUC-ROC (%) | Δ |
|--------------|-------------|---|
| Base (MIL only) | 77.4 | - |
| + DSM | 80.2 | +2.8 |
| + GNN | 83.5 | +3.3 |
| + RA²R (Full) | 86.7 | +3.2 |

---

## Quick Commands

```bash
# Training
python train.py

# Evaluation
python test.py

# Demo (Web)
streamlit run demo/streamlit_app.py

# Demo (CLI)
python demo/cli_demo.py <video_features.npy>

# Experiments
python experiments/run_experiments.py --all

# Generate Diagrams
python docs/generate_diagrams.py
```

---

## File Structure

```
├── train.py              # Training script
├── test.py               # Evaluation script
├── models/
│   ├── proposed_model.py # Main model
│   └── base_model.py     # Base components
├── modules/
│   ├── dsm.py            # Dynamic Similarity
│   ├── gnn_layer.py      # Graph Convolution
│   ├── ra2r.py           # Relation Reasoning
│   └── mil_loss.py       # MIL Loss
├── utils/
│   ├── config.py         # Configuration
│   ├── dataloader.py     # Data loading
│   └── eval_utils.py     # Evaluation
├── demo/                 # Demo interfaces
├── experiments/          # Experiment scripts
└── docs/                 # Documentation
```

---

## Key Formulas

**Attention Scores:**
$$\alpha_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right)$$

**Message Passing:**
$$h'_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \cdot V_j$$

**MIL Ranking Loss:**
$$\mathcal{L} = \max(0, 1 - \max(S_{abn}) + \max(S_{norm}))$$

---

## Viva Quick Answers

**Q: Why GNN over temporal models?**
A: GNN captures non-local dependencies between segments regardless of temporal distance.

**Q: How does DSM work?**
A: Computes attention between all segment pairs, applies context gating, thresholds to create sparse adjacency.

**Q: What is RA²R?**
A: Encodes pairwise relations (concat, diff, product) and uses transformer to reason about segment relationships.

**Q: Why MIL?**
A: Only video-level labels available; MIL allows learning from weak supervision.

---

## Contact & Resources

- **Code**: See `README.md` for setup instructions
- **Docs**: See `docs/` for detailed documentation
- **Demo**: Web interface at `demo/streamlit_app.py`
