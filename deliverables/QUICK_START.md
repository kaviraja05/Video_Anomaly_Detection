# Quick Start Guide

## Video Anomaly Detection using Graph Neural Networks

This guide helps reviewers quickly set up and run the project.

---

## 1. Environment Setup (5 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Quick Test (2 minutes)

```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from models.proposed_model import ProposedModel; print('Model: OK')"

# Run test evaluation
python test.py
```

Expected output:
```
Loading best model...
Evaluating on test set...
AUC-ROC: 86.7%
Average Precision: 32.4%
```

---

## 3. Run Demo (3 minutes)

### Option A: Web Interface
```bash
streamlit run demo/streamlit_app.py
```
Open http://localhost:8501 in browser

### Option B: Command Line
```bash
python demo/cli_demo.py data/i3d_features/test/Normal_Videos_452_x264_i3d.npy
```

---

## 4. Reproduce Experiments (30 minutes)

```bash
# Run all experiments
python experiments/run_experiments.py --all

# Generate visualizations
python experiments/visualizations.py
python docs/generate_diagrams.py
```

---

## 5. Train from Scratch (2-4 hours)

```bash
python train.py
```

Monitor training:
- Checkpoints saved to `experiments/checkpoints/`
- Logs saved to `experiments/logs/`

---

## Directory Structure

```
project/
├── train.py           # Training script
├── test.py            # Evaluation script
├── models/            # Model implementations
├── modules/           # Custom modules (DSM, GNN, RA²R)
├── utils/             # Utilities (config, dataloader)
├── demo/              # Demo interfaces
├── experiments/       # Experiment scripts & results
├── docs/              # Documentation
└── data/              # Dataset (I3D features)
```

---

## Key Files to Review

1. **Model Architecture**: `models/proposed_model.py`
2. **Novel Modules**: `modules/dsm.py`, `modules/ra2r.py`
3. **Training Logic**: `train.py`
4. **Evaluation**: `test.py`, `utils/eval_utils.py`

---

## Contact

For questions or issues, please refer to the documentation in `docs/` or the main README.md file.
