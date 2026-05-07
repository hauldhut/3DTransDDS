# 3DTransDDS: A Multimodal Framework for Drug Synergy Prediction via Gated Fusion of 2D-3D Drug Representations and Transformer-Based Interaction Modeling
 https://github.com/hauldhut/3DTransDDS/blob/master/Figure1.png

## Project Layout

- `create_data.py` - build processed graph datasets (`.pt`) from CSV files
- `main.py` - train 5-fold model and save best checkpoints
- `attention_synergy.py` - generate attention plots
- `data/` - raw data + dataset/graph builder code
- `model/` - model architecture (`GCNPointNetGateTransformer`)

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

1) Build processed data
```bash
python create_data.py --datafile new_labels_0_10 --cellfile data/cell_features.csv --data-dir data --work-root data
```

2) Train
```bash
python main.py --datafile new_labels_0_10 --data-root data --result-dir data/result --epochs 200 --lr 0.0002
```

3) Run attention plots
```bash
python attention_synergy.py --model-path "data/result/GCNPointNet_Gate_Transformer0--model_new_labels_0_10.model" --data-root data --datafile new_labels_0_10 --fold 0 --seed 0 --batch-size 128 --output-dir data/result
```

## Outputs

- Model checkpoints: `data/result/*.model`
- Metrics: `data/result/*AUCs*.txt`
- Figures:
  - `Fig_Attention_Main.png`
  - `Fig_PerHead.png`
  - `Fig_CaseStudy_Combined.png`

