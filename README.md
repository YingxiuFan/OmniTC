# OmniTC

## Project Structure

```
OmniTC/
├── config.py                       # Configuration file
├── run.py                          # Training entry point
├── main.py                         # Core training functions
├── predict_and_filter.py           # Prediction and score-threshold filtering script
├── models/                         # Model definitions
│   ├── __init__.py
│   ├── target_synergy.py          # Main model
│   ├── target_encoder.py          # Target encoder (JK-Net)
│   ├── target_pair_aggregator.py  # Target-pair aggregator (CBP)
│   ├── disease_encoder.py         # Disease encoder (G-means)
│   └── interaction.py             # Cross-attention layer
├── data/                           # Data loading
│   ├── __init__.py
│   ├── dataset.py                 # Dataset and negative sampler
│   ├── data_splitter_v2.py        # Data splitting
│   └── disease_feature_calculator.py
├── losses/                         # Loss functions
│   ├── __init__.py
│   ├── hybrid_loss.py             # Hybrid loss (BCE+Orth+Con+Adv)
│   └── adversarial_loss.py        # Adversarial loss
├── output/                         # Precomputed disease features
│   └── disease_features/ttd_based/ # 10 cancer n2v features (9 train + TNBC)
├── ckpt/                           # Checkpoint & test sets
│   ├── best_model.pt            # best checkpoint
│   └── test_cross_disease.csv           # test set
└── utils/                          # Utility functions
    ├── __init__.py
    ├── clustering.py              # G-means clustering
    └── metrics.py                 # Evaluation metrics
```

## Installation

```bash
conda activate OmniTC
cd /OmniTC
```

The key packages in that environment are:

- Python 3.9.23
- PyTorch 2.7.1+cu118
- torch-geometric 2.6.1
- torch-cluster 1.6.3+pt27cu118
- numpy 2.0.2
- pandas 2.3.3
- scipy 1.13.1
- h5py 3.14.0
- scikit-learn 1.6.1
- networkx 3.2.1
- tqdm 4.67.1
- node2vec 0.5.0

If you need to recreate the environment, use the same package family and CUDA/PyTorch-Geometric compatibility. A minimal starting point is:

```bash
conda create -n OmniTC python=3.9 -y
conda activate OmniTC
pip install numpy==2.0.2 pandas==2.3.3 scipy==1.13.1 h5py==3.14.0 scikit-learn==1.6.1 networkx==3.2.1 tqdm==4.67.1 node2vec==0.5.0
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1 torch-cluster==1.6.3
```

For PyTorch-Geometric extension packages, install wheels matching the selected PyTorch and CUDA versions if the plain `pip install` command does not resolve a compatible build.

## Data Requirements

This repository provides precomputed disease features, TTD target files, target-pair data, and test splits under `output/`, `datasets/`, and `ckpt/` respectively. The following external data must be obtained separately:

- `datasets/full_graph/heterogeneous_graph_with_features.h5`: full heterogeneous graph with node features and edges：https://doi.org/10.5281/zenodo.20912868
- `datasets/targetpair/final_merged_ttd_filtered.csv`: target-pair data from TargetPair: https://www.targetpair.aiddlab.com/home/index (a copy is provided in this repo for reproducibility)
- `datasets/TTD/*.csv`: disease-target files from Therapeutic Target Database (TTD): https://idrblab.org/ttd/ (copies for 13 diseases are provided in this repo)
- `output/disease_features/ttd_based/*_n2v_features.npz`: precomputed disease features. Training can generate these when the graph and TTD files are available, or use the 10 precomputed features provided in this repo.

## Training

Run all folds with the default diseases:

```bash
conda activate OmniTC
python run.py
```

Train selected diseases:

```bash
python run.py --diseases LUAD BRCA COAD
```

Train a single fold for a short run:

```bash
python run.py --fold_index 0 --num_epochs 10
```

Useful runtime options:

- `--device cuda:0` or `--device cpu`
- `--batch_size 64`
- `--num_workers 4`
- `--output_dir ckpt/my_run`
- `--force_recompute` to recompute disease features instead of reusing existing files

Training checkpoints are saved under `ckpt/` by default, unless `--output_dir` is specified.

## Prediction and Filtering

Prepare an input CSV with either:

- `gene1`, `gene2`, `disease`
- `genename1`, `genename2`, plus one disease column from `disease`, `specific_cancer`, `fd_name`, or `md_name`

The disease value can be a supported disease code such as `BRCA`, or an English disease name such as `Breast Cancer`.

Run prediction:

```bash
python predict_and_filter.py \
    -c ckpt/best_model.pt \
    -i input_pairs.csv \
    -o predictions.csv \
    --min-score 0.5 \
    --device cuda:0
```

The script writes all predictions to `predictions.csv` and writes filtered predictions to `predictions_filtered.csv` unless `--filtered-output` is provided.

Output labels:

- `valid`: both genes and disease features were found.
- `oog`: at least one gene was out of the graph vocabulary.
- `error`: disease features could not be loaded.

## Zero-Shot Prediction for Novel Diseases (TNBC Example)

The pre-trained model can predict target pairs for diseases **not seen during training**, as long as disease features can be computed from TTD target data. This section uses **Triple-Negative Breast Cancer (TNBC)** as a concrete example — TNBC is not among the 9 training diseases, yet the model can make predictions for it without retraining.

### How It Works

The disease encoder transforms any disease's TTD target set into a fixed-dimensional cluster representation via G-means clustering and Node2Vec topological embeddings. As long as the disease's target genes exist in the full heterogeneous graph and a TTD target list is available, the model can generate valid disease feature vectors.

### Step 1: Prepare TTD Target File

A TNBC TTD file (`datasets/TTD/icd11_TNBC_targets.csv`) is already provided in this repo. For your own novel disease, prepare a CSV in the same format (gene names + ICD-11 codes).

### Step 2: Register the Disease in config.py

Add the new disease to `DISEASE_CONFIGS` in `config.py`:

```python
DISEASE_CONFIGS = {
    # ... existing 9 diseases ...
    'TNBC': {
        'name_cn': '三阴性乳腺癌',
        'name_en': 'Triple-Negative Breast Cancer',
        'tcga_list': ['TNBC'],
        'icd11_files': ['icd11_TNBC_targets.csv'],
    },
}
```

### Step 3: Precompute Disease Features

TNBC features (`output/disease_features/ttd_based/TNBC_n2v_features.npz`) are already provided. To compute features for a new disease:

**Option A** — via training script (easiest):
```bash
python run.py --diseases NEW_DISEASE --fold_index 0 --num_epochs 1
# Interrupt after features are generated
```

**Option B** — via Python API:
```python
from data.disease_feature_calculator import compute_disease_features_for_disease

features, n_clusters = compute_disease_features_for_disease(
    disease_code='TNBC',
    full_graph_path='datasets/full_graph/heterogeneous_graph_with_features.h5',
    ttd_dir='datasets/TTD',
    output_dir='output/disease_features/ttd_based',
    k_max=30, node2vec_dim=128, node2vec_gpu=True,
)
```

### Step 4: Run Zero-Shot Prediction

```bash
python predict_and_filter.py \
    -c ckpt/best_model.pt \
    -i tnbc_candidates.csv \
    -o tnbc_predictions.csv \
    --min-score 0.5 \
    --device cuda:0
```

Example `tnbc_candidates.csv`:

```csv
gene1,gene2,disease
BRCA1,TP53,TNBC
EGFR,MYC,TNBC
PARP1,BRCA1,TNBC
```

> **Notes**: Zero-shot quality depends on the biological similarity between the novel disease and the training diseases. TNBC, as a breast cancer subtype, shares molecular features with BRCA (included in training), leading to reasonable zero-shot performance. If many target genes of the novel disease are absent from the full graph, those samples will be labeled `oog`.

## Few-Shot Training for Novel Diseases (TNBC Example)

If you have a small number of positive target pairs for a novel disease, you can fine-tune the model.

### Strategy 1: Joint Training (recommended)

Append TNBC positive samples to `datasets/targetpair/final_merged_ttd_filtered.csv`, then train jointly:

```bash
python run.py --diseases LUAD BRCA COAD KIRC LIHC SKCM OV PAAD PRAD TNBC
```

The 9 training diseases act as a regularizer, preventing overfitting on the small TNBC dataset.


### Strategy 2: Fine-Tune from Pretrained Weights

Modify `run.py` to load the pretrained checkpoint before training:

```python
model = TargetSynergyModel(...)
checkpoint = torch.load('ckpt/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# Then continue training
```

`strict=False` allows partial weight loading when the novel disease introduces genes not in the pretrained graph vocabulary.

## Precomputed Disease Features

This repository includes precomputed disease features (`*_n2v_features.npz`) for 10 cancers in `output/disease_features/ttd_based/`:

| Code | Disease | Use |
|------|---------|-----|
| BRCA | Breast Cancer | Training |
| COAD | Colorectal Cancer | Training |
| KIRC | Kidney Cancer | Training |
| LIHC | Liver Cancer | Training |
| LUAD | Lung Cancer | Training |
| OV | Ovarian Cancer | Training |
| PAAD | Pancreatic Cancer | Training |
| PRAD | Prostate Cancer | Training |
| SKCM | Melanoma | Training |
| **TNBC** | **Triple-Negative BC** | **Zero-shot demo** |

Each `.npz` contains `disease_features` (shape `[K, hidden_dim]`) and `n_clusters` (number of G-means clusters).

## Example Checkpoint and Test Set

- `ckpt/best_model.pt` — Ready-to-use checkpoint (trained on 9 solid tumors).
- `ckpt/test_cross_disease.csv` — Cross-disease negative sampling test set with columns `gene1, gene2, target1_idx, target2_idx, disease, label`.

