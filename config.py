# -*- coding: utf-8 -*-
"""
TargetSynergy Simplified Configuration
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

PATHS = {
    "full_graph_h5": DATASETS_ROOT / "full_graph/heterogeneous_graph_with_features.h5",
    "positive_pairs": DATASETS_ROOT / "targetpair/final_merged_ttd_filtered.csv",
    "ttd_dir": DATASETS_ROOT / "TTD",
    "reactome_path": DATASETS_ROOT / "Reactom/Ensembl2Reactome_Homo_sapiens_GeneName.txt",
    "output_dir": PROJECT_ROOT / "output",
    "cluster_output": PROJECT_ROOT / "output/clustering",
    "subgraph_output": PROJECT_ROOT / "output/subgraphs",
    "data_output": PROJECT_ROOT / "output/training_data",
    "log_dir": PROJECT_ROOT / "log",
    "ckpt_dir": PROJECT_ROOT / "ckpt",
    "global_clusters": PROJECT_ROOT / "output/clustering/full_graph_clusters_original_kmax30.npz",
}

DISEASE_CONFIGS = {
    'LUAD': {'name_cn': '肺癌', 'name_en': 'Lung Cancer', 'tcga_list': ['LUAD', 'LUSC'], 'icd11_files': ['icd11_2C25_targets.csv']},
    'BRCA': {'name_cn': '乳腺癌', 'name_en': 'Breast Cancer', 'tcga_list': ['BRCA'], 'icd11_files': ['icd11_2C60-2C6Y_targets.csv', 'icd11_2C60-2C6Z_targets.csv']},
    'COAD': {'name_cn': '结直肠癌', 'name_en': 'Colorectal Cancer', 'tcga_list': ['COAD', 'READ'], 'icd11_files': ['icd11_2B90_targets.csv', 'icd11_2B91_targets.csv', 'icd11_2B92_targets.csv']},
    'KIRC': {'name_cn': '肾透明细胞癌', 'name_en': 'Kidney Cancer', 'tcga_list': ['KIRC', 'KIRP'], 'icd11_files': ['icd11_2C90_targets.csv']},
    'LIHC': {'name_cn': '肝癌', 'name_en': 'Liver Cancer', 'tcga_list': ['LIHC'], 'icd11_files': ['icd11_2C12_targets.csv']},
    'SKCM': {'name_cn': '黑色素瘤', 'name_en': 'Melanoma', 'tcga_list': ['SKCM'], 'icd11_files': ['icd11_2C30_targets.csv']},
    'OV': {'name_cn': '卵巢癌', 'name_en': 'Ovarian Cancer', 'tcga_list': ['OV'], 'icd11_files': ['icd11_2C73_targets.csv']},
    'PAAD': {'name_cn': '胰腺癌', 'name_en': 'Pancreatic Cancer', 'tcga_list': ['PAAD'], 'icd11_files': ['icd11_2C10_targets.csv']},
    'PRAD': {'name_cn': '前列腺癌', 'name_en': 'Prostate Cancer', 'tcga_list': ['PRAD'], 'icd11_files': ['icd11_2C82_targets.csv']},
}

TARGETPAIR_SOLID_TUMORS = ['BRCA', 'COAD', 'KIRC', 'LIHC', 'LUAD', 'SKCM', 'OV', 'PAAD', 'PRAD']

# ============================================
# 节点类型定义
# ============================================
NODE_TYPE_NAMES = ["circRNA", "miRNA", "Gene", "lncRNA", "TF", "RBP", "Reactom", "GO"]
NODE_TYPE_MAP = {name: i for i, name in enumerate(NODE_TYPE_NAMES)}

# 蛋白质编码类型（Gene, TF, RBP）
PROTEIN_TYPES = {2, 4, 5}
TYPE_GENE = 2
TYPE_TF = 4
TYPE_RBP = 5

# ============================================
# 模型默认超参数（基线配置）
# ============================================
MODEL_CONFIG = {
    "jk_num_layers": 3,
    "jk_hidden_dim": 128,
    "jk_mode": "concat",
    "jk_dropout": 0.3,
    "gmeans_k_max": 30,
    "num_attention_heads": 4,
    "attention_hidden_dim": 128,
    "num_recurrent_iterations": 1,
    "bidirectional": True,
    "final_dropout": 0.3,
}

TRAINING_CONFIG = {
    "external_test_ratio": 0.1,
    "n_folds": 5,
    "neg_strategy": "cross_disease",
    "neg_ratio": 1.0,
    "batch_size": 32,
    "num_epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "alpha_orth": 0.1,
    "beta_con": 0.5,
    "temperature": 0.1,
    "patience": 20,
    "min_delta": 1e-4,
    "optim_metric": "auprc",
}

# ============================================
# 基线模型固定配置
# ============================================
BASELINE_CONFIG = {
    "use_node2vec": True,
    "directed": True,
    "subgraph_strategy": "ttd_based",
    "eval_mode": "stratified_kfold",
    "split_mode": "sample_level_stratified",
    "target_pair_aggregation": "cbp",
    "use_bce": True,
    "use_orth": True,
    "use_con": True,
    "use_adversarial": True,
    "gamma_adversarial": 1.0,
    "adversarial_temperature": 0.5,
}

for key, path in PATHS.items():
    if "dir" in key and not path.is_symlink():
        path.mkdir(parents=True, exist_ok=True)
