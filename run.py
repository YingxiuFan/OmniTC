#!/usr/bin/env python3
"""
TargetSynergy Training Entry Point (Simplified)

Fixed Configuration:
- Target Encoder: JK-Net 3 layers GIN + concat fusion
- Disease Encoder: G-means clustering + centrality weighted aggregation
- Interaction: 1 round bidirectional recurrent attention, 4 heads
- Target Pair Aggregation: CBP (Compact Bilinear Pooling)
- Loss: BCE + Orth + Con + Adv (all enabled)
- Graph: Directed graph + node2vec topology embedding
- Training Negative Sampling: cross_disease
- External Test Sets: hard and cross_disease
- Eval Mode: stratified_kfold
- Split Mode: sample_level_stratified
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, TRAINING_CONFIG, DISEASE_CONFIGS, TARGETPAIR_SOLID_TUMORS
from models.target_synergy import TargetSynergyModel
from data.dataset import NegativeSampler, collate_fn
from torch.utils.data import Dataset, DataLoader
from data.data_splitter_v2 import create_multi_disease_splits_v2
from main import train, test


def parse_args():
    parser = argparse.ArgumentParser(
        description='TargetSynergy Training (Simplified)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument('--diseases', nargs='+', default=None,
                       help='Diseases to train on (default: all 9 solid tumors)')
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                       help='Positive/negative sample ratio')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--fold_index', type=int, default=None,
                       help='Specific fold to train (None = all folds)')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=64,
                       help='Output dimension')
    parser.add_argument('--jk_num_layers', type=int, default=3,
                       help='JK-Net layers')
    parser.add_argument('--disease_k_max', type=int, default=30,
                       help='G-means max clusters')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                       help='Attention heads')
    parser.add_argument('--num_recurrent_iterations', type=int, default=1,
                       help='Recurrent iterations')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optim_metric', type=str, default='auprc',
                       choices=['auc', 'mcc', 'f1', 'auprc'],
                       help='Optimization metric for early stopping')

    # Loss parameters
    parser.add_argument('--alpha_orth', type=float, default=0.1,
                       help='Orthogonal loss weight')
    parser.add_argument('--beta_con', type=float, default=0.5,
                       help='Contrastive loss weight')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Contrastive loss temperature')
    parser.add_argument('--gamma_adversarial', type=float, default=1.0,
                       help='Adversarial loss weight')
    parser.add_argument('--adversarial_temperature', type=float, default=0.5,
                       help='Adversarial loss temperature')

    # Other parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recompute disease features')

    return parser.parse_args()


def setup_output_dir(output_dir: str = None) -> Path:
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PATHS['ckpt_dir'] / f"simplified_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    return output_dir


def load_full_graph() -> tuple:
    print(f"\n[Loading Full Graph] {PATHS['full_graph_h5']}")

    with h5py.File(PATHS['full_graph_h5'], 'r') as f:
        node_names_raw = f['nodes']['node_names'][:]
        node_names = np.array([
            n.decode('utf-8') if isinstance(n, bytes) else n
            for n in node_names_raw
        ])
        node_features = f['nodes']['node_features'][:]
        feature_dim = node_features.shape[1]

        edge_list = []
        for edge_type in f['edges']:
            src = f['edges'][edge_type]['source'][:]
            tgt = f['edges'][edge_type]['target'][:]
            edge_list.append(np.stack([src, tgt], axis=1))

        if edge_list:
            all_edges = np.concatenate(edge_list, axis=0)
            edge_index = torch.LongTensor(all_edges.T)
        else:
            raise ValueError("No edges found in graph")

        src_list, tgt_list = [], []
        for edge_type in f['edges']:
            src = f['edges'][edge_type]['source'][:]
            tgt = f['edges'][edge_type]['target'][:]
            src_list.extend([src, tgt])
            tgt_list.extend([tgt, src])

        all_src = np.concatenate(src_list)
        all_tgt = np.concatenate(tgt_list)
        adj_matrix = sp.csr_matrix(
            (np.ones(len(all_src), dtype=bool), (all_src, all_tgt)),
            shape=(len(node_names), len(node_names))
        )

    print(f"  - Nodes: {len(node_names)}")
    print(f"  - Edges: {edge_index.shape[1]}")
    print(f"  - Feature dim: {feature_dim}")

    return node_features, edge_index, adj_matrix, node_names, feature_dim


def precompute_disease_features(
    diseases: List[str],
    k_max: int,
    skip_existing: bool = False,
    node2vec_dim: int = 128,
) -> Dict[str, tuple]:
    from data.disease_feature_calculator import compute_disease_features_for_disease

    print(f"\n{'='*60}")
    print("Precomputing Disease Features")
    print(f"{'='*60}")

    output_dir = PATHS['output_dir'] / "disease_features" / "ttd_based"
    output_dir.mkdir(parents=True, exist_ok=True)

    disease_features_dict = {}

    for disease in diseases:
        print(f"\n[Processing] {disease}")
        feature_path = output_dir / f"{disease}_n2v_features.npz"

        if skip_existing and feature_path.exists():
            print(f"  - Loading existing: {feature_path}")
            data = np.load(feature_path)
            disease_features = data['disease_features']
            n_clusters = int(data['n_clusters'])
            disease_features_dict[disease] = (disease_features, n_clusters)
            continue

        disease_features, n_clusters = compute_disease_features_for_disease(
            disease_code=disease,
            full_graph_path=str(PATHS['full_graph_h5']),
            ttd_dir=str(PATHS['ttd_dir']),
            output_dir=str(output_dir),
            k_max=k_max,
            node2vec_dim=node2vec_dim,
            node2vec_gpu=True,
            gmeans_alpha=0.05,
            gmeans_random_state=42,
            skip_existing=False,
            save_subgraph=True
        )
        disease_features_dict[disease] = (disease_features, n_clusters)

    return disease_features_dict


def load_positive_samples() -> List[dict]:
    """Load positive target-pair samples together with disease labels."""
    df_pos = pd.read_csv(PATHS['positive_pairs'])
    samples_with_diseases = []

    for _, row in df_pos.iterrows():
        gene1 = row['genename1']
        gene2 = row['genename2']

        if 'specific_cancer' in row and pd.notna(row['specific_cancer']):
            disease_name_en = row['specific_cancer']
        elif 'fd_name' in row and pd.notna(row['fd_name']):
            disease_name_en = row['fd_name']
        elif 'md_name' in row and pd.notna(row['md_name']):
            disease_name_en = row['md_name']
        else:
            continue

        disease_code = None
        for dc, config in DISEASE_CONFIGS.items():
            if config['name_en'] == disease_name_en:
                disease_code = dc
                break

        if disease_code is None:
            continue

        samples_with_diseases.append({
            'gene1': gene1,
            'gene2': gene2,
            'pair': (gene1, gene2),
            'pair_sorted': tuple(sorted([gene1, gene2])),
            'disease': disease_code,
            'disease_name_en': disease_name_en,
        })

    return samples_with_diseases


def build_cross_disease_mappings(samples_with_diseases: List[dict]) -> tuple:
    """Build helper mappings used by cross-disease negatives and test-set generation."""
    pair_to_diseases = {}
    all_disease_pos_pairs = {}
    disease_ttd_genes_dict = {}

    for sample in samples_with_diseases:
        disease = sample['disease']
        pair = tuple(sorted((sample['gene1'], sample['gene2'])))

        if pair not in pair_to_diseases:
            pair_to_diseases[pair] = set()
        pair_to_diseases[pair].add(disease)

        if disease not in all_disease_pos_pairs:
            all_disease_pos_pairs[disease] = set()
        all_disease_pos_pairs[disease].add(pair)

        if disease not in disease_ttd_genes_dict:
            disease_ttd_genes_dict[disease] = set()
        disease_ttd_genes_dict[disease].add(sample['gene1'])
        disease_ttd_genes_dict[disease].add(sample['gene2'])

    return pair_to_diseases, all_disease_pos_pairs, disease_ttd_genes_dict


def generate_and_save_test_negatives(
    splits: dict,
    node_names: np.ndarray,
    disease_features_dict: dict,
    all_disease_pos_pairs: Dict[str, set],
    pair_to_diseases: Dict[tuple, set],
    disease_ttd_genes_dict: Dict[str, set],
    output_dir: Path,
    name_to_idx: Dict[str, int],
    neg_strategies: List[str],
    neg_ratio: float = 1.0,
    random_state: int = 42,
) -> Dict[str, dict]:
    """Generate external test sets for the supported evaluation strategies."""
    print(f"\n{'='*60}")
    print("Generating External Test Sets")
    print(f"{'='*60}")

    test_sets_dir = output_dir / 'data_splits' / 'test_sets'
    test_sets_dir.mkdir(parents=True, exist_ok=True)

    def map_pair_to_index(pair):
        idx1 = name_to_idx.get(pair[0])
        idx2 = name_to_idx.get(pair[1])
        if idx1 is None or idx2 is None:
            return None
        return np.array([idx1, idx2], dtype=np.int64)

    test_pairs = splits['external_test']['all_pairs']
    test_diseases = splits['external_test']['all_diseases']

    if len(test_pairs) == 0:
        print("[Warning] No external test positives found")
        return {}

    print(f"[External Test Positives] {len(test_pairs)}")

    all_train_val_indices = set()
    all_train_val_gene_pairs = set()
    for fold_data in splits['cv_splits']:
        for pair in fold_data['train_pairs']:
            all_train_val_gene_pairs.add(tuple(sorted((str(pair[0]), str(pair[1])))))
            idx_pair = map_pair_to_index(pair)
            if idx_pair is not None:
                all_train_val_indices.add(tuple(sorted(idx_pair)))
        for pair in fold_data['val_pairs']:
            all_train_val_gene_pairs.add(tuple(sorted((str(pair[0]), str(pair[1])))))
            idx_pair = map_pair_to_index(pair)
            if idx_pair is not None:
                all_train_val_indices.add(tuple(sorted(idx_pair)))

    print(f"[Excluded Train/Val Pairs] {len(all_train_val_indices)}")
    external_test_gene_pairs = {
        tuple(sorted((str(pair[0]), str(pair[1]))))
        for pair in test_pairs
    }

    test_pairs_by_disease = defaultdict(list)
    for pair, disease in zip(test_pairs, test_diseases):
        test_pairs_by_disease[disease].append(pair)

    global_cluster_labels = None
    cluster_path = PATHS.get('global_clusters')
    if cluster_path and Path(cluster_path).exists():
        global_cluster_labels = np.load(cluster_path, allow_pickle=True)['cluster_labels']
        print(f"[Loaded Global Clusters] {cluster_path}")
    else:
        print(f"[Warning] Global clusters not found: {cluster_path}")

    test_sets_by_strategy = {}

    for strategy in neg_strategies:
        print(f"\n[Test Strategy] {strategy}")

        strategy_neg_pairs_by_disease = {}
        strategy_pos_indices_by_disease = {}

        for disease, disease_test_pairs in test_pairs_by_disease.items():
            disease_pos_indices = []
            for pair in disease_test_pairs:
                idx_pair = map_pair_to_index(pair)
                if idx_pair is not None:
                    disease_pos_indices.append(idx_pair)

            disease_pos_indices = (
                np.array(disease_pos_indices, dtype=np.int64)
                if disease_pos_indices
                else np.empty((0, 2), dtype=np.int64)
            )
            strategy_pos_indices_by_disease[disease] = disease_pos_indices

            neg_sampler = NegativeSampler(
                strategy=strategy,
                global_cluster_labels=global_cluster_labels if strategy == 'hard' else None,
                current_disease=disease,
                all_disease_pos_pairs=all_disease_pos_pairs,
                pair_to_diseases=pair_to_diseases,
                node_names=node_names,
                disease_ttd_genes=disease_ttd_genes_dict.get(disease),
            )

            exclusion_pairs = (
                all_train_val_gene_pairs
                if strategy == 'cross_disease'
                else all_train_val_indices
            )
            disease_neg_pairs = neg_sampler.generate_negatives(
                pos_pairs=disease_pos_indices,
                ratio=neg_ratio,
                exclude_pairs=exclusion_pairs,
                disease_ttd_genes=disease_ttd_genes_dict.get(disease),
            )

            if len(disease_neg_pairs) > 0:
                sample = disease_neg_pairs[0]
                if isinstance(sample[0], str):
                    valid_neg_pairs = []
                    for pair in disease_neg_pairs:
                        idx_pair = map_pair_to_index(pair)
                        if idx_pair is not None:
                            valid_neg_pairs.append(idx_pair)
                    disease_neg_pairs = (
                        np.array(valid_neg_pairs, dtype=np.int64)
                        if valid_neg_pairs
                        else np.empty((0, 2), dtype=np.int64)
                    )
                else:
                    disease_neg_pairs = np.array(disease_neg_pairs, dtype=np.int64)
            else:
                disease_neg_pairs = np.empty((0, 2), dtype=np.int64)

            if strategy == 'cross_disease':
                generated_gene_pairs = {
                    tuple(sorted((str(node_names[pair[0]]), str(node_names[pair[1]]))))
                    for pair in disease_neg_pairs
                }
                outside_test_partition = generated_gene_pairs - external_test_gene_pairs
                if outside_test_partition:
                    raise RuntimeError(
                        f"External test negatives contain {len(outside_test_partition)} "
                        "target pairs from the train/validation partition"
                    )

            strategy_neg_pairs_by_disease[disease] = disease_neg_pairs

        total_pos = sum(len(pairs) for pairs in strategy_pos_indices_by_disease.values())
        total_neg = sum(len(pairs) for pairs in strategy_neg_pairs_by_disease.values())
        print(f"  Positives: {total_pos}, Negatives: {total_neg}")

        all_pairs_list = []
        all_labels_list = []
        all_diseases_list = []
        all_gene_pairs_list = []

        for disease, disease_pos_indices in strategy_pos_indices_by_disease.items():
            if len(disease_pos_indices) == 0:
                continue

            for idx_pair in disease_pos_indices:
                all_pairs_list.append(idx_pair)
                all_labels_list.append(1)
                all_diseases_list.append(disease)
                all_gene_pairs_list.append((node_names[idx_pair[0]], node_names[idx_pair[1]]))

            for idx_pair in strategy_neg_pairs_by_disease[disease]:
                all_pairs_list.append(idx_pair)
                all_labels_list.append(0)
                all_diseases_list.append(disease)
                all_gene_pairs_list.append((node_names[idx_pair[0]], node_names[idx_pair[1]]))

        all_pairs = (
            np.array(all_pairs_list, dtype=np.int64)
            if all_pairs_list
            else np.empty((0, 2), dtype=np.int64)
        )
        all_labels = np.array(all_labels_list, dtype=np.int64)
        all_diseases_arr = np.array(all_diseases_list, dtype=object)

        csv_df = pd.DataFrame([
            {
                'gene1': all_gene_pairs_list[i][0],
                'gene2': all_gene_pairs_list[i][1],
                'target1_idx': int(all_pairs[i][0]),
                'target2_idx': int(all_pairs[i][1]),
                'disease': all_diseases_arr[i],
                'label': int(all_labels[i]),
            }
            for i in range(len(all_pairs))
        ])
        csv_path = test_sets_dir / f'test_{strategy}.csv'
        csv_df.to_csv(csv_path, index=False)

        npz_path = test_sets_dir / f'test_{strategy}.npz'
        np.savez(
            npz_path,
            pairs=all_pairs,
            labels=all_labels,
            diseases=all_diseases_arr,
            gene_pairs=np.array(all_gene_pairs_list, dtype=object),
        )

        print(f"  Saved: {csv_path}")
        print(f"  Saved: {npz_path}")

        test_sets_by_strategy[strategy] = {
            'pairs': all_pairs,
            'labels': all_labels,
            'diseases': all_diseases_arr,
            'gene_pairs': all_gene_pairs_list,
        }

    metadata_path = test_sets_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(
            {
                'neg_strategies': neg_strategies,
                'neg_ratio': neg_ratio,
                'random_state': random_state,
                'created_at': datetime.now().isoformat(),
                'n_test_positives': int(len(test_pairs)),
                'n_train_val_excluded': int(len(all_train_val_indices)),
            },
            f,
            indent=2,
        )
    print(f"\n[Saved] {metadata_path}")

    return test_sets_by_strategy


def load_precomputed_test_set(
    output_dir: Path,
    strategy: str,
    node_features: np.ndarray,
    edge_index: torch.Tensor,
    disease_features_dict: dict,
    disease_to_idx: dict,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
) -> DataLoader:
    """Load a previously generated external test set."""
    npz_path = output_dir / 'data_splits' / 'test_sets' / f'test_{strategy}.npz'

    if not npz_path.exists():
        print(f"[Error] Missing precomputed test set: {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)
    pairs = data['pairs']
    labels = data['labels']
    diseases = data['diseases']

    print(
        f"[Loaded Test Set] {strategy}: {len(pairs)} samples "
        f"(pos={int(np.sum(labels))}, neg={int(len(labels) - np.sum(labels))})"
    )

    disease_features_list = []
    n_clusters_list = []
    disease_indices_list = []

    default_disease = list(disease_features_dict.keys())[0]
    default_feat, default_n_clusters = disease_features_dict[default_disease]

    for disease in diseases:
        if disease in disease_features_dict:
            disease_feat, disease_n_clusters = disease_features_dict[disease]
            disease_idx = disease_to_idx.get(disease, -1)
        else:
            disease_feat, disease_n_clusters = default_feat, default_n_clusters
            disease_idx = -1

        disease_features_list.append(disease_feat)
        n_clusters_list.append(disease_n_clusters)
        disease_indices_list.append(disease_idx)

    test_dataset = TargetSynergyDatasetWithPerSampleFeatures(
        pair_indices=pairs,
        labels=labels,
        node_features=node_features,
        edge_index=edge_index,
        disease_features_list=disease_features_list,
        n_clusters_list=n_clusters_list,
        disease_indices_list=disease_indices_list,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
    )


class TargetSynergyDatasetWithPerSampleFeatures(Dataset):
    def __init__(
        self,
        pair_indices,
        labels,
        node_features,
        edge_index,
        disease_features_list,
        n_clusters_list,
        disease_indices_list=None
    ):
        self.pair_indices = pair_indices
        self.labels = labels
        self.node_features = torch.FloatTensor(node_features)
        self.disease_features_list = disease_features_list
        self.n_clusters_list = torch.LongTensor(n_clusters_list)
        self.edge_index = edge_index

        if disease_indices_list is not None:
            self.disease_indices_list = torch.LongTensor(disease_indices_list)
        else:
            self.disease_indices_list = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        target1_idx, target2_idx = self.pair_indices[idx]
        label = self.labels[idx]

        disease_feat = self.disease_features_list[idx]
        if not isinstance(disease_feat, torch.Tensor):
            disease_feat = torch.FloatTensor(disease_feat)

        sample = {
            'target1_idx': torch.LongTensor([target1_idx]),
            'target2_idx': torch.LongTensor([target2_idx]),
            'disease_features': disease_feat,
            'n_clusters': self.n_clusters_list[idx],
            'label': torch.FloatTensor([label]),
            'node_features': self.node_features,
            'edge_index': self.edge_index,
        }

        if self.disease_indices_list is not None:
            sample['disease_idx'] = torch.LongTensor([self.disease_indices_list[idx]])
        else:
            sample['disease_idx'] = torch.LongTensor([-1])

        return sample


def create_datasets(
    splits: dict,
    disease_features_dict: dict,
    node_features: np.ndarray,
    edge_index: torch.Tensor,
    node_names: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path = None
) -> tuple:
    print(f"\n[Creating Datasets]")

    neg_ratio = args.neg_ratio
    neg_strategy = 'cross_disease'  # 固定使用跨疾病负采样

    samples_with_diseases = load_positive_samples()
    print(f"[Mapped] {len(samples_with_diseases)} samples")

    name_to_idx = {name: i for i, name in enumerate(node_names)}

    def map_pair_to_index(pair):
        # 确保 pair 元素是字符串（处理 numpy 字符串数组）
        gene1, gene2 = str(pair[0]), str(pair[1])
        idx1 = name_to_idx.get(gene1)
        idx2 = name_to_idx.get(gene2)
        if idx1 is None or idx2 is None:
            return None
        return (idx1, idx2)

    print(f"\n[Cross-disease] Precomputing pair-disease mapping")
    pair_to_diseases_global, all_disease_pos_pairs, disease_ttd_genes_dict = (
        build_cross_disease_mappings(samples_with_diseases)
    )

    neg_sampler = NegativeSampler(
        strategy=neg_strategy,
        node_names=node_names,
        current_disease=None,
        all_disease_pos_pairs=all_disease_pos_pairs,
        pair_to_diseases=pair_to_diseases_global
    )

    available_diseases = list(disease_features_dict.keys())
    disease_to_idx = {disease: idx for idx, disease in enumerate(sorted(available_diseases))}
    idx_to_disease = {idx: disease for disease, idx in disease_to_idx.items()}

    # Cross-disease candidates use gene-name pairs, so exclusions must use the same representation.
    external_test_set = set()
    if 'external_test' in splits and 'all_pairs' in splits['external_test']:
        for pair in splits['external_test']['all_pairs']:
            external_test_set.add(tuple(sorted((str(pair[0]), str(pair[1])))))
    print(f"[External test pairs for exclusion] {len(external_test_set)}")

    cv_datasets = []

    for fold_idx, fold_data in enumerate(splits['cv_splits']):
        train_pairs = fold_data['train_pairs']
        val_pairs = fold_data['val_pairs']

        # 过滤掉 None 值并指定 dtype
        train_indices_list = [map_pair_to_index(pair) for pair in train_pairs]
        train_indices_list = [x for x in train_indices_list if x is not None]
        train_indices = np.array(train_indices_list, dtype=np.int64)

        val_indices_list = [map_pair_to_index(pair) for pair in val_pairs]
        val_indices_list = [x for x in val_indices_list if x is not None]
        val_indices = np.array(val_indices_list, dtype=np.int64)

        train_pairs_by_disease = defaultdict(list)
        val_pairs_by_disease = defaultdict(list)

        if 'train_pair_diseases' in fold_data and 'val_pair_diseases' in fold_data:
            train_pair_diseases = fold_data['train_pair_diseases']
            val_pair_diseases = fold_data['val_pair_diseases']

            for pair, disease in zip(train_pairs, train_pair_diseases):
                if disease in disease_features_dict:
                    train_pairs_by_disease[disease].append(pair)

            for pair, disease in zip(val_pairs, val_pair_diseases):
                if disease in disease_features_dict:
                    val_pairs_by_disease[disease].append(pair)

        train_neg_pairs_by_disease = {}
        val_neg_pairs_by_disease = {}

        def convert_neg_pairs_to_indices(neg_pairs):
            """将基因名负样本对转换为整数索引，确保形状始终为 (n, 2)"""
            if len(neg_pairs) == 0:
                return np.array([]).reshape(0, 2).astype(np.int64)

            sample = neg_pairs[0] if isinstance(neg_pairs, list) else neg_pairs[0]

            if isinstance(sample[0], str):
                # 基因名 → 索引
                valid_pairs = []
                for pair in neg_pairs:
                    gene1, gene2 = pair
                    if gene1 in name_to_idx and gene2 in name_to_idx:
                        valid_pairs.append((name_to_idx[gene1], name_to_idx[gene2]))
                return np.array(valid_pairs) if valid_pairs else np.array([]).reshape(0, 2).astype(np.int64)
            else:
                # 已经是整数索引对，确保 2D
                arr = np.array(neg_pairs, dtype=np.int64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2)
                return arr

        for disease, disease_train_pairs in train_pairs_by_disease.items():
            # 过滤掉 None 值并指定 dtype
            disease_train_indices_list = [map_pair_to_index(pair) for pair in disease_train_pairs]
            disease_train_indices_list = [x for x in disease_train_indices_list if x is not None]
            disease_train_indices = np.array(disease_train_indices_list, dtype=np.int64)
            if disease_train_indices.ndim == 1:
                disease_train_indices = disease_train_indices.reshape(-1, 2)

            disease_exclude_pairs = external_test_set.copy()
            disease_ttd_genes = disease_ttd_genes_dict.get(disease, None)
            neg_sampler.current_disease = disease

            disease_neg_pairs = neg_sampler.generate_negatives(
                pos_pairs=disease_train_indices,
                ratio=neg_ratio,
                exclude_pairs=disease_exclude_pairs,
                disease_ttd_genes=disease_ttd_genes
            )
            disease_neg_pairs = convert_neg_pairs_to_indices(disease_neg_pairs)
            train_neg_pairs_by_disease[disease] = disease_neg_pairs

        for disease, disease_val_pairs in val_pairs_by_disease.items():
            # 过滤掉 None 值并指定 dtype
            disease_val_indices_list = [map_pair_to_index(pair) for pair in disease_val_pairs]
            disease_val_indices_list = [x for x in disease_val_indices_list if x is not None]
            disease_val_indices = np.array(disease_val_indices_list, dtype=np.int64)
            if disease_val_indices.ndim == 1:
                disease_val_indices = disease_val_indices.reshape(-1, 2)

            disease_exclude_pairs = external_test_set.copy()

            disease_ttd_genes = disease_ttd_genes_dict.get(disease, None)
            neg_sampler.current_disease = disease

            disease_neg_pairs = neg_sampler.generate_negatives(
                pos_pairs=disease_val_indices,
                ratio=neg_ratio,
                exclude_pairs=disease_exclude_pairs,
                disease_ttd_genes=disease_ttd_genes
            )
            disease_neg_pairs = convert_neg_pairs_to_indices(disease_neg_pairs)
            val_neg_pairs_by_disease[disease] = disease_neg_pairs

        train_all_pairs_list = []
        train_all_labels_list = []
        train_disease_features_list = []
        train_n_clusters_list = []
        train_disease_indices_list = []

        for disease, disease_train_pairs in train_pairs_by_disease.items():
            # 过滤掉 None 值（基因不在图中的情况）
            disease_train_indices_list = [map_pair_to_index(pair) for pair in disease_train_pairs]
            disease_train_indices_list = [x for x in disease_train_indices_list if x is not None]
            disease_train_indices = np.array(disease_train_indices_list, dtype=np.int64)
            if disease_train_indices.ndim == 1:
                disease_train_indices = disease_train_indices.reshape(-1, 2)

            # 负样本对已在 convert_neg_pairs_to_indices 中转换为 (n, 2) 整数数组
            disease_neg_indices = train_neg_pairs_by_disease[disease]

            disease_feat, disease_n_clusters = disease_features_dict[disease]
            disease_idx = disease_to_idx.get(disease, -1)

            for _ in disease_train_indices:
                train_disease_features_list.append(disease_feat)
                train_n_clusters_list.append(disease_n_clusters)
                train_disease_indices_list.append(disease_idx)

            for _ in disease_neg_indices:
                train_disease_features_list.append(disease_feat)
                train_n_clusters_list.append(disease_n_clusters)
                train_disease_indices_list.append(disease_idx)

            disease_all_pairs = np.concatenate([disease_train_indices, disease_neg_indices], axis=0)
            disease_all_labels = np.concatenate([
                np.ones(len(disease_train_indices)),
                np.zeros(len(disease_neg_indices))
            ])

            train_all_pairs_list.append(disease_all_pairs)
            train_all_labels_list.append(disease_all_labels)

        val_all_pairs_list = []
        val_all_labels_list = []
        val_disease_features_list = []
        val_n_clusters_list = []
        val_disease_indices_list = []

        for disease, disease_val_pairs in val_pairs_by_disease.items():
            # 过滤掉 None 值（基因不在图中的情况）
            disease_val_indices_list = [map_pair_to_index(pair) for pair in disease_val_pairs]
            disease_val_indices_list = [x for x in disease_val_indices_list if x is not None]
            disease_val_indices = np.array(disease_val_indices_list, dtype=np.int64)
            if disease_val_indices.ndim == 1:
                disease_val_indices = disease_val_indices.reshape(-1, 2)

            # 负样本对已在 convert_neg_pairs_to_indices 中转换为 (n, 2) 整数数组
            disease_neg_indices = val_neg_pairs_by_disease[disease]

            disease_feat, disease_n_clusters = disease_features_dict[disease]
            disease_idx = disease_to_idx.get(disease, -1)

            for _ in disease_val_indices:
                val_disease_features_list.append(disease_feat)
                val_n_clusters_list.append(disease_n_clusters)
                val_disease_indices_list.append(disease_idx)

            for _ in disease_neg_indices:
                val_disease_features_list.append(disease_feat)
                val_n_clusters_list.append(disease_n_clusters)
                val_disease_indices_list.append(disease_idx)

            disease_all_pairs = np.concatenate([disease_val_indices, disease_neg_indices], axis=0)
            disease_all_labels = np.concatenate([
                np.ones(len(disease_val_indices)),
                np.zeros(len(disease_neg_indices))
            ])

            val_all_pairs_list.append(disease_all_pairs)
            val_all_labels_list.append(disease_all_labels)

        train_all_pairs = np.concatenate(train_all_pairs_list, axis=0)
        train_all_labels = np.concatenate(train_all_labels_list, axis=0)
        val_all_pairs = np.concatenate(val_all_pairs_list, axis=0)
        val_all_labels = np.concatenate(val_all_labels_list, axis=0)

        print(f"  Fold {fold_data['fold']}: Train={len(train_all_pairs)}, Val={len(val_all_pairs)}")

        train_dataset = TargetSynergyDatasetWithPerSampleFeatures(
            pair_indices=train_all_pairs,
            labels=train_all_labels,
            node_features=node_features,
            edge_index=edge_index,
            disease_features_list=train_disease_features_list,
            n_clusters_list=train_n_clusters_list,
            disease_indices_list=train_disease_indices_list
        )

        val_dataset = TargetSynergyDatasetWithPerSampleFeatures(
            pair_indices=val_all_pairs,
            labels=val_all_labels,
            node_features=node_features,
            edge_index=edge_index,
            disease_features_list=val_disease_features_list,
            n_clusters_list=val_n_clusters_list,
            disease_indices_list=val_disease_indices_list
        )

        cv_datasets.append({
            'fold': fold_data['fold'],
            'train': train_dataset,
            'val': val_dataset,
        })

    print(f"\n[Dataset Created]")
    print(f"  - CV folds: {len(cv_datasets)}")
    print(f"  - External test sets: generated separately (hard, cross_disease)")

    return cv_datasets, idx_to_disease


def main():
    args = parse_args()
    train_neg_strategy = 'cross_disease'
    test_neg_strategies = ['hard', 'cross_disease']

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.diseases is not None:
        diseases = args.diseases
    else:
        diseases = TARGETPAIR_SOLID_TUMORS

    print(f"\n{'='*60}")
    print(f"TargetSynergy Training (Simplified)")
    print(f"{'='*60}")
    print(f"[Diseases] {len(diseases)}: {', '.join(diseases)}")
    print(f"[Device] {args.device}")
    print(f"[Fixed Config]")
    print(f"  - Training negative strategy: {train_neg_strategy}")
    print(f"  - External test strategies: {', '.join(test_neg_strategies)}")
    print(f"  - JK layers: {args.jk_num_layers}")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Output dim: {args.output_dim}")
    print(f"  - Attention heads: {args.num_attention_heads}")
    print(f"  - Recurrent iterations: {args.num_recurrent_iterations}")
    print(f"  - Dropout: {args.dropout}")

    output_dir = setup_output_dir(args.output_dir)
    print(f"[Output Dir] {output_dir}")

    config = vars(args).copy()
    config.update({
        'neg_strategy': train_neg_strategy,
        'test_neg_strategies': test_neg_strategies,
        'eval_mode': 'stratified_kfold',
        'split_mode': 'sample_level_stratified',
    })
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[Config Saved] {config_path}")

    node_features, edge_index, adj_matrix, node_names, feature_dim = load_full_graph()

    disease_features_dict = precompute_disease_features(
        diseases=diseases,
        k_max=args.disease_k_max,
        skip_existing=not args.force_recompute,
        node2vec_dim=128,
    )

    print(f"\n{'='*60}")
    print("Creating Data Splits")
    print(f"{'='*60}")

    splits = create_multi_disease_splits_v2(
        positive_pairs_csv=PATHS['positive_pairs'],
        diseases=diseases,
        eval_mode='stratified_kfold',
        split_mode='sample_level_stratified',
        n_folds=args.n_folds,
        test_ratio=TRAINING_CONFIG['external_test_ratio'],
    )

    samples_with_diseases = load_positive_samples()
    pair_to_diseases, all_disease_pos_pairs, disease_ttd_genes_dict = (
        build_cross_disease_mappings(samples_with_diseases)
    )
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    generate_and_save_test_negatives(
        splits=splits,
        node_names=node_names,
        disease_features_dict=disease_features_dict,
        all_disease_pos_pairs=all_disease_pos_pairs,
        pair_to_diseases=pair_to_diseases,
        disease_ttd_genes_dict=disease_ttd_genes_dict,
        output_dir=output_dir,
        name_to_idx=name_to_idx,
        neg_strategies=test_neg_strategies,
        neg_ratio=args.neg_ratio,
        random_state=args.seed,
    )

    cv_datasets, idx_to_disease = create_datasets(
        splits=splits,
        disease_features_dict=disease_features_dict,
        node_features=node_features,
        edge_index=edge_index,
        node_names=node_names,
        args=args,
        output_dir=output_dir,
    )

    def create_dataloader(dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False,
        )

    all_results = []

    for fold_idx, fold_data in enumerate(cv_datasets):
        if args.fold_index is not None and fold_idx != args.fold_index:
            continue

        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{len(cv_datasets)}")
        print(f"{'='*60}")

        train_loader = create_dataloader(fold_data['train'], args.batch_size, shuffle=True)
        val_loader = create_dataloader(fold_data['val'], args.batch_size, shuffle=False)

        model = TargetSynergyModel(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            jk_num_layers=args.jk_num_layers,
            num_attention_heads=args.num_attention_heads,
            num_recurrent_iterations=args.num_recurrent_iterations,
            dropout=args.dropout,
        ).to(args.device)

        print(f"[Model] Created successfully")

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            save_dir=output_dir / f'fold_{fold_idx}',
            early_stopping_patience=TRAINING_CONFIG['patience'],
            alpha_orth=args.alpha_orth,
            beta_con=args.beta_con,
            temperature=args.temperature,
            optim_metric=args.optim_metric,
            gamma_adversarial=args.gamma_adversarial,
            adversarial_temperature=args.adversarial_temperature,
        )

        history_path = output_dir / f'fold_{fold_idx}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        best_model_path = output_dir / f'fold_{fold_idx}' / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, weights_only=False)
            print(f"\n[Best Model] Epoch: {checkpoint.get('epoch', 'N/A')}")

        all_results.append({
            'fold': fold_idx,
            'best_val_auc': max(history['val_auc']) if history['val_auc'] else 0,
            'best_val_auprc': max(history['val_auprc']) if history['val_auprc'] else 0,
        })

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    if not all_results:
        print("[Warning] No folds were trained.")
        return

    for result in all_results:
        print(f"  Fold {result['fold'] + 1}: Val AUC={result['best_val_auc']:.4f}, Val AUPRC={result['best_val_auprc']:.4f}")

    if len(all_results) > 1:
        mean_auc = np.mean([r['best_val_auc'] for r in all_results])
        std_auc = np.std([r['best_val_auc'] for r in all_results])
        mean_auprc = np.mean([r['best_val_auprc'] for r in all_results])
        std_auprc = np.std([r['best_val_auprc'] for r in all_results])
        print(f"\n  Mean Val AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  Mean Val AUPRC: {mean_auprc:.4f} +/- {std_auprc:.4f}")

    summary_path = output_dir / 'results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'all_results': all_results,
            'mean_val_auc': float(mean_auc) if len(all_results) > 1 else all_results[0]['best_val_auc'],
            'mean_val_auprc': float(mean_auprc) if len(all_results) > 1 else all_results[0]['best_val_auprc'],
        }, f, indent=2)
    print(f"\n[Results Saved] {summary_path}")

    if len(all_results) == args.n_folds:
        print(f"\n{'='*80}")
        print("External Test Evaluation")
        print(f"{'='*80}")

        disease_to_idx = {
            disease: idx
            for idx, disease in enumerate(sorted(disease_features_dict.keys()))
        }
        all_strategy_results = {}

        for test_neg_strategy in test_neg_strategies:
            print(f"\n{'='*60}")
            print(f"Testing Strategy: {test_neg_strategy}")
            print(f"{'='*60}")

            test_loader = load_precomputed_test_set(
                output_dir=output_dir,
                strategy=test_neg_strategy,
                node_features=node_features,
                edge_index=edge_index,
                disease_features_dict=disease_features_dict,
                disease_to_idx=disease_to_idx,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )

            if test_loader is None:
                continue

            strategy_test_results = []
            for fold_idx in range(args.n_folds):
                checkpoint_path = output_dir / f'fold_{fold_idx}' / 'best_model.pt'
                if not checkpoint_path.exists():
                    print(f"[Warning] Missing checkpoint: {checkpoint_path}")
                    continue

                checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                test_model = TargetSynergyModel(
                    input_dim=feature_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.output_dim,
                    jk_num_layers=args.jk_num_layers,
                    num_attention_heads=args.num_attention_heads,
                    num_recurrent_iterations=args.num_recurrent_iterations,
                    dropout=args.dropout,
                ).to(args.device)
                test_model.load_state_dict(state_dict)
                test_model.eval()

                test_metrics, detailed_results = test(
                    model=test_model,
                    dataloader=test_loader,
                    device=args.device,
                    return_details=True,
                    node_names=node_names,
                    idx_to_disease=idx_to_disease,
                )

                pred_dir = output_dir / f'predictions_{test_neg_strategy}'
                pred_dir.mkdir(exist_ok=True)
                pred_path = pred_dir / f'fold_{fold_idx}_predictions.csv'
                pd.DataFrame(detailed_results).to_csv(pred_path, index=False)
                print(f"  Fold {fold_idx}: saved predictions to {pred_path}")

                strategy_test_results.append({
                    'fold': fold_idx,
                    'metrics': {
                        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                        for key, value in test_metrics.items()
                    },
                })

                print(
                    f"  Fold {fold_idx}: "
                    f"AUC={test_metrics.get('auc_roc', 0):.4f}, "
                    f"AUPRC={test_metrics.get('auc_pr', 0):.4f}, "
                    f"MCC={test_metrics.get('mcc', 0):.4f}, "
                    f"F1={test_metrics.get('f1', 0):.4f}"
                )

            if not strategy_test_results:
                continue

            all_strategy_results[test_neg_strategy] = strategy_test_results

            avg_metrics = {}
            for key in strategy_test_results[0]['metrics'].keys():
                values = [result['metrics'][key] for result in strategy_test_results]
                avg_metrics[key] = float(np.mean(values))
                std_val = float(np.std(values))
                print(f"  {key}: {avg_metrics[key]:.4f} +/- {std_val:.4f}")

            strategy_result_path = output_dir / f'test_results_{test_neg_strategy}.json'
            with open(strategy_result_path, 'w') as f:
                json.dump(
                    {
                        'test_neg_strategy': test_neg_strategy,
                        'train_neg_strategy': train_neg_strategy,
                        'avg_metrics': avg_metrics,
                        'fold_results': strategy_test_results,
                    },
                    f,
                    indent=2,
                )
            print(f"  Saved: {strategy_result_path}")

        if all_strategy_results:
            summary_table = []
            for strategy, results in all_strategy_results.items():
                avg_metrics = {}
                for key in results[0]['metrics'].keys():
                    values = [result['metrics'][key] for result in results]
                    avg_metrics[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }

                summary_table.append({
                    'test_neg_strategy': strategy,
                    'auc_roc': avg_metrics.get('auc_roc', {}),
                    'auc_pr': avg_metrics.get('auc_pr', {}),
                    'mcc': avg_metrics.get('mcc', {}),
                    'f1': avg_metrics.get('f1', {}),
                    'accuracy': avg_metrics.get('accuracy', {}),
                })

            test_summary_path = output_dir / 'test_summary_all_strategies.json'
            with open(test_summary_path, 'w') as f:
                json.dump(
                    {
                        'train_neg_strategy': train_neg_strategy,
                        'strategies_tested': list(all_strategy_results.keys()),
                        'summary_table': summary_table,
                        'detailed_results': all_strategy_results,
                    },
                    f,
                    indent=2,
                )
            print(f"\n[Results Saved] {test_summary_path}")
    else:
        print(
            f"\n[Skip] Full external test evaluation requires all folds. "
            f"Completed {len(all_results)}/{args.n_folds} folds."
        )


if __name__ == '__main__':
    main()
