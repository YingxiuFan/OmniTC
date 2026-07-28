# -*- coding: utf-8 -*-
"""
TargetSynergy Dataset Module
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal, Any
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp

from config import PATHS, TRAINING_CONFIG


class TargetSynergyDataset(Dataset):
    def __init__(
        self,
        pair_indices: np.ndarray,
        labels: np.ndarray,
        node_features: np.ndarray,
        adj_matrix: sp.csr_matrix,
        cluster_labels: Optional[np.ndarray] = None,
        disease_name: Optional[str] = None,
        disease_features: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None
    ):
        self.pair_indices = pair_indices
        self.labels = labels
        self.node_features = torch.FloatTensor(node_features)
        self.adj_matrix = adj_matrix
        self.cluster_labels = cluster_labels
        self.disease_name = disease_name
        self.disease_features = torch.FloatTensor(disease_features) if disease_features is not None else None
        self.n_clusters = n_clusters
        self.edge_index = self._csr_to_edge_index(adj_matrix)

    def _csr_to_edge_index(self, adj_matrix: sp.csr_matrix) -> torch.Tensor:
        adj_coo = adj_matrix.tocoo()
        edge_index = torch.stack([torch.LongTensor(adj_coo.row), torch.LongTensor(adj_coo.col)], dim=0)
        return edge_index

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target1_idx, target2_idx = self.pair_indices[idx]
        label = self.labels[idx]

        sample = {
            'target1_idx': torch.LongTensor([target1_idx]),
            'target2_idx': torch.LongTensor([target2_idx]),
            'disease_features': self.disease_features,
            'n_clusters': torch.LongTensor([self.n_clusters]) if self.n_clusters is not None else None,
            'label': torch.FloatTensor([label]),
            'node_features': self.node_features,
            'edge_index': self.edge_index,
        }
        return sample


class NegativeSampler:
    def __init__(
        self,
        strategy: Literal['cross_disease', 'hard'] = 'cross_disease',
        global_cluster_labels: Optional[np.ndarray] = None,
        disease_ttd_genes: Optional[set] = None,
        node_names: Optional[np.ndarray] = None,
        current_disease: Optional[str] = None,
        all_disease_pos_pairs: Optional[Dict[str, set]] = None,
        pair_to_diseases: Optional[Dict[tuple, set]] = None
    ):
        self.strategy = strategy
        self.global_cluster_labels = global_cluster_labels
        self.disease_ttd_genes = disease_ttd_genes
        self.node_names = node_names
        self.current_disease = current_disease
        self.all_disease_pos_pairs = all_disease_pos_pairs
        self.pair_to_diseases = pair_to_diseases

        if node_names is not None:
            self.gene_to_idx = {name: idx for idx, name in enumerate(node_names)}
        else:
            self.gene_to_idx = None

    def generate_negatives(
        self,
        pos_pairs: np.ndarray,
        node_indices: np.ndarray = None,
        ratio: float = 1.0,
        exclude_pairs: set = None,
        disease_ttd_genes: Optional[set] = None
    ) -> np.ndarray:
        num_neg = int(len(pos_pairs) * ratio)

        current_ttd_genes = disease_ttd_genes if disease_ttd_genes is not None else self.disease_ttd_genes

        if self.strategy == 'cross_disease':
            pos_set = self._to_gene_pair_set(pos_pairs)
            exclude_set = self._to_gene_pair_set(exclude_pairs)
            current_disease_genes = current_ttd_genes if current_ttd_genes is not None else self.disease_ttd_genes
            neg_set = self._generate_cross_disease_negatives(
                num_neg=num_neg,
                pos_set=pos_set,
                exclude_pairs=exclude_set,
                current_disease_genes=current_disease_genes
            )
            return np.array(list(neg_set))

        elif self.strategy == 'hard':
            pos_set = set(tuple(p) for p in pos_pairs)
            if exclude_pairs is not None:
                pos_set = pos_set.union(exclude_pairs)
            neg_set = self._generate_hard_negatives(
                num_neg=num_neg, pos_set=pos_set, exclude_pairs=exclude_pairs
            )
            return np.array(list(neg_set))

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _to_gene_pair_set(self, pairs) -> set:
        """Normalize gene-name or node-index pairs to sorted gene-name tuples."""
        normalized = set()
        if pairs is None:
            return normalized

        for pair in pairs:
            first, second = pair[0], pair[1]
            if isinstance(first, (str, np.str_)):
                gene1, gene2 = str(first), str(second)
            else:
                if self.node_names is None:
                    raise ValueError(
                        "node_names is required to compare index pairs with cross-disease gene pairs"
                    )
                gene1 = str(self.node_names[int(first)])
                gene2 = str(self.node_names[int(second)])
            normalized.add(tuple(sorted((gene1, gene2))))

        return normalized

    def _generate_cross_disease_negatives(
        self,
        num_neg: int,
        pos_set: set,
        exclude_pairs: set = None,
        current_disease_genes: set = None
    ) -> set:
        """
        生成跨疾病负样本

        核心逻辑：
        1. 从其他疾病的正样本中选择
        2. 只选择"疾病特异性正样本"（只在一个疾病中的靶点对）
        3. 排除当前疾病的正样本
        4. 基因重叠约束：候选负样本至少有一个基因在当前疾病的靶点集中（可选）
        """
        import random

        if self.current_disease is None or self.all_disease_pos_pairs is None or self.pair_to_diseases is None:
            print("  ⚠️  cross_disease 策略缺少必要参数，返回空集")
            return set()

        neg_set = set()

        candidate_pairs_with_overlap = []
        candidate_pairs_no_overlap = []

        for disease, pairs in self.all_disease_pos_pairs.items():
            if disease == self.current_disease:
                continue
            for pair in pairs:
                if pair in self.pair_to_diseases:
                    diseases_of_pair = self.pair_to_diseases[pair]
                    if len(diseases_of_pair) == 1 and pair not in pos_set:
                        if pair not in exclude_pairs:
                            if current_disease_genes is not None:
                                gene1, gene2 = pair
                                if gene1 in current_disease_genes or gene2 in current_disease_genes:
                                    candidate_pairs_with_overlap.append(pair)
                                else:
                                    candidate_pairs_no_overlap.append(pair)
                            else:
                                candidate_pairs_with_overlap.append(pair)

        # 优先从有基因重叠的候选池采样，不足时从无重叠的池补充
        if current_disease_genes is not None:
            print(f"  ℹ️  跨疾病候选（有基因重叠）：{len(candidate_pairs_with_overlap)}，"
                  f"（无重叠）：{len(candidate_pairs_no_overlap)}")
            candidate_pairs = candidate_pairs_with_overlap + candidate_pairs_no_overlap
        else:
            candidate_pairs = candidate_pairs_with_overlap
            print(f"  ℹ️  跨疾病候选负样本池：{len(candidate_pairs)}")

        if len(candidate_pairs) >= num_neg:
            neg_set = set(random.sample(candidate_pairs, num_neg))
        else:
            neg_set = set(candidate_pairs)
            if len(candidate_pairs) < num_neg:
                print(f"  ⚠️  跨疾病候选不足，实际生成：{len(neg_set)}/{num_neg}")

        return neg_set

    def _generate_hard_negatives(
        self,
        num_neg: int,
        pos_set: set,
        exclude_pairs: set = None
    ) -> set:
        """
        生成簇内负样本（hard negatives）

        策略：对于每个正样本对 (a, b)，随机替换其中一个节点为同一簇内的其他节点
        """
        import random

        if self.global_cluster_labels is None:
            print("  [Warning] hard 策略需要全局簇标签，但未提供")
            return set()

        neg_set = set()
        pos_pairs_list = list(pos_set)
        max_attempts = num_neg * 10
        attempts = 0

        while len(neg_set) < num_neg and attempts < max_attempts:
            # 随机选择一个正样本对
            orig_pair = random.choice(pos_pairs_list)

            # 随机选择替换哪一个节点
            if random.random() < 0.5:
                replace_idx = 0
            else:
                replace_idx = 1

            # 获取被替换节点的簇标签
            node_to_replace = orig_pair[replace_idx]
            node_to_keep = orig_pair[1 - replace_idx]

            cluster_of_replace = self.global_cluster_labels[node_to_replace]

            # 找到同一簇内的所有节点
            same_cluster_nodes = np.where(self.global_cluster_labels == cluster_of_replace)[0]

            if len(same_cluster_nodes) > 1:
                # 从同一簇中随机选择一个不同的节点
                candidates = [n for n in same_cluster_nodes if n != node_to_replace]
                if len(candidates) > 0:
                    new_node = random.choice(candidates)

                    # 创建新的负样本对
                    if replace_idx == 0:
                        new_pair = tuple(sorted((new_node, node_to_keep)))
                    else:
                        new_pair = tuple(sorted((node_to_keep, new_node)))

                    # 确保不是正样本且未重复
                    if new_pair not in pos_set and new_pair not in neg_set:
                        if exclude_pairs is None or new_pair not in exclude_pairs:
                            neg_set.add(new_pair)

            attempts += 1

        return neg_set


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("batch 为空")

    target1_indices = torch.cat([item['target1_idx'] for item in batch])
    target2_indices = torch.cat([item['target2_idx'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])

    if target1_indices.dim() > 1:
        target1_indices = target1_indices.squeeze()
    if target2_indices.dim() > 1:
        target2_indices = target2_indices.squeeze()
    if labels.dim() > 1:
        labels = labels.squeeze()

    disease_features = torch.stack([item['disease_features'] for item in batch], dim=0)
    n_clusters = torch.stack([item['n_clusters'] for item in batch])

    sample = batch[0]

    result = {
        'target1_indices': target1_indices,
        'target2_indices': target2_indices,
        'disease_features': disease_features,
        'n_clusters': n_clusters,
        'labels': labels,
        'node_features': sample['node_features'],
        'edge_index': sample['edge_index'],
        'batch_size': len(batch),
    }

    if 'disease_idx' in batch[0]:
        disease_indices = torch.cat([item['disease_idx'] for item in batch])
        if disease_indices.dim() > 1:
            disease_indices = disease_indices.squeeze()
        result['disease_indices'] = disease_indices

    return result


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler
