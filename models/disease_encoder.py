# -*- coding: utf-8 -*-
"""
Disease Encoder: G-means Clustering + Centrality Weighted Aggregation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
import networkx as nx

from utils.clustering import GMeansClusterer


class DiseaseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        k_max: int = 30,
        random_state: int = 42,
        pretrained_clusters: Optional[Dict] = None
    ):
        super(DiseaseEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_max = k_max
        self.random_state = random_state
        self.pretrained_clusters = pretrained_clusters or {}

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
        )

        self.cluster_refine = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

        self.max_clusters = k_max

    def perform_gmeans_clustering(self, node_features: torch.Tensor, adj_matrix: np.ndarray, n_nodes: int) -> np.ndarray:
        gmeans = GMeansClusterer(k_max=self.k_max, random_state=self.random_state, alpha=0.05)
        gmeans.fit(node_features)
        return gmeans.labels_

    def compute_centralities(self, adj_matrix: np.ndarray, node_indices: Optional[np.ndarray] = None) -> np.ndarray:
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = adj_matrix.astype(float)
        G = nx.from_numpy_array(adj_matrix)
        centrality_dict = nx.betweenness_centrality(G, normalized=True)
        n_nodes = adj_matrix.shape[0]
        centralities = np.array([centrality_dict.get(i, 0.0) for i in range(n_nodes)])
        if centralities.max() > 0:
            centralities = centralities / centralities.max()
        return centralities

    def aggregate_clusters(
        self,
        node_features: torch.Tensor,
        cluster_labels: np.ndarray,
        centralities: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, int]:
        device = node_features.device
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)

        cluster_reps_list = []
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_features = node_features[mask]
            if centralities is not None:
                weights = torch.FloatTensor(centralities[mask]).to(device)
                weights = weights / (weights.sum() + 1e-8)
                cluster_rep = (cluster_features * weights.unsqueeze(-1)).sum(dim=0)
            else:
                cluster_rep = cluster_features.mean(dim=0)
            cluster_reps_list.append(cluster_rep)

        cluster_reps = torch.stack(cluster_reps_list, dim=0)
        return cluster_reps, n_clusters

    def pad_clusters(self, cluster_reps: torch.Tensor, n_clusters: int) -> torch.Tensor:
        max_clusters = self.max_clusters
        if n_clusters >= max_clusters:
            return cluster_reps[:max_clusters, :]
        padding = torch.zeros(max_clusters - n_clusters, cluster_reps.shape[1], device=cluster_reps.device)
        padded_reps = torch.cat([cluster_reps, padding], dim=0)
        return padded_reps

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, cluster_labels: Optional[np.ndarray] = None, compute_clustering: bool = False, adj_matrix: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, int]:
        n_nodes = node_features.shape[0]

        if cluster_labels is None and compute_clustering:
            if adj_matrix is None:
                raise ValueError("需要 adj_matrix 来执行聚类")
            cluster_labels = self.perform_gmeans_clustering(node_features, adj_matrix, n_nodes)

        centralities = None
        if compute_clustering and adj_matrix is not None:
            centralities = self.compute_centralities(adj_matrix)

        projected_features = self.feature_proj(node_features)

        if cluster_labels is not None:
            cluster_reps, n_clusters = self.aggregate_clusters(projected_features, cluster_labels, centralities)
        else:
            cluster_reps = projected_features.mean(dim=0, keepdim=True)
            n_clusters = 1

        cluster_reps = self.cluster_refine(cluster_reps)
        padded_reps = self.pad_clusters(cluster_reps, n_clusters)

        return padded_reps, torch.tensor([n_clusters], device=node_features.device)
