# -*- coding: utf-8 -*-
"""
Disease Feature Calculator (Simplified - Pure Node2Vec Features)
"""

import numpy as np
import torch
import h5py
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DiseaseFeatureCalculator:
    def __init__(
        self,
        full_graph_path: str,
        ttd_dir: Optional[str] = None,
        disease_code: Optional[str] = None,
        seed_genes: Optional[List[str]] = None,
        k_max: int = 30,
        node2vec_dim: int = 128,
        node2vec_gpu: bool = True,
        gmeans_alpha: float = 0.05,
        gmeans_random_state: int = 42
    ):
        self.full_graph_path = Path(full_graph_path)
        self.ttd_dir = Path(ttd_dir) if ttd_dir else None
        self.disease_code = disease_code
        self.seed_genes = seed_genes
        self.k_max = k_max
        self.node2vec_dim = node2vec_dim
        self.node2vec_gpu = node2vec_gpu
        self.gmeans_alpha = gmeans_alpha
        self.gmeans_random_state = gmeans_random_state

        self.node_features = None
        self.node_names = None
        self.node_types = None
        self.gene_name_to_idx = {}
        self.adj_matrix = None

        self._load_full_graph()

    def _load_full_graph(self):
        print(f"[Loading Full Graph] {self.full_graph_path}")

        with h5py.File(self.full_graph_path, 'r') as f:
            self.node_features = f['nodes']['node_features'][:]
            node_names_raw = f['nodes']['node_names'][:]
            self.node_names = np.array([n.decode('utf-8') if isinstance(n, bytes) else n for n in node_names_raw])
            self.node_types = f['nodes']['node_type'][:]

            priority = {2: 0, 4: 1, 3: 2, 1: 3, 0: 4, 5: 5}
            name_to_all_indices = defaultdict(list)
            for i, name in enumerate(self.node_names):
                name_to_all_indices[name].append(i)

            self.gene_name_to_idx = {
                name: min(indices, key=lambda i: priority.get(self.node_types[i], 6))
                for name, indices in name_to_all_indices.items()
            }

            src_list, tgt_list = [], []
            for edge_type in f['edges']:
                src = f['edges'][edge_type]['source'][:]
                tgt = f['edges'][edge_type]['target'][:]
                src_list.extend([src, tgt])
                tgt_list.extend([tgt, src])

            if src_list:
                all_src = np.concatenate(src_list)
                all_tgt = np.concatenate(tgt_list)
                edge_index = np.stack([all_src, all_tgt], axis=0)
                self.edge_index = torch.from_numpy(edge_index).long()
            else:
                raise ValueError("No edge data")

        n = self.node_features.shape[0]
        self.adj_matrix = sp.csr_matrix(
            (np.ones(len(self.edge_index[0])), (self.edge_index[0], self.edge_index[1])),
            shape=(n, n)
        )

        print(f"  - Nodes: {n:,}")
        print(f"  - Edges: {self.adj_matrix.nnz:,}")
        print(f"  - Feature dim: {self.node_features.shape[1]}")

    def load_seed_genes_from_ttd(self) -> List[str]:
        if not self.ttd_dir or not self.disease_code:
            raise ValueError("Need ttd_dir and disease_code")

        from config import DISEASE_CONFIGS

        if self.disease_code not in DISEASE_CONFIGS:
            raise ValueError(f"Unknown disease: {self.disease_code}")

        disease_info = DISEASE_CONFIGS[self.disease_code]
        seed_genes = set()

        icd11_files = disease_info.get('icd11_files', [])
        if not icd11_files:
            icd11_code = disease_info.get('icd11')
            if icd11_code:
                icd11_files = [f"icd11_{icd11_code}_targets.csv"]

        import pandas as pd
        for icd11_file in icd11_files:
            ttd_file = self.ttd_dir / icd11_file
            if ttd_file.exists():
                df = pd.read_csv(ttd_file, sep='\t', comment='#')
                df.columns = df.columns.str.upper().str.strip()
                if 'GENENAME' in df.columns:
                    genes = df['GENENAME'].dropna().astype(str).str.split(';').explode().str.strip()
                    new_genes = [g for g in genes if g and g.lower() != 'nan']
                    seed_genes.update(new_genes)
                    print(f"  - TTD ({icd11_file}): +{len(new_genes)} genes")
            else:
                print(f"  File not found: {ttd_file}")

        print(f"  Total seed genes: {len(seed_genes)}")
        return list(seed_genes)

    def extract_subgraph(self, seed_genes: List[str]) -> Tuple[np.ndarray, sp.csr_matrix]:
        print(f"\n[Extract Subgraph] Strategy: ttd-based (seeds + 1-hop neighbors)")

        mapped_indices = [self.gene_name_to_idx[g] for g in seed_genes if g in self.gene_name_to_idx]

        if len(mapped_indices) == 0:
            raise ValueError("No seed genes mapped to graph")

        seed_idx = np.array(mapped_indices, dtype=np.int32)
        print(f"  - Mapped seed nodes: {len(seed_idx)}")

        mask = np.zeros(len(self.node_names), dtype=bool)
        mask[seed_idx] = True
        mask[self.adj_matrix[seed_idx, :].indices] = True

        subgraph_indices = np.where(mask)[0]
        subgraph_adj = self.adj_matrix[subgraph_indices][:, subgraph_indices].tocsr()

        print(f"  - Subgraph nodes: {len(subgraph_indices)}")
        print(f"  - Subgraph edges: {subgraph_adj.nnz}")

        return subgraph_indices, subgraph_adj

    def run_node2vec(self, adj_matrix: sp.csr_matrix) -> np.ndarray:
        print(f"\n[Node2vec] Training topology embeddings ({self.node2vec_dim}D)")

        if self.node2vec_gpu:
            return self._run_node2vec_gpu(adj_matrix)
        else:
            return self._run_node2vec_cpu(adj_matrix)

    def _run_node2vec_gpu(self, adj_matrix: sp.csr_matrix) -> np.ndarray:
        print(f"  Using GPU version...")
        try:
            from torch_geometric.nn import Node2Vec as PyGNode2Vec
            from torch_geometric.utils import from_scipy_sparse_matrix
        except ImportError as e:
            print(f"  GPU version not available: {e}")
            print(f"  Fallback to CPU...")
            return self._run_node2vec_cpu(adj_matrix)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print(f"  No GPU detected, using CPU...")
            return self._run_node2vec_cpu(adj_matrix)
        else:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        edge_index = edge_index.to(device)

        model = PyGNode2Vec(
            edge_index,
            embedding_dim=self.node2vec_dim,
            walk_length=50,
            context_size=10,
            walks_per_node=100,
            num_negative_samples=1,
            p=0.5,
            q=0.5,
            sparse=True
        ).to(device)

        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        num_epochs = 5
        batch_size = 128

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            embeddings = model.embedding.weight.cpu().detach().numpy()

        return embeddings

    def _run_node2vec_cpu(self, adj_matrix: sp.csr_matrix) -> np.ndarray:
        print(f"  Using CPU version (NetworkX + gensim)...")
        try:
            from node2vec import Node2Vec
            from gensim.models import Word2Vec
        except ImportError:
            print("  Missing dependencies: pip install node2vec gensim networkx")
            return np.zeros((adj_matrix.shape[0], self.node2vec_dim))

        import networkx as nx
        G = nx.from_scipy_sparse_array(adj_matrix)

        node2vec_model = Node2Vec(
            G,
            dimensions=self.node2vec_dim,
            walk_length=50,
            num_walks=100,
            workers=4,
            p=0.5,
            q=0.5,
            quiet=False
        )

        w2v_model = node2vec_model.fit(window=10, min_count=0, batch_words=4)

        n2v_embeddings = np.zeros((G.number_of_nodes(), self.node2vec_dim))
        for i in range(G.number_of_nodes()):
            n2v_embeddings[i] = w2v_model.wv[f"node_{i}"]

        print(f"  Node2vec done: {n2v_embeddings.shape}")
        return n2v_embeddings

    def compute_disease_features(
        self,
        subgraph_adj: sp.csr_matrix,
        n2v_embeddings: np.ndarray,
        k_max: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        from utils.clustering import GMeansClusterer

        k_max = k_max or self.k_max
        n_nodes = n2v_embeddings.shape[0]

        print(f"\n[G-means] Clustering (k_max={k_max}, alpha={self.gmeans_alpha})")
        gmeans = GMeansClusterer(k_max=k_max, random_state=self.gmeans_random_state, alpha=self.gmeans_alpha)
        gmeans.fit(n2v_embeddings)
        cluster_labels = gmeans.labels_
        n_clusters = len(np.unique(cluster_labels))
        print(f"  - Num clusters: {n_clusters}")

        print(f"\n[Centrality] Computing Betweenness Centrality...")
        import networkx as nx
        G = nx.from_scipy_sparse_array(subgraph_adj)
        centrality_dict = nx.betweenness_centrality(G, normalized=True)
        centralities = np.array([centrality_dict.get(i, 0.0) for i in range(n_nodes)])
        if centralities.max() > 0:
            centralities = centralities / centralities.max()

        print(f"\n[Feature Aggregation] Centrality-weighted pooling...")
        disease_features_list = []
        for c in range(n_clusters):
            c_mask = cluster_labels == c
            c_features = n2v_embeddings[c_mask]
            c_weights = centralities[c_mask]
            c_weights = c_weights / (c_weights.sum() + 1e-8)
            cluster_rep = (c_features * c_weights[:, np.newaxis]).sum(axis=0)
            disease_features_list.append(cluster_rep)

        disease_features = np.stack(disease_features_list, axis=0)
        print(f"  - Feature shape: {disease_features.shape}")

        if n_clusters < k_max:
            padding = np.zeros((k_max - n_clusters, self.node2vec_dim))
            disease_features = np.concatenate([disease_features, padding], axis=0)

        print(f"  - Final shape (after padding): {disease_features.shape}")

        return disease_features, n_clusters

    def compute_and_save_features(
        self,
        seed_genes: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        save_subgraph: bool = False,
        k_max: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        k_max = k_max or self.k_max

        if seed_genes is None:
            seed_genes = self.load_seed_genes_from_ttd()

        subgraph_indices, subgraph_adj = self.extract_subgraph(seed_genes)
        n2v_embeddings = self.run_node2vec(subgraph_adj)
        disease_features, n_clusters = self.compute_disease_features(subgraph_adj, n2v_embeddings, k_max)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if self.disease_code:
                feature_file = output_path / f"{self.disease_code}_n2v_features.npz"
            else:
                feature_file = output_path / f"custom_n2v_features.npz"

            np.savez_compressed(feature_file, disease_features=disease_features, n_clusters=n_clusters)
            print(f"  Saved: {feature_file}")

        return disease_features, n_clusters


def compute_disease_features_for_disease(
    disease_code: str,
    full_graph_path: str,
    ttd_dir: str,
    output_dir: str,
    k_max: int = 30,
    node2vec_dim: int = 128,
    node2vec_gpu: bool = True,
    gmeans_alpha: float = 0.05,
    gmeans_random_state: int = 42,
    skip_existing: bool = True,
    save_subgraph: bool = False
) -> Tuple[np.ndarray, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_file = output_path / f"{disease_code}_n2v_features.npz"

    if skip_existing and feature_file.exists():
        print(f"[Skip] Feature exists: {feature_file}")
        data = np.load(feature_file)
        return data['disease_features'], int(data['n_clusters'])

    calculator = DiseaseFeatureCalculator(
        full_graph_path=full_graph_path,
        ttd_dir=ttd_dir,
        disease_code=disease_code,
        k_max=k_max,
        node2vec_dim=node2vec_dim,
        node2vec_gpu=node2vec_gpu,
        gmeans_alpha=gmeans_alpha,
        gmeans_random_state=gmeans_random_state
    )

    disease_features, n_clusters = calculator.compute_and_save_features(
        output_dir=output_dir, save_subgraph=save_subgraph, k_max=k_max
    )

    return disease_features, n_clusters
