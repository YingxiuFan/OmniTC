# -*- coding: utf-8 -*-
"""
Interaction Layer: Recurrent Attention Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        batch_size, target_len, embed_dim = query.shape
        source_len = key.shape[1]

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, target_len, embed_dim)
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights
        return output, None


class InteractionLayer(nn.Module):
    def __init__(self, target_dim: int, disease_dim: int, hidden_dim: int, num_heads: int = 4, num_iterations: int = 3, dropout: float = 0.1):
        super(InteractionLayer, self).__init__()
        self.target_dim = target_dim
        self.disease_dim = disease_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations

        self.target_to_hidden = nn.Linear(target_dim, hidden_dim)
        self.disease_to_hidden = nn.Linear(disease_dim, hidden_dim)

        self.target2disease_attn = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.disease2target_attn = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        self.target_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.disease_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.target_norm = nn.LayerNorm(hidden_dim)
        self.disease_norm = nn.LayerNorm(hidden_dim)

    def forward(self, target_pair_emb: torch.Tensor, disease_cluster_emb: torch.Tensor, n_clusters: torch.Tensor) -> tuple:
        batch_size = target_pair_emb.shape[0]

        if n_clusters.dim() == 0:
            n_clusters = n_clusters.unsqueeze(0)
        elif n_clusters.dim() > 1:
            n_clusters = n_clusters.squeeze()

        z_pair = self.target_to_hidden(target_pair_emb)
        v_dis = self.disease_to_hidden(disease_cluster_emb)

        max_clusters = disease_cluster_emb.shape[1]
        cluster_mask = torch.zeros(batch_size, max_clusters, device=target_pair_emb.device, dtype=torch.bool)

        for i in range(batch_size):
            n_c = n_clusters[i].item()
            if n_c < max_clusters:
                cluster_mask[i, n_c:] = True

        for iteration in range(self.num_iterations):
            z_pair_expanded = z_pair.unsqueeze(1)

            v_dis_updated, _ = self.target2disease_attn(query=z_pair_expanded, key=v_dis, value=v_dis, key_padding_mask=cluster_mask)
            v_dis_updated = v_dis_updated.squeeze(1)

            global_disease_vec = self._compute_global_vector(v_dis, n_clusters)
            global_disease_vec = global_disease_vec + v_dis_updated
            global_disease_vec = self.disease_norm(global_disease_vec)
            global_disease_ffn = self.disease_ffn(global_disease_vec)
            global_disease_vec = global_disease_vec + global_disease_ffn

            global_disease_vec_expanded = global_disease_vec.unsqueeze(1)

            z_pair_updated, _ = self.disease2target_attn(query=global_disease_vec_expanded, key=z_pair.unsqueeze(1), value=z_pair.unsqueeze(1))
            z_pair_updated = z_pair_updated.squeeze(1)

            z_pair = z_pair + z_pair_updated
            z_pair = self.target_norm(z_pair)
            z_pair = self.target_ffn(z_pair) + z_pair

        final_global_disease_vec = self._compute_global_vector(v_dis, n_clusters)

        return z_pair, v_dis, final_global_disease_vec

    def _compute_global_vector(self, cluster_emb: torch.Tensor, n_clusters: torch.Tensor) -> torch.Tensor:
        batch_size, max_clusters, hidden_dim = cluster_emb.shape
        global_vecs = []
        for i in range(batch_size):
            n_c = n_clusters[i]
            valid_clusters = cluster_emb[i, :n_c, :]
            global_vec = valid_clusters.mean(dim=0)
            global_vecs.append(global_vec)
        global_vecs = torch.stack(global_vecs, dim=0)
        return global_vecs
