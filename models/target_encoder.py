# -*- coding: utf-8 -*-
"""
Target Encoder: Jumping Knowledge Networks (JK-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv
from typing import Literal


class GINLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super(GINLayer, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.conv = GINConv(self.nn)
        self.batch_norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.batch_norm(out)
        return out


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(input_dim, output_dim)
        self.batch_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class JKNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        gnn_type: Literal['gin', 'gcn'] = 'gin',
        dropout: float = 0.3
    ):
        super(JKNNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.graph_layers = nn.ModuleList()
        LayerClass = GINLayer if gnn_type == 'gin' else GCNLayer
        for i in range(num_layers):
            self.graph_layers.append(LayerClass(hidden_dim, hidden_dim, dropout=dropout))
        self.output_proj = nn.Linear(hidden_dim * (num_layers + 1), output_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)
        x = F.relu(x)
        layer_features = [x]
        for graph_layer in self.graph_layers:
            x = graph_layer(x, edge_index)
            layer_features.append(x)
        concatenated = torch.cat(layer_features, dim=-1)
        node_embeddings = self.output_proj(concatenated)
        return node_embeddings


class TargetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: Literal['gin', 'gcn'] = 'gin',
        dropout: float = 0.3
    ):
        super(TargetEncoder, self).__init__()
        self.jknet = JKNNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

    def forward(self, node_features, edge_index, target_indices):
        node_embeddings = self.jknet(node_features, edge_index)
        if target_indices.dim() == 1:
            target_embeddings = node_embeddings[target_indices]
            target_embeddings = target_embeddings.unsqueeze(1)
        else:
            batch_size, n_targets = target_indices.shape
            target_indices_flat = target_indices.view(-1)
            target_embeddings_flat = node_embeddings[target_indices_flat]
            target_embeddings = target_embeddings_flat.view(batch_size, n_targets, -1)
        return target_embeddings
