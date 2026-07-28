# -*- coding: utf-8 -*-
"""
TargetSynergy Main Model
"""

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path

from .target_encoder import TargetEncoder
from .disease_encoder import DiseaseEncoder
from .interaction import InteractionLayer
from .target_pair_aggregator import TargetPairAggregator


class TargetSynergyModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        jk_num_layers: int = 3,
        gnn_type: str = 'gin',
        num_attention_heads: int = 4,
        num_recurrent_iterations: int = 1,
        dropout: float = 0.3,
    ):
        super(TargetSynergyModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.jk_num_layers = jk_num_layers
        self.gnn_type = gnn_type

        self.target_encoder = TargetEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=jk_num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

        self.target_pair_aggregator = TargetPairAggregator(hidden_dim=hidden_dim, dropout=dropout)

        self.interaction_layer = InteractionLayer(
            target_dim=hidden_dim,
            disease_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_iterations=num_recurrent_iterations,
            dropout=dropout
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )

        self.target_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.disease_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        target1_indices: torch.Tensor,
        target2_indices: torch.Tensor,
        disease_features: torch.Tensor,
        n_clusters: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(target1_indices, torch.Tensor):
            raise TypeError(f"target1_indices 必须是 tensor，实际类型：{type(target1_indices)}")
        if not isinstance(target2_indices, torch.Tensor):
            raise TypeError(f"target2_indices 必须是 tensor，实际类型：{type(target2_indices)}")

        if target1_indices.dim() == 0:
            target1_indices = target1_indices.unsqueeze(0)
        if target2_indices.dim() == 0:
            target2_indices = target2_indices.unsqueeze(0)

        batch_size = target1_indices.shape[0]
        if batch_size == 0:
            raise ValueError(f"batch_size 为 0, target1_indices.shape: {target1_indices.shape}")

        target1_emb = self.target_encoder(node_features, edge_index, target1_indices)
        target2_emb = self.target_encoder(node_features, edge_index, target2_indices)

        z_pair = self.target_pair_aggregator(target1_emb, target2_emb)

        z_pair_updated, disease_cluster_updated, global_disease_vec = self.interaction_layer(
            target_pair_emb=z_pair,
            disease_cluster_emb=disease_features,
            n_clusters=n_clusters
        )

        concatenated = torch.cat([z_pair_updated, global_disease_vec], dim=1)
        logits = self.predictor(concatenated).squeeze(-1)
        prob = torch.sigmoid(logits)

        projected_target = self.target_projection(z_pair_updated)
        projected_disease = self.disease_projection(global_disease_vec)

        return {
            'logits': logits,
            'prob': prob,
            'target_pair_emb': z_pair_updated,
            'global_disease_emb': global_disease_vec,
            'disease_cluster_emb': disease_cluster_updated,
            'projected_target': projected_target,
            'projected_disease': projected_disease,
        }

    def save_checkpoint(self, path: str, **kwargs):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'gnn_type': self.gnn_type,
            },
            **kwargs
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        state_dict = checkpoint['model_state_dict']

        # 从 state_dict 推断实际的 jk_num_layers
        if 'target_encoder.jknet.output_proj.weight' in state_dict:
            output_proj_shape = state_dict['target_encoder.jknet.output_proj.weight'].shape
            hidden_dim = config.get('hidden_dim', 128)
            num_layers_features = output_proj_shape[1] // hidden_dim
            inferred_jk_num_layers = num_layers_features - 1  # 减去输入层
            print(f"[Load Checkpoint] Inferred jk_num_layers={inferred_jk_num_layers} from output_proj shape {output_proj_shape}")
            config['jk_num_layers'] = inferred_jk_num_layers

        # 过滤掉不需要的参数（简化版本固定配置）
        valid_keys = ['input_dim', 'hidden_dim', 'output_dim', 'jk_num_layers',
                      'num_attention_heads', 'num_recurrent_iterations', 'dropout', 'gnn_type']
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}

        # 设置默认值
        if 'gnn_type' not in filtered_config:
            filtered_config['gnn_type'] = 'gin'
        if 'jk_num_layers' not in filtered_config:
            filtered_config['jk_num_layers'] = 3  # 默认 3 层
        if 'num_attention_heads' not in filtered_config:
            filtered_config['num_attention_heads'] = 4
        if 'num_recurrent_iterations' not in filtered_config:
            filtered_config['num_recurrent_iterations'] = 1
        if 'dropout' not in filtered_config:
            filtered_config['dropout'] = 0.3

        model = cls(**filtered_config)
        model.load_state_dict(state_dict)
        model.to(device)
        return model, checkpoint
