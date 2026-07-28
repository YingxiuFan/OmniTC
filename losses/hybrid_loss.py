# -*- coding: utf-8 -*-
"""
混合损失函数模块

实现四种损失的加权和：
1. BCE Loss: 二分类交叉熵
2. Orthogonality Loss: 簇表征正交损失
3. Contrastive Loss: 投影对比损失
4. Adversarial Loss: 自适应负采样对抗损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .adversarial_loss import AdversarialLoss


class HybridLoss(nn.Module):
    """
    混合损失函数

    Loss_total = Loss_BCE + α * Loss_orth + β * Loss_con + γ * Loss_adversarial
    """

    def __init__(
        self,
        alpha_orth: float = 0.1,
        beta_con: float = 0.5,
        temperature: float = 0.1,
        gamma_adversarial: float = 1.0,
        adversarial_temperature: float = 0.5
    ):
        """
        Args:
            alpha_orth: 正交损失权重
            beta_con: 对比损失权重
            temperature: 对比损失温度参数
            gamma_adversarial: 对抗损失权重
            adversarial_temperature: 对抗损失温度参数
        """
        super(HybridLoss, self).__init__()

        self.alpha_orth = alpha_orth
        self.beta_con = beta_con
        self.temperature = temperature
        self.gamma_adversarial = gamma_adversarial

        self.bce_loss = nn.BCELoss()

        # 对抗损失
        self.adversarial_loss_fn = AdversarialLoss(
            temperature=adversarial_temperature,
            reduction='mean'
        )

    def bce_loss_fn(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        二分类交叉熵损失

        Args:
            pred_logits: 预测 logits (batch_size,)
            labels: 真实标签 (batch_size,)

        Returns:
            loss: BCE 损失
        """
        pred_prob = torch.sigmoid(pred_logits)
        loss = self.bce_loss(pred_prob, labels)
        return loss

    def orthogonality_loss_fn(
        self,
        cluster_reps: torch.Tensor,
        n_clusters: torch.Tensor
    ) -> torch.Tensor:
        """
        正交损失

        约束不同功能簇的表征保持正交：
        Loss_orth = ||V_c @ V_c^T - I||_F^2

        Args:
            cluster_reps: 簇表征 (batch_size, max_clusters, hidden_dim)
            n_clusters: 每个样本的实际簇数 (batch_size,)

        Returns:
            loss: 正交损失
        """
        batch_size, max_clusters, hidden_dim = cluster_reps.shape

        # 确保 n_clusters 是 1 维 tensor
        if n_clusters.dim() == 0:
            # 标量，batch_size=1 的情况
            n_clusters = n_clusters.unsqueeze(0)
        elif n_clusters.dim() > 1:
            # 多维，squeeze 多余的维度
            n_clusters = n_clusters.squeeze()

        total_loss = 0.0
        valid_samples = 0

        for i in range(batch_size):
            n_c = n_clusters[i].item()  # 显式转换为整数

            if n_c <= 1:
                # 只有 1 个簇时，无需计算正交损失
                continue

            # 提取非 Padding 的簇
            valid_reps = cluster_reps[i, :n_c, :]  # (n_c, hidden_dim)

            # L2 归一化
            norms = valid_reps.norm(p=2, dim=1, keepdim=True) + 1e-8
            normalized_reps = valid_reps / norms

            # 计算 V_c @ V_c^T
            vt = torch.matmul(normalized_reps, normalized_reps.t())  # (n_c, n_c)

            # 计算与单位矩阵的 Frobenius 距离
            identity = torch.eye(n_c, device=cluster_reps.device)
            loss = torch.norm(vt - identity, p='fro') ** 2

            total_loss += loss
            valid_samples += 1

        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=cluster_reps.device)

    def contrastive_loss_fn(
        self,
        z_pair: torch.Tensor,
        v_dis_global: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        投影对比损失 (InfoNCE)

        拉近协同对与对应疾病特征的距离，推开非协同对

        Args:
            z_pair: 靶点对表征 (batch_size, hidden_dim)
            v_dis_global: 全局疾病表征 (batch_size, hidden_dim)
            labels: 标签 (batch_size,)，1 表示正样本（协同），0 表示负样本

        Returns:
            loss: 对比损失
        """
        # L2 归一化
        z_norm = F.normalize(z_pair, p=2, dim=1)  # (batch_size, hidden_dim)
        v_norm = F.normalize(v_dis_global, p=2, dim=1)  # (batch_size, hidden_dim)

        # 计算相似度
        similarities = torch.sum(z_norm * v_norm, dim=1) / self.temperature  # (batch_size,)

        # 分离正负样本
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        # InfoNCE 损失
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_sim = similarities[pos_mask]  # (n_pos,)
            neg_sim = similarities[neg_mask]  # (n_neg,)

            # 对于每个正样本，计算与所有负样本的对比
            pos_loss = 0.0
            for ps in pos_sim:
                numerator = torch.exp(ps)
                denominator = numerator + torch.sum(torch.exp(neg_sim))
                pos_loss -= torch.log(numerator / (denominator + 1e-8))

            pos_loss = pos_loss / max(len(pos_sim), 1)

            # 负样本使用 hinge loss
            margin = 1.0
            neg_loss = torch.mean(F.relu(neg_sim - margin))

            loss = pos_loss + neg_loss

        elif pos_mask.sum() > 0:
            # 只有正样本
            loss = -torch.mean(similarities[pos_mask])

        elif neg_mask.sum() > 0:
            # 只有负样本
            margin = 1.0
            loss = torch.mean(F.relu(similarities[neg_mask] - margin))

        else:
            loss = torch.tensor(0.0, device=z_pair.device)

        return loss

    def forward(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
        target_pair_emb: Optional[torch.Tensor] = None,
        disease_global_emb: Optional[torch.Tensor] = None,
        cluster_reps: Optional[torch.Tensor] = None,
        n_clusters: Optional[torch.Tensor] = None
    ) -> dict:
        """
        计算混合损失

        Args:
            pred_logits: 预测 logits (batch_size,)
            labels: 真实标签 (batch_size,)
            target_pair_emb: 靶点对表征（用于对比损失）
            disease_global_emb: 全局疾病表征（用于对比损失）
            cluster_reps: 簇表征（用于正交损失）
            n_clusters: 实际簇数（用于正交损失）

        Returns:
            losses: {
                'total': 总损失，
                'bce': BCE 损失，
                'orth': 正交损失，
                'con': 对比损失，
                'adversarial': 对抗损失（如果启用）,
            }
        """
        # 1. BCE 损失
        loss_bce = self.bce_loss_fn(pred_logits, labels)

        # 2. 正交损失
        if cluster_reps is not None and n_clusters is not None:
            loss_orth = self.orthogonality_loss_fn(cluster_reps, n_clusters)
        else:
            loss_orth = torch.tensor(0.0, device=pred_logits.device)

        # 3. 对比损失
        if target_pair_emb is not None and disease_global_emb is not None:
            loss_con = self.contrastive_loss_fn(
                target_pair_emb, disease_global_emb, labels
            )
        else:
            loss_con = torch.tensor(0.0, device=pred_logits.device)

        # 4. 对抗损失
        loss_adv = self.adversarial_loss_fn(pred_logits, labels)

        # 总损失
        loss_total = loss_bce + self.alpha_orth * loss_orth + self.beta_con * loss_con + self.gamma_adversarial * loss_adv

        return {
            'total': loss_total,
            'bce': loss_bce,
            'orth': loss_orth,
            'con': loss_con,
            'adversarial': loss_adv,
        }
