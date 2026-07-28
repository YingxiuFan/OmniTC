# -*- coding: utf-8 -*-
"""
自适应负采样对抗损失函数模块

灵感来源于知识图谱嵌入模型（如 RotatE），用于为负样本分配自适应权重。

核心思想：
1. 将 (靶点 1, 靶点 2, 疾病) 视为三元组
2. 利用负样本的得分计算权重，让模型更关注"困难的"负样本（得分高的负样本）
3. 温度参数控制权重分布的尖锐程度

数学形式：
    negative_score = model(negative_samples)  # 负样本得分
    weights = F.softmax(negative_score * temperature, dim=1).detach()
    adversarial_loss = (weights * F.logsigmoid(-negative_score)).sum(dim=1)

关键特性：
- 温度参数 α：控制权重分布的尖锐程度，α越大越关注最难负样本
- 梯度阻断：.detach() 确保权重计算不产生梯度，仅作为系数
- 期望损失：加权后的 log-sigmoid 损失之和
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdversarialLoss(nn.Module):
    """
    自适应负采样对抗损失

    核心思想：
    1. 对负样本，用模型预测的 logit 作为"打分"
    2. 用 softmax(negative_logit * temperature) 计算权重
    3. 梯度阻断：权重计算不参与反向传播
    4. 期望损失：权重 * log_sigmoid(-logit)

    参数：
        temperature: 温度参数，控制权重分布的尖锐程度
        reduction: 缩减方式 ('mean' | 'sum' | 'none')

    Examples:
        >>> loss_fn = AdversarialLoss(temperature=0.5)
        >>> logits = torch.tensor([1.0, -0.5, 2.0, -1.0])  # 2 正 2 负
        >>> labels = torch.tensor([1, 0, 1, 0])
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        super(AdversarialLoss, self).__init__()

        if temperature <= 0:
            raise ValueError(f"temperature 必须为正数，得到 {temperature}")

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction 必须为 'mean', 'sum', 或 'none'，得到 {reduction}")

        self.temperature = temperature
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算自适应负采样对抗损失

        Args:
            logits: 模型预测 logits (batch_size,)
            labels: 真实标签 (batch_size,), 1=正样本，0=负样本

        Returns:
            loss: 对抗损失标量

        注意：
            - 只对 label=0 的负样本计算损失
            - 如果 batch 中没有负样本，返回 0
        """
        # 1. 筛选负样本 (label = 0)
        neg_mask = (labels == 0)
        neg_logits = logits[neg_mask]  # (n_neg,)

        # 如果没有负样本，返回 0
        if neg_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 2. 计算 softmax 权重（带温度参数，梯度阻断）
        # 温度越小，权重分布越均匀；温度越大，越关注高分（困难）负样本
        neg_weights = F.softmax(neg_logits * self.temperature, dim=0).detach()

        # 3. 计算 log_sigmoid 损失
        # log(sigmoid(-x)) = -log(1 + exp(x))，当 x 很大时趋近于 -x
        # 我们希望负样本的 logit 尽可能小（负），这样 -logit 就会很大
        neg_loss = F.logsigmoid(-neg_logits)  # (n_neg,)

        # 4. 加权期望损失
        # 负号是因为 logsigmoid 是负值，我们希望最小化负的对数似然
        adv_loss = -(neg_weights * neg_loss).sum()

        # 5. 根据 reduction 参数调整输出
        if self.reduction == 'none':
            # 对于 'none' 情况，返回未缩减的损失（每个负样本的贡献）
            # 注意：这与 BCE 的 'none' 行为不同，因为这里是加权求和
            return adv_loss.unsqueeze(0).expand(neg_logits.shape)
        elif self.reduction == 'sum':
            return adv_loss
        else:  # 'mean'
            # 对 batch 内所有样本（包括正样本）取平均，保持一致性
            return adv_loss / logits.size(0)
