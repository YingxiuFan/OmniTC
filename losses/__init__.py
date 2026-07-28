# -*- coding: utf-8 -*-
"""
损失函数模块
"""

from .hybrid_loss import HybridLoss
from .adversarial_loss import AdversarialLoss

__all__ = ['HybridLoss', 'AdversarialLoss']
