# -*- coding: utf-8 -*-
"""
Target Pair Aggregator: Compact Bilinear Pooling (CBP)
"""

import torch
import torch.nn as nn


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        for i in range(2):
            h = torch.randint(0, output_dim, (input_dim,))
            s = torch.tensor(2 * torch.randint(0, 2, (input_dim,)).numpy() - 1, dtype=torch.float32)
            self.register_buffer(f'h{i+1}', h)
            self.register_buffer(f's{i+1}', s)

    def _sketch(self, x: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        sketch = x.new_zeros(batch_size, self.output_dim)
        sketch.scatter_add_(1, h.unsqueeze(0).expand(batch_size, -1), x * s.unsqueeze(0))
        return sketch

    def forward(self, target1_emb: torch.Tensor, target2_emb: torch.Tensor) -> torch.Tensor:
        x = target1_emb.squeeze(1)
        y = target2_emb.squeeze(1)
        sketch_x = self._sketch(x, self.h1, self.s1)
        sketch_y = self._sketch(y, self.h2, self.s2)
        fft_x = torch.fft.fft(sketch_x)
        fft_y = torch.fft.fft(sketch_y)
        fft_prod = fft_x * fft_y
        out = torch.fft.ifft(fft_prod).real
        return out


class TargetPairAggregator(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3):
        super(TargetPairAggregator, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregator = CompactBilinearPooling(input_dim=hidden_dim, output_dim=hidden_dim)

    def forward(self, target1_emb: torch.Tensor, target2_emb: torch.Tensor) -> torch.Tensor:
        return self.aggregator(target1_emb, target2_emb)
