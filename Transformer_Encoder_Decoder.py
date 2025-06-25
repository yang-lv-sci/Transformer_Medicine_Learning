# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:31:22 2025

@author: lvyang
"""

import torch.nn as nn
from Transformer_blocks import TransformerEncoderBlock,TransformerDecoderBlock

"""堆叠N层Encoder/Decoder
用nn.ModuleList循环堆叠EncoderBlock/DecoderBlock"""

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x
        