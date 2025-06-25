# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:43:51 2025

@author: lvyang
"""
from Transformer_blocks import InputEmbedding,PositionalEncoding
from Transformer_Encoder_Decoder import TransformerEncoder,TransformerDecoder
import torch.nn as nn

class TransformerClassifier(nn.Module):
    """支持多分类只需num_classes改为大于2"""
    def __init__(self, vocab_size, d_model, max_len, n_layers, num_heads, d_ff, dropout, num_classes=2):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(n_layers, d_model, num_heads, d_ff, dropout)
        """分类头可以换成多层/带激活的MLP等"""
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x, attention_mask)
        # 这里用CLS池化方式（取第一个token），也可以平均池化
        x = x[:, 0, :]  # [batch, d_model]
        logits = self.classifier(x)
        return logits