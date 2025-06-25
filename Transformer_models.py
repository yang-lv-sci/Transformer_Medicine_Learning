# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:43:51 2025

@author: lvyang
"""
from Transformer_blocks import InputEmbedding,PositionalEncoding,FeatureEmbedding
from Transformer_Encoder_Decoder import TransformerEncoder,TransformerDecoder
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
class TransformerFeatureClassifier(nn.Module):
    """支持多分类只需num_classes改为大于2"""
    def __init__(self, num_features, d_model, n_layers,
                                  num_heads, d_ff, dropout, num_classes=2):
        super().__init__()
        self.feature_embed = FeatureEmbedding(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, num_features)
        self.encoder = TransformerEncoder(n_layers, d_model, num_heads, d_ff, dropout)
        
        """分类头可以换成多层/带激活的MLP等"""
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.feature_embed(x)
        x = self.pos_encoding(x) #很重要
        x = self.encoder(x)
        # 这里用CLS池化方式（取第一个token），也可以平均池化
        x = x[:, 0, :]  # [batch, d_model]
        logits = self.classifier(x)
        return logits
    
class TransformerRegression(nn.Module):
    def __init__(self, num_features, d_model=64, n_layers=2, num_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.feature_embed = FeatureEmbedding(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, num_features)
        self.encoder = TransformerEncoder(n_layers, d_model, num_heads, d_ff, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 对所有特征池化
        self.reg_head = nn.Linear(d_model, 1).to(device)

    def forward(self, x):
        """
        x: [batch, num_features]
        """
        x = self.feature_embed(x)  # [batch, num_features, d_model]
        x = self.pos_encoding(x) #很重要
        x = self.encoder(x)
        x = x.transpose(1, 2)       # [batch, d_model, num_features]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]
        out = self.reg_head(x).squeeze(-1)  # [batch]
        return out