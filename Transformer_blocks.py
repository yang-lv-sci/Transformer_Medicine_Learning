# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:25:42 2025

@author: Administrator
"""
import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Input Embedding
# Transformer 接收的是一系列整数（token IDs），这些整数是通过 tokenizer 将文本转成词汇表索引。
#但模型不能直接处理整数，它需要将这些整数转为高维向量。这一步就是 Input Embedding 的任务

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()
        """
        调用父类（即 nn.Module）的构造函数 __init__()，完成父类部分的初始化
        子类需要显式调用父类的构造函数，否则父类的初始化代码不会运行
        因为PyTorch 的 nn.Module 在它的 __init__() 里做了很多重要的初始化工作
        """
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        """
        vocab_size: tokenizer 的词汇总数，例如 BERT 是 30522。
        d_model: 每个词被表示成的向量维度（通常为 512 或 768）。
        实际上这里建立了一个 [vocab_size, d_model] 的查找表（矩阵），每行表示一个词的向量表示
        """
        self.d_model = d_model

    def forward(self, x):
        # x: [batch_size, seq_len] -> input token IDs
        embed = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        return embed * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        #Transformer 原始论文中提到，为了保持嵌入的方差与位置编码相匹配，必须放大输入嵌入。
        #如果不乘这个因子，embedding 的值可能相对较小，和后面加的 positional encoding 不成比例
        
# 5. Positional Encoding
#由于标准 Transformer 结构中没有 RNN，因此必须加入显式的位置信息，这个数值是固定的
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        # 创建一个 (max_len, d_model) 的全零矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #计算指数部分

        # 偶数维度：sin，奇数维度：cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 第0,2,4...列
        pe[:, 1::2] = torch.cos(position * div_term)  # 第1,3,5...列

        # 增加 batch 维度：shape = [1, max_len, d_model]
        pe = pe.unsqueeze(0).to(device)

        # 注册为 buffer，不参与训练但会随模型保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 把前 seq_len 个位置编码加到输入上
        x = x + self.pe[:, :seq_len]
        return x
    
# 6. Multi-Head Self-Attention

#输入输出都是[B, L, D]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能整除 num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Wq, Wk, Wv
        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)

        # 最终输出层
        self.out_proj = nn.Linear(d_model, d_model).to(device)

    def forward(self, x, mask=None):
        B, L, D = x.shape  # batch_size, seq_len, d_model

        # 线性映射 & 分头
        # (B, L, heads, head_dim)->(B, heads, L, head_dim)
        # self.q_linear(x)做映射，映射之后.view(B, L, self.num_heads, self.head_dim)拆分成多个头，拆分后对调2，3维度，为了做 attention
        Q = self.q_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  
        K = self.k_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention 计算
        # K转置只处理后两个维度
        # 表示每个头（H）在每个位置（第一个 L）对所有位置（第二个 L）的注意力打分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, L, L)

        if mask is not None:
            # 为什么要加 Mask？
            # 1. 对于不等长句子，我们会用 [PAD] token 补齐。这些 token 不应该被模型注意到（无信息）
            # 2. 在训练语言模型时，为了防止模型“偷看”未来的 token，需要屏蔽未来位置（用在 Decoder）
            scores = scores.masked_fill(mask == 0, float("-inf"))
            """对于 mask == 0 的位置（即 不应该关注的位置），把 scores 置为负无穷 -inf。
            后面进入 softmax，这些位置就会变成 0 权重（因为 softmax(-inf) ≈ 0）"""
        
        # 详细解释 a_ij = softmax([Q_i @ K_j^]/sqrt(d_k))
        # 对一个 attention head，scores[0, 0] 的 shape 是 [L_query , L_key] s_ij​：表示第 i 个 token 关注第 j 个 token 的注意力打分
        # dim=-1 是“沿着最后一个维度做 softmax” 对每一个 scores[b, h, i, :]，做 softmax —— 固定 i，对第 j 维做 softmax
        # Softmax 是按行做的，对每一个i，在j维度求和 a_ij 的结果是1，每个 query 看所有 key 的权重
        attn_weights = torch.softmax(scores, dim=-1)  # (B, heads, L, L)
        
        context = torch.matmul(attn_weights, V)       # (B, heads, L, head_dim)
        
        """举例（假设 head_dim = 2）对于某一 head
        attn_weights[0, 0] =      V[0, 0] =
        [                        [
            [0.1, 0.3, 0.6],         [1.0, 0.0],
            [0.2, 0.5, 0.3],         [0.0, 1.0],
            [0.7, 0.2, 0.1]          [1.0, 1.0]
        ]                        ]
        
        context[0]=0.1⋅V[0]+0.3⋅V[1]+0.6⋅V[2]
        [0.1, 0.0]+[0.0, 0.3]+[0.6, 0.6]=[0.7, 0.9]
        
        虽然 V 是每个 token 的特征，但 Attention 根据 Query 关注程度决定聚合哪些 Value
        可以理解为：第 0 个 token 觉得第 2 个 token 很重要（权重 0.6），因此吸收了更多它的信息
        最后输出的 context[0] 是一个“动态聚合”的结果，代表上下文感知的表示。
        """
        
        # 拼回去,contiguous()：确保内存连续性（防止 .view() 报错）
        context = context.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, H, head_dim] -> (B, L, D)
       
        # 再投影回原始维度,将多个 head 融合的信息统一变换，提供最终输出允许模型重新混合各个 head 的特征
        out = self.out_proj(context)  
        
        return out
    
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout = 0.1 ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model).to(device)
        #[PAD] 是否是 0 向量不会影响其他 token只要在 Attention 和 Loss 时屏蔽掉 padding 部分，一切都正常 ✅
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.dropout(sublayer_out))
        """x：原始输入
        sublayer_out：比如来自 MultiHeadAttention(x) 或 FFN(x)
        输出：做了残差 + LayerNorm 的结果"""
        
class FeedForward(nn.Module):
    """
    Transformer 论文中，FeedForward Network 是两个全连接层 + 激活 + dropout：
    FFN(x)=max(0,x*W_1​ +b_1​ )W_2​ +b_2​ 第一层将维度从 d_model映射到更大的隐藏维度 d_ff（如 2048）
    第二层映射回原维度 d model
    ReLU 激活
    dropout 防止过拟合
    """
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        ).to(device)

    def forward(self, x):
        return self.net(x)