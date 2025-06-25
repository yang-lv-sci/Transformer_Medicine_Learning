# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:24:34 2025

@author: lvyang
"""
"""Classifier只用Encoder+Pooling+分类头"""

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from Transformer_models import TransformerClassifier
import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

"""
对“自写Transformer+不加载预训练权重”的小样本任务来说，d_model=512、4层稍大
如果用小数据子集，d_model=128~256，n_layers=1~2 更容易收敛更快
"""

# 预设超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DL_batch_size = 2048
vocab_size = 30522 # tokenizer 的词汇总数，例如 BERT 是 30522
d_model = 256  # 一般是512或者768
max_len = 64 # 超过进行截断
num_heads = 4
dropout_rate = 0.1
d_ff = 256 #d_ff（前馈全连接层的隐藏维度）应该比 d_model 大一到数倍
n_layers = 2  # Encoder层数
num_classes = 2  # SST-2是二分类
learing_rate = 1e-3
num_epochs = 10

# 1. 准备 NLP 数据集
# pip install datasets
# conda install fsspec
# conda install numpy
# conda install filelock
# pip install --upgrade huggingface-hub 重启内核

# SST-2（Stanford Sentiment Treebank）二分类，句子非常短（大多小于30词）大小：训练约6,900条，测试约900条
dataset = load_dataset("glue", "sst2")
#可以访问 dataset['train'] 来获取训练数据，dataset['test'] 来获取测试数据
print('训练集中的一条数据示例\n',dataset['train'][0])

"""
训练集中的一条数据示例
 {'sentence': 'hide new secretions from the parental units ', 'label': 0, 'idx': 0}
"""

# 2. 数据预处理与 Tokenization
#为了将文本数据转换为模型可以处理的格式，我们需要将文本进行 Tokenization (Inputs 数字化)

# 使用 BERT 的 Tokenizer （大小写不敏感）
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 对一条样本文本进行 Tokenization
# numpy 1.26.3 会出现ModuleNotFoundError: No module named 'numpy.rec'问题
# 使用conda uninstall numpy 后使用 pip install numpy
# 出现ModuleNotFoundError: No module named 'transformers.models.auto.tokenization_auto' 重启内核无效
# pip install transformers 后跳回 # 1. 准备 NLP 数据集解决

# Tokenization 函数
def tokenize_function(examples):
    # return_tensors="pt"将结果返回为 PyTorch 张量格式，以便后续训练
    # padding=True：确保每个句子的长度一致（根据最大句子长度进行填充）。
    # truncation=True：如果句子超过了最大长度，会进行截断
    """truncation的最大长度是多少？"""
    #return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length = max_len,return_tensors="pt")
    return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length = max_len)

# 对 IMDB 数据集进行 Tokenization
def preprocess_data(dataset):
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    """有坑"""
    """
    datasets.Dataset.map() 内部自动将返回值转换成 Python 列表或 NumPy 数组，
    它会把你的 Tensor 自动转为 list,这是 datasets 库的默认行为，为了兼容其存储结构
    正确做法：必须 显式设置格式为 tensor，不然Embedding那边会报错
    """
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_datasets

# 处理训练集和测试集
# 处理后每一条数据有一个input_ids和attention_mask
train_data = preprocess_data(dataset['train'])
test_data = preprocess_data(dataset['validation'])  # SST-2 没有'test'标签，用'validation'

print(train_data['input_ids'][0])  # 查看第一条数据的 input_ids

# 3. 处理 DataLoader
# 当我们处理完 Tokenization 后，我们可以使用 DataLoader 来迭代数据集，方便后续的训练。
# 什么是DataLoader

# 创建 DataLoader

train_dataloader = DataLoader(train_data, batch_size = DL_batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size = DL_batch_size)

# 每个批次的 input_ids 和 attention_mask 都将是一个 PyTorch 张量，形状为 [batch_size, sequence_length]
# input_ids 是输入的词汇 ID 序列。
# attention_mask 是一个二值矩阵，指示哪些位置是真实的输入（1），哪些是填充的位置（0）

# 获取一批数据并嵌入,仅用于测试，后面训练会在循环中持续推进, 输入编码不管attention_mask
"""batch = next(iter(train_dataloader))
input_ids = batch["input_ids"].to(device)  # Tensor

embed_layer = InputEmbedding(vocab_size, d_model).to(device) #实例化
out = embed_layer(input_ids)
print(out.shape)  # torch.Size([16, 512, 512]) [batch_size, seq_len, d_model]
pos_encoder = PositionalEncoding(d_model, max_len)
out_with_pos = pos_encoder(out)

transformer_encoder = TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_rate)
mask = batch["attention_mask"].to(device)
transformer_encoder_out = transformer_encoder(out_with_pos,mask)"""

model = TransformerClassifier(
    vocab_size, d_model, max_len, n_layers, num_heads, d_ff, dropout_rate, num_classes
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learing_rate)

for epoch in range(num_epochs):
    model.train() #切换到训练模式，启用 dropout/batchnorm 等
    total_loss, total_correct, total_samples = 0, 0, 0 #分别累计总损失、总预测正确样本数、总样本数
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad() #每个batch训练前，先把上一步的梯度清零（否则会累加）
        #前向计算，得到分类logits（shape: [batch, num_classes]）。logits 不是概率，而是原始分数
        logits = model(input_ids, attention_mask)
        #用交叉熵损失函数 criterion，比较模型输出logits与真实标签label，返回一个标量loss
        loss = criterion(logits, labels)
        #loss.backward()：反向传播，计算梯度。optimizer.step()：参数按梯度更新（如SGD/Adam）
        loss.backward()
        optimizer.step()
        
        """loss.item()：取出当前batch损失（标量），乘以样本数后累加（用于计算平均loss）。
        preds = logits.argmax(dim=1)：取最大logit对应的类别作为模型预测结果。
        (preds == labels).sum().item()：统计预测对的数量，累计到total_correct。
        total_samples：累计样本数量"""
        
        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss, val_correct, val_samples = 0, 0, 0
        for batch in test_dataloader:
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_samples += input_ids.size(0)
            
        val_loss /= val_samples
        val_acc = val_correct / val_samples
        print(f"          | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")