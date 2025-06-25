# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:19:54 2025

@author: lvyang
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Transformer_models import TransformerFeatureClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

batch_size = 1024
d_model=256
n_layers=2
num_heads=8
d_ff=d_model*2
dropout=0.5
learing_rate = 2e-4
num_epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取Excel
file_path = r'../age/Age_Regression/merged_result_v1_joined_dropna.xlsx'
df = pd.read_excel(file_path)
df = df.drop(columns=df.columns[0])  # 删除第一列（序号）

# ---------------- ② K-Means 聚类 Age ----------------
age_array = df['Age'].values.reshape(-1, 1)

k_candidates = range(2, 11)         # 探索 k=2~10
sil_scores = []

for k in k_candidates:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(age_array)
    sil_scores.append(silhouette_score(age_array, labels))

best_k = k_candidates[int(np.argmax(sil_scores))]
print(f"✅ 轮廓系数最优 k = {best_k}, 分值 {max(sil_scores):.4f}")

# 最终聚类
km_final = KMeans(n_clusters=best_k, random_state=42)
df['Age'] = km_final.fit_predict(age_array)

y = df['Age'].values
X = df.drop(columns=['Age'])

categorical_cols = [
    'Pulse_regular',
    'Pulse_type',
    'Enhancement_used_first_reading',
    'Enhancement_used_second_reading',
    'Enhancement_used_third_reading',
    'Sex'
]
X_encoded = pd.get_dummies(X, columns=categorical_cols)
feature_names = X_encoded.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]     

train_dataset = TabularDataset(X_train, y_train)
test_dataset  = TabularDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

num_features = X_train.shape[1]
model = TransformerFeatureClassifier(num_features, d_model, n_layers,
                              num_heads, d_ff, dropout, num_classes=best_k).to(device)

criterion = nn.CrossEntropyLoss()      
optimizer = torch.optim.Adam(model.parameters(), lr= learing_rate)

for epoch in range(num_epochs):
    # ---------- 训练 ----------
    model.train()
    train_losses = []
    y_true_tr, y_pred_tr, y_prob_tr = [], [], []

    for batch in train_loader:
        Xb = batch[0].to(device)
        yb = batch[1].to(device)               # long tensor, 0‒(k-1)

        optimizer.zero_grad()
        logits = model(Xb)                         # (B, k)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        y_true_tr.append(yb.cpu().numpy())
        y_pred_tr.append(logits.argmax(1).cpu().numpy())          # 预测标签
        y_prob_tr.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())  # 预测概率

    # ---------- 训练指标 ----------
    y_true_tr = np.concatenate(y_true_tr, axis=0)
    y_pred_tr = np.concatenate(y_pred_tr, axis=0)
    y_prob_tr = np.concatenate(y_prob_tr, axis=0)
    
    acc_tr  = accuracy_score(y_true_tr, y_pred_tr)
    pre_tr, rec_tr, f1_tr, _ = precision_recall_fscore_support(
        y_true_tr, y_pred_tr, average='weighted', zero_division=0)
    mcc_tr  = matthews_corrcoef(y_true_tr, y_pred_tr)
    if best_k == 2:
        auc_tr = roc_auc_score(y_true_tr, y_prob_tr[:, 1])
    else:
        auc_tr = roc_auc_score(y_true_tr, y_prob_tr,
                               multi_class='ovr', average='weighted')

    # ---------- 测试 ----------
    model.eval()
    y_true_te, y_pred_te, y_prob_te = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            Xb = batch[0].to(device)
            yb = batch[1].to(device)

            logits = model(Xb)
            y_true_te.append(yb.cpu().numpy())
            y_pred_te.append(logits.argmax(1).cpu().numpy())
            y_prob_te.append(torch.softmax(logits, 1).cpu().numpy())

    y_true_te = np.concatenate(y_true_te)
    y_pred_te = np.concatenate(y_pred_te)
    y_prob_te = np.concatenate(y_prob_te)

    acc_te  = accuracy_score(y_true_te, y_pred_te)
    pre_te, rec_te, f1_te, _ = precision_recall_fscore_support(
        y_true_te, y_pred_te, average='weighted', zero_division=0)
    mcc_te  = matthews_corrcoef(y_true_te, y_pred_te)
    if best_k == 2:
        auc_te = roc_auc_score(y_true_te, y_prob_te[:, 1])
    else:
        auc_te = roc_auc_score(y_true_te, y_prob_te,
                               multi_class='ovr', average='weighted')

    # ---------- 打印 ----------
    print(f"Epoch {epoch+1:>3d} | "
      f"Train Loss {np.mean(train_losses):.4f} | "
      f"Train Acc {acc_tr:.3f} | Train Prec {pre_tr:.3f} | Train Rec {rec_tr:.3f} | "
      f"Train F1 {f1_tr:.3f} | Train AUC {auc_tr:.3f} | Train MCC {mcc_tr:.3f} || "
      f"Test Acc {acc_te:.3f} | Test Prec {pre_te:.3f} | Test Rec {rec_te:.3f} | "
      f"Test F1 {f1_te:.3f} | Test AUC {auc_te:.3f} | Test MCC {mcc_te:.3f}")

# ---------- 终轮混淆矩阵 ----------
cm = confusion_matrix(y_true_te, y_pred_te, labels=np.arange(best_k))
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', square=True,
            xticklabels=np.arange(best_k), yticklabels=np.arange(best_k))
plt.title("Test Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()