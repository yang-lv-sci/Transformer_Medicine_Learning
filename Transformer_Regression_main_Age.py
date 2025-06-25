# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:09:01 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from Transformer_models import TransformerRegression

batch_size = 256
d_model=128
n_layers=2
num_heads=4
d_ff=256
dropout=0.1
learing_rate = 1e-3
num_epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取Excel
file_path = r'../age/Age_Regression/merged_result_v1_joined_dropna.xlsx'
df = pd.read_excel(file_path)
df = df.drop(columns=df.columns[0])  # 删除第一列（序号）

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
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {"features": self.X[idx], "age": self.y[idx]}

train_dataset = TabularDataset(X_train, y_train)
test_dataset  = TabularDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

num_features = X_train.shape[1]
model = TransformerRegression(num_features, d_model, n_layers, num_heads, d_ff, dropout)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learing_rate)

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_true, train_pred = [], []

    for batch in train_loader:
        Xb = batch["features"].to(device)
        yb = batch["age"].to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 累加训练集的预测与真实值（用于统计R等指标）
        train_true.append(yb.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    avg_train_loss = np.mean(train_losses)
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_r, _ = pearsonr(train_true, train_pred)

    # --- 测试集评估 ---
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            Xb = batch["features"].to(device)
            yb = batch["age"].cpu().numpy()
            preds = model(Xb).cpu().numpy()
            y_true.append(yb)
            y_pred.append(preds)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    test_r, _ = pearsonr(y_true, y_pred)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train R: {train_r:.3f} | Test MSE: {mse:.3f} | Test R2: {r2:.3f} | Test R: {test_r:.3f}")