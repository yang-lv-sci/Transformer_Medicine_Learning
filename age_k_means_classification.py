# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 16:46:03 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report,roc_auc_score, matthews_corrcoef)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False
    print("⚠️  未检测到 xgboost，已自动跳过该模型。")

# ---------------- ① 读取数据 ----------------
file_path = 'merged_result_v1_joined_dropna.xlsx'      
df = pd.read_excel(file_path)
df = df.drop(columns=df.columns[0])                    # 删除序号列

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

# 如需查看每个聚类对应的年龄区间：
for c in sorted(df['Age'].unique()):
    print(f"Cluster {c}: {df.loc[df['Age']==c,'Age'].shape[0]} 样本，"
          f"年龄范围 [{age_array[df['Age']==c].min():.1f}, {age_array[df['Age']==c].max():.1f}]")

# ---------------- ③ 特征工程 ----------------
categorical_cols = [
    'Pulse_regular', 'Pulse_type',
    'Enhancement_used_first_reading',
    'Enhancement_used_second_reading',
    'Enhancement_used_third_reading',
    'Sex'
]

y = df['Age']                       # 目标 = 聚类编号
X = df.drop(columns=['Age'])        # 其余列做特征

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
feature_names = X_encoded.columns

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- ④ 定义模型 ----------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=100, multi_class='multinomial'),
    "Random Forest":       RandomForestClassifier(n_estimators=25, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "SVC (RBF)":           SVC(probability=True, kernel='rbf', random_state=42)
}

if xgb_installed:
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42
    )

# ---------------- ⑤ 训练 & 评估 ----------------
results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)

    for split, X_split, y_split in [("Train", X_train, y_train),
                                    ("Test",  X_test,  y_test)]:
        y_pred = clf.predict(X_split)

        # ---------- 基础四项 ----------
        acc = accuracy_score(y_split, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_split, y_pred, average='weighted', zero_division=0)

        # ---------- AUC ----------
        # 1) 先拿到 y_score
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_split)
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_split)
            # decision_function 在二分类时是一维，需要扩成二维:
            if y_score.ndim == 1:
                y_score = np.vstack([1 - y_score, y_score]).T
        else:
            y_score = None

        # 2) 计算 AUC（如不可算则设为 nan）
        if y_score is not None:
            if len(np.unique(y)) == 2:     # 二分类
                auc = roc_auc_score(y_split, y_score[:, 1])
            else:                          # 多分类
                auc = roc_auc_score(y_split, y_score,
                                    multi_class='ovr', average='weighted')
        else:
            auc = np.nan

        # ---------- MCC ----------
        mcc = matthews_corrcoef(y_split, y_pred)

        # ---------- 记录 ----------
        results.append({
            "Model": name, "Split": split,
            "Accuracy": acc, "Precision": precision,
            "Recall": recall, "F1-score": f1,
            "AUC": auc, "MCC": mcc
        })

        # ---------- 混淆矩阵 ----------
        cm = confusion_matrix(y_split, y_pred,
                              labels=sorted(df['Age'].unique()))
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', square=True,
                    xticklabels=sorted(df['Age'].unique()),
                    yticklabels=sorted(df['Age'].unique()))
        plt.title(f"{name} – {split} Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        plt.show()

# ---------------- ⑥ 汇总结果 ----------------
results_df = pd.DataFrame(results).round(4)
print("\n=== 综合指标 ===")
print(results_df.pivot(index="Model", columns="Split"))