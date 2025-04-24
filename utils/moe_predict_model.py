#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from joblib import load
import torch

#####################################
# 全局常量设置
#####################################
EMBED_DIM    = 16     # token embedding 维度
DeepSeek_Router = "/workspace/MoE-ICML/trained_models/tokenid_model_DeepSeek.joblib"
Qwen15_Router = "/workspace/MoE-ICML/trained_models/tokenid_model_Qwen1.5.joblib"
Qwen2_Router = "/workspace/MoE-ICML/trained_models/tokenid_model_Qwen2.joblib"

class SimpleEmbeddingPooler:
    """
    随机初始化 embedding 表，将 token_ids 序列（List[int]）转换为定长向量（平均池化）
    """
    def __init__(self, vocab_size=30000, embed_dim=EMBED_DIM, seed=42):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        np.random.seed(seed)
        self.embedding_table = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim)).astype(np.float32)
        
    def transform(self, token_ids):
        """
        token_ids: List[int]
        返回：形状 (embed_dim,) 的向量（若为空返回零向量）
        """
        if token_ids is None or token_ids.numel() == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        vecs = []

        for tid in token_ids:
            tid = int(tid.item())
            if tid < 0:
                tid = 0
            if tid >= self.vocab_size:
                tid = tid % self.vocab_size
            vecs.append(self.embedding_table[tid])
        return np.mean(vecs, axis=0)

class MoeExpertPredict():
    def __init__(self, model_name, num_experts):
        if model_name == "DeepSeek":
            self.model = load(DeepSeek_Router)
            self.total_layers = 26
        elif model_name == "Qwen1.5":
            self.model = load(Qwen15_Router)
            self.total_layers = 24
        elif model_name == "Qwen2":
            self.model = load(Qwen2_Router)
            self.total_layers = 28
        else:
            self.model = None
            self.total_layers = 0
        # prev layers
        self.prev_rows = self.total_layers - 1
        # embed
        self.embed_pooler = SimpleEmbeddingPooler(vocab_size=30000, embed_dim=EMBED_DIM, seed=42)
        # num_experts
        self.n_experts = num_experts
        # actual_experts_his
        self.actual_experts_his = {i: None for i in range(self.n_experts)}

    def experts_to_onehot(self, experts):
        """tools: make expert lists to one-hot vector (with length=n_experts)"""
        vec = np.zeros(self.n_experts, dtype=np.float32)
        if experts is None:
            return vec
        for e in experts:
            try:
                idx = int(e)
                if 0 <= idx < self.n_experts:
                    vec[idx] = 1.0
            except:
                continue
        return vec
    
    def record_actual_experts(self, layer_idx, actural_experts):
        """record his actual experts for build_features"""
        self.actual_experts_his[layer_idx] = self.experts_to_onehot(actural_experts)


    def build_feature(self, layer_idx, req_embed, num_steps_predict_s, predicted_experts):
        feat_parts = [req_embed.reshape(-1)]  # (16,)
        feat_parts.append(np.array([float(num_steps_predict_s)], dtype=np.float32))  # (1,)
        # 依然保留 predicted experts one-hot（训练目标原来为 predicted-actual），
        pred_onehot = self.experts_to_onehot(predicted_experts)  # (64,)
        feat_parts.append(pred_onehot)
        
        # 前面 全部 层 actual experts 拼接
        prev_matrix = np.zeros((self.prev_rows, self.n_experts), dtype=np.float32)
        for i in range(layer_idx):
            prev_matrix[i, :] = self.actual_experts_his[i]
        feat_parts.append(prev_matrix.flatten())
        
        # layer_idx
        feat_parts.append(np.array([float(layer_idx)], dtype=np.float32))  # (1,)
        feature = np.concatenate(feat_parts, axis=0)

        return feature

    def predict(self, layer_idx, token_ids, num_steps_predict_s, predicted_experts):
        req_embed = self.embed_pooler.transform(token_ids)
        orig_pred = self.experts_to_onehot(predicted_experts)

        # 构造特征向量
        feature = self.build_feature(layer_idx, req_embed, num_steps_predict_s, predicted_experts)
        feature = feature.reshape(1, -1)

        # predict delta
        delta_pred = self.model.predict(feature)[0]
        re_correct = orig_pred - delta_pred     # one-hot code
        re_correct_experts = np.where((re_correct >= 0.5)) # get the selected expert_ids
        return torch.Tensor(re_correct_experts)

if __name__ == "__main__":
    s = MoeExpertPredict("Qwen2", 64)
    s.record_actual_experts(0, [])
    s.record_actual_experts(1, [2, 5, 6, 7, 8, 9, 11, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 29, 30, 31, 32, 34, 36, 39, 40, 43, 44, 47, 48, 49, 50, 51, 54, 55, 56, 57, 60, 61, 63])
    s.record_actual_experts(2, [0, 1, 2, 3, 4, 9, 11, 14, 15, 17, 18, 19, 20, 25, 26, 27, 30, 31, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 54, 55, 57, 59, 60, 62, 63])
    y = s.predict(3, 
                  torch.Tensor([100000,   9081,    245,  20002,    786,    245,   3438,   5946, 285, 8069,12531]), 
                  1, 
                  [0, 1, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 21, 24, 27, 28, 32, 35, 37, 38, 39, 41, 43, 46, 47, 48, 49, 50, 51, 54, 56, 57, 58, 59, 60, 61])
    print(f"predicted: {y}")
    # target is : array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 14, 15, 16, 17,
    #    18, 20, 21, 22, 24, 26, 27, 28, 30, 32, 35, 36, 37, 38, 39, 41, 43,
    #    44, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61])
