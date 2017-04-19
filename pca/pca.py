# coding: utf8
# pca/pca.py

import numpy as np

def normalize(X):
    """数据标准化处理

    Args:
        X 样本
    Returns:
        XNorm 标准化后的样本
    """
    XNorm = X.copy()
    m,n = XNorm.shape
    mean = np.mean(XNorm, axis=0)
    std = np.std(XNorm, axis=0)
    XNorm = (XNorm - mean) / std
    return XNorm

def PCA(X, k = 1):
    """PCA

    Args:
        X 样本
        k 目的维度
    Returns:
        XNorm 标准化后的样本
        Z 降维后的新样本
        U U
        UReduce UReduce
        S S
        V V
    """
    m, n = X.shape
    # 数据归一化
    XNorm = normalize(X)
    # 计算协方差矩阵
    Coef = XNorm.T * XNorm/m
    # 奇异值分解
    U, S, V = np.linalg.svd(Coef)
    # 取出前 k 个向量
    UReduce = U[:, 0:k]
    Z = XNorm * UReduce
    return XNorm, Z, U, UReduce, S, V

def recover(UReduce, Z):
    """数据恢复

    Args:
        UReduce UReduce
        Z 降维后的样本
    """
    return Z * UReduce.T
