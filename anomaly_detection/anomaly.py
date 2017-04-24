# coding: utf8
# anomaly_detection/anomaly.py

import numpy as np

def F1(predictions, y):
    """F_1Score

    Args:
        predictions 预测
        y 真实值
    Returns:
        F_1Score
    """
    TP = np.sum((predictions == 1) & (y == 1))
    FP = np.sum((predictions == 1) & (y == 0))
    FN = np.sum((predictions == 0) & (y == 1))
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / (TP + FN)
    if precision + recall == 0:
        return 0
    else:
        return (2.0 * precision * recall) / (precision + recall)


def gaussianModel(X):
    """高斯模型

    Args:
        X 样本集
    Returns:
        p 模型
    """
    # 参数估计
    m, n = X.shape
    mu = np.mean(X, axis=0)
    delta2 = np.var(X, axis=0)
    def p(x):
        """p(x)

        Args:
            x x
            mu mu
            delta2 delta2
        Returns:
            p
        """
        total = 1
        for j in range(x.shape[0]):
            total *= np.exp(-np.power((x[j, 0] - mu[0, j]), 2) / (2 * delta2[0, j]**2)
                            ) / (np.sqrt(2 * np.pi * delta2[0, j]))
        return total
    return p


def multivariateGaussianModel(X):
    """多元高斯模型

    Args:
        X 样本集
    Returns:
        p 模型
    """
    # 参数估计
    m, n = X.shape
    mu = np.mean(X.T, axis=1)
    Sigma = np.var(X, axis=0)
    Sigma = np.diagflat(Sigma)
    # Sigma = np.mat(np.cov(X.T))
    detSigma = np.linalg.det(Sigma)

    def p(x):
        """p(x)

        Args:
            x x
            mu mu
            delta2 delta2
        Returns:
            p
        """
        x = x - mu
        return np.exp(-x.T * np.linalg.pinv(Sigma) * x / 2).A[0] * \
            ((2*np.pi)**(-n/2) * (detSigma**(-0.5) ))
    return p


def train(X, model=gaussianModel):
    """训练函数

    Args:
        X 样本集
    Returns:
        p 概率模型
    """
    return model(X)
