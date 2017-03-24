# coding: utf-8
# algorithm_analysis/linear_regression.py
import numpy as np
import matplotlib as plt
import time


def exeTime(func):
    """ 耗时计算装饰器
    """
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc


def h(theta, x):
    """预测函数

    Args:
        theta 相关系数矩阵
        x 特征向量

    Returns:
        预测结果
    """
    return (theta.T * x)[0, 0]


def J(theta, X, y, theLambda=0):
    """代价函数

    Args:
        theta 相关系数矩阵
        X 样本集矩阵
        y 标签集矩阵

    Returns:
        预测误差（代价）
    """
    m = len(X)
    return (X * theta - y).T * (X * theta - y) / (2 * m) + theLambda * np.sum(np.square(theta)) / (2*m)


@exeTime
def gradient(X, y, rate=1, maxLoop=50, epsilon=1e-1, theLambda=0, initTheta=None):
    """批量梯度下降法

    Args:
        X 样本矩阵
        y 标签矩阵
        rate 学习率
        maxLoop 最大迭代次数
        epsilon 收敛精度
        theLambda 正规化参数
    Returns:
        (theta, errors), timeConsumed
    """
    m, n = X.shape
    # 初始化theta
    if initTheta is None:
        theta = np.zeros((n, 1))
    else:
        theta = initTheta
    count = 0
    converged = False
    error = float('inf')
    errors = []
    for i in range(maxLoop):
        theta = theta + (1.0 / m) * rate * ((y - X * theta).T * X).T
        error = J(theta, X, y, theLambda)
        if np.isnan(error) is True:
            error = np.inf
        else:
            error = error[0, 0]
        errors.append(error)
        # 如果已经收敛
        if(error < epsilon):
            break
    return theta, errors

def standardize(X):
    """特征标准化处理

    Args:
        X 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X

def normalize(X):
    """特征归一化处理

    Args:
        X 样本集
    Returns:
        归一化后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           X[:,j] = (features-minVal)/diff
        else:
           X[:,j] = 0
    return X

def getLearningCurves(X, y, Xval, yval, rate=1, maxLoop=50, epsilon=0.1, theLambda=0):
    """获得学习曲线

    Args:
        X 样本集
        y 标签集
        Xval 交叉验证集
        yval 交叉验证集标签
    Returns:
        trainErrors 训练误差随样本规模的变化
        valErrors 校验验证集误差随样本规模的变化
    """
    # 绘制随样本规模学习曲线
    m, n = X.shape
    trainErrors = np.zeros((1,m))
    valErrors = np.zeros((1,m))
    for i in range(m):
        Xtrain = X[0:i+1]
        ytrain = y[0:i+1]
        res, timeConsumed = gradient(
            Xtrain, ytrain, rate=rate, maxLoop=maxLoop, epsilon=epsilon,theLambda=theLambda)
        theta, errors = res
        trainErrors[0,i] = errors[-1]
        valErrors[0,i] = J(theta, Xval, yval, theLambda=theLambda)
    return trainErrors, valErrors
