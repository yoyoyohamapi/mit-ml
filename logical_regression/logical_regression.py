# coding: utf-8
# logical_regression/logical_regression.py
import numpy as np
import matplotlib as plt
import time

def exeTime(func):
    """耗时计算装饰器

    Args:
        func 待装饰函数
    Returns:
        newFunc 装饰后的函数
    """
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc

def loadDataSet(filename):
    """读取数据集
    数据以TAB进行分割

    Args:
        filename 文件名
    Returns:
        X 训练样本集矩阵
        y 标签集矩阵
    """
    numFeat = len(open(filename).readline().split('\t')) - 1
    X = []
    y = []
    file = open(filename)
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        X.append([1.0, float(lineArr[0]), float(lineArr[1])])
        y.append(float(curLine[-1]))
    return np.mat(X), np.mat(y).T

def sigmoid(z):
    """sigmoid函数
    """
    return 1.0/(1.0+np.exp(-z))

def J(theta, X, y, theLambda=0):
    """预测代价函数
    """
    m, n = X.shape
    h = sigmoid(X.dot(theta))
    J = (-1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (theLambda/(2.0*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return(np.inf)
    return J.flatten()[0,0]

@exeTime
def gradient(X, y, options):
    """随机梯度下降法
    Args:
        X 样本矩阵
        y 标签矩阵
        rate 学习率
        options.theLambda 正规参数
        options.maxLoop 最大迭代次数
        options.epsilon 收敛精度
        options.method
            - 'sgd' 随机梯度下降法
            - 'bgd' 批量梯度下降法
    Returns:
        (thetas, errors), timeConsumed
    """
    m,n = X.shape
    # 初始化参数矩阵
    theta = np.ones((n,1))
    count = 0 # 迭代次数
    # 初始化误差无限大
    error = float('inf')
    # 保存误差变化状况
    errors = []
    # 保存参数的变化状况
    thetas = []
    rate = options.get('rate', 0.01)
    epsilon = options.get('epsilon', 0.1)
    maxLoop = options.get('maxLoop', 1000)
    theLambda = options.get('theLambda', 0)
    method = options['method']
    def _sgd(theta):
        converged = False
        for i in range(maxLoop):
            if converged:
                break
            for j in range(m):
                h = sigmoid(X[j] *theta)
                diff = h - y[j]
                theta = theta - rate*(1.0/m)*X[j].T*diff
                error = J(theta, X, y)
                errors.append(error)
                if error < epsilon:
                    converged = True
                    break
                thetas.append(theta)
        return thetas, errors, i+1
    def _bgd(theta):
        for i in range(maxLoop):
            h = sigmoid(X.dot(theta))
            diff = h - y
            # theta0 should not be regularized
            theta = theta - rate*((1.0/m)*X.T*diff + (theLambda/m)*np.r_[[[0]], theta[1:]])
            error = J(theta, X, y, theLambda)
            errors.append(error)
            if error < epsilon:
                break
            thetas.append(theta)
        return thetas, errors, i+1
    methods = {
        'sgd': _sgd,
        'bgd': _bgd
    }
    return methods[method](theta)
