# coding: utf8
# recommender_system/recommender.py
import numpy as np
from scipy.optimize import minimize, check_grad
from pydash import py_

def getRecommender(Y, R, params=None, n=10, theLambda=10, maxIter=100):
    """训练方法

    Args:
        Y 评价矩阵
        R 是否评价矩阵
        params 是否具有初始化参数
        n 商品特征数
        theLambda 正规化参数
        maxIter 最大迭代次数
    Returns:
        train 训练方法
        predict 预测方法
    """

    # 商品数，用户数
    nm, nu = Y.shape

    # normalize YMean
    mu = np.mean(Y, axis=0)
    mu = np.zeros((Y.shape[0], 1), dtype=np.float)
    for i in range(nm):
        totalRates = np.sum(Y[i])
        validCount = len(np.nonzero(R[i])[0])
        mu[i] = totalRates / validCount
    Y = Y - mu

    def unroll(Theta, X):
        """参数折叠

        Args:
            Theta 用户偏好矩阵
            X 商品内容矩阵
        Returns:
            vector 折叠后的参数
        """

        return np.hstack((X.A.T.flatten(), Theta.A.T.flatten()))

    def roll(vector):
        """参数回复

        Args:
            vector 参数向量
        Returns:
            Theta 用户偏好矩阵
            X 商品内容矩阵
        """
        X = np.mat(vector[:nm * n].reshape(n, nm).T)
        Theta = np.mat(vector[nm * n:].reshape(n, nu).T)
        return Theta, X

    def initParams():
        """初始化参数

        Returns:
            Theta 用户对内容的偏好矩阵
            X 商品内容矩阵
        """
        Theta = np.mat(np.random.rand(nu, n))
        X = np.mat(np.random.rand(nm, n))
        return Theta, X

    def regParams(param):
        """正规化参数
        Args:
            param 参数
        Return:
            regParam 正规化后的参数
        """
        return theLambda * 0.5 * np.sum(np.power(param, 2))

    def J(params):
        """代价函数

        Args:
            params 参数向量
            nu 用户数
            nm 商品数
            n 特征数
        Return:
            J 预测代价
        """
        # 参数展开
        Theta, X = roll(params)
        # 计算误差
        rows, cols = np.nonzero(R)
        # 预测
        H = predict(Theta, X)
        Diff = H - Y
        Diff[R != 1] = 0
        error = 0.5 * np.sum(np.power(Diff, 2))
        #  正规化 Theta
        regTheta = regParams(Theta)
        #  正规化 x
        regX = regParams(X)
        return error + regTheta + regX

    def gradient(params):
        """梯度下降

        Args:
            params 参数向量
        Returns:
            grad 梯度向量
        """
        Theta, X = roll(params)
        ThetaGrad = np.mat(np.zeros(Theta.shape))
        XGrad = np.mat(np.zeros(X.shape))
        error = predict(Theta, X) - Y
        error[R != 1] = 0
        ThetaGrad = error.T * X + theLambda * Theta
        XGrad =  error * Theta + theLambda * X
        return unroll(ThetaGrad, XGrad)

    def train():
        """训练方法

        Returns:
            Theta 用户的偏好矩阵
            X 商品的内容矩阵
        """
        # 初始化参数
        if not params:
            Theta, X = initParams()
        else:
            Theta = params['Theta']
            X = params['X']
        # 最小化目标函数
        res = minimize(J, x0=unroll(Theta, X), jac=gradient,
                       method='CG', options={'disp': True, 'maxiter': maxIter})
        Theta, X = roll(res.x)
        return Theta, X

    def predict(Theta, X):
        """预测
        Args:
            Theta 用户对内容的偏好矩阵
            X 商品内容矩阵
        Return:
            h 预测
        """
        return X * Theta.T + mu

    def getTopRecommends(Theta, X, i, count, rated, items):
        """获得推荐

        Args:
            Theta Theta
            X X
            i 用户下标
            count 获得推荐的数目
            rated 已经评价的类目id
            items 商品清单
        Returns:
            topRecommends 推荐项目
        """
        predictions = predict(Theta, X)[:, i]
        return py_(items) \
            .map(lambda item, idx: (item, predictions[idx])) \
            .sort_by(lambda item: item[1], reverse = True) \
            .take(count) \
            .value()

    return train, predict, getTopRecommends
