# coding: utf-8
# logical_regression/test_onevsall.py
"""OneVsAll 多分类测试
"""
import numpy as np
import logical_regression as regression
from scipy.io import loadmat

if __name__ == "__main__":
    data = loadmat('data/ex3data1.mat')
    X = np.mat(data['X'])
    y = np.mat(data['y'])
    # 为X添加偏置
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    # 采用批量梯度下降法
    options = {
        'rate': 0.1,
        'epsilon': 0.1,
        'maxLoop': 5000,
        'method': 'bgd'
    }
    # 训练
    Thetas = regression.oneVsAll(X,y,options)
    # 预测
    H = regression.predictOneVsAll(X, Thetas)
    pred = np.argmax(H,axis=0)+1
    # 计算准确率
    print 'Training accuracy is: %.2f%'%(np.mean(pred == y.ravel())*100)
