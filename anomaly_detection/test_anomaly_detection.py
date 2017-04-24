# coding: utf8
# anomaly_detection/test_anomaly_detection.py

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import anomaly

def selectEpsilon(XVal, yVal, p):
    # 通过交叉验证集，选择最好的 epsilon 参数
    pVal = np.mat([p(x.T) for x in XVal]).reshape(-1, 1)
    step = (np.max(pVal) - np.min(pVal)) / 1000
    bestEpsilon = 0
    bestF1 = 0
    for epsilon in np.arange(np.min(pVal), np.max(pVal), step):
        predictions = pVal < epsilon
        F1 = anomaly.F1(predictions, yVal)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

# 小维度测试......
data = loadmat('data/ex8data1.mat')
X = np.mat(data['X'])
XVal = np.mat(data['Xval'])
yVal = np.mat(data['yval'])

# p = anomaly.train(X)
p = anomaly.train(X, model=anomaly.multivariateGaussianModel)
pTest = np.mat([p(x.T) for x in X]).reshape(-1, 1)

# 绘制数据点
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.plot(X[:, 0], X[:, 1], 'bx')
epsilon, F1 = selectEpsilon(XVal, yVal, p)

print 'Best epsilon found using cross-validation: %e\n'%epsilon
print 'Best F1 on Cross Validation Set:  %f\n'%F1
print '# Outliers found: %d' % np.sum(pTest < epsilon)

# 获得训练集的异常点
outliers = np.where(pTest < epsilon, True, False).ravel()
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', lw=2, markersize=10, fillstyle='none', markeredgewidth=1)
n = np.linspace(0, 35, 100)
X1 = np.meshgrid(n,n)
XFit = np.mat(np.column_stack((X1[0].T.flatten(), X1[1].T.flatten())))
pFit = np.mat([p(x.T) for x in XFit]).reshape(-1, 1)
pFit = pFit.reshape(X1[0].shape)
# Do not plot if there are infinities
if not np.isinf(np.sum(pFit)):
    plt.contour(X1[0], X1[1], pFit, 10.0**np.arange(-20, 0, 3).T)
plt.show()


# 大维度测试......
data = loadmat('data/ex8data2.mat')
X = np.mat(data['X'])
XVal = np.mat(data['Xval'])
yVal = np.mat(data['yval'])

# p = anomaly.train(X)
p = anomaly.train(X, model=anomaly.multivariateGaussianModel)
pTest = np.mat([p(x.T) for x in X]).reshape(-1, 1)

epsilon, F1 = selectEpsilon(XVal, yVal, p)

print 'Best epsilon found using cross-validation: %e\n'%epsilon
print 'Best F1 on Cross Validation Set:  %f\n'%F1
print '# Outliers found: %d' % np.sum(pTest < epsilon)
