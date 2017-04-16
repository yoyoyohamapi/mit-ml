# coding: utf8
# svm/test_diagnose.py

import numpy as np
import smo
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('data/ex6data3.mat')

X = np.mat(data['X'], dtype=np.float)
y = np.mat(data['y'], dtype=np.float)
XVal = np.mat(data['Xval'], dtype=np.float)
yVal = np.mat(data['yval'], dtype=np.float)

m, n = X.shape
mVal, _ = XVal.shape

# 纠正负样本
y[y == 0] = -1
yVal[yVal == 0] = -1

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
deltas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

# 获得所有 C 及 delta 的组合
deltaCPairs = [[delta, C] for C in Cs for delta in deltas]

# 获得训练模型
tol = 1e-3
maxIter = 5
models = [smo.getSmo(X, y, C, tol, maxIter, kernel=smo.rbfKernel(delta))
          for delta, C in deltaCPairs]

# 开始训练
results = [train() for train, _, _  in models]

# 利用交叉验证集选择模型
predictions = [models[idx][2](XVal, alphas, b, supportVectorsIndex, supportVectors)
    for idx, (alphas, w, b, supportVectorsIndex, supportVectors, iterCount) in enumerate(results)]
errorRates = [(np.multiply(prediction, yVal).A < 0).sum() /
              float(mVal) for prediction in predictions]
minIdx = np.argmin(errorRates)
alphas, w, b, supportVectorsIndex, supportVectors, iterCount = results[minIdx]
delta, C = deltaCPairs[minIdx]

# 绘制数据点
x1Min = X[:, 0].min()
x1Max = X[:, 0].max()
x2Min = X[:, 1].min()
x2Max = X[:, 1].max()
plt.title(r'C=%.2f, $\delta$=%.2f, error=%.2f'%(C, delta, errorRates[minIdx]))
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x1Min, x1Max)
plt.ylim(x2Min, x2Max)

for i in range(m):
    x = X[i].A[0]
    if y[i] == 1:
        color = 'black'
        if i in supportVectorsIndex:
            color = 'red'
        plt.scatter(x[0], x[1], marker='*', color=color, s=50)
    else:
        color = 'green'
        if i in supportVectorsIndex:
            color = 'red'
        plt.scatter(x[0], x[1], marker='o', color=color, s=50)


# 绘制决策边界
xx1, xx2 = np.meshgrid(
    np.linspace(x1Min, x1Max, 100),
    np.linspace(x2Min, x2Max, 100)
)
_, _, predict = models[minIdx]
predictX = np.mat(np.c_[xx1.ravel(), xx2.ravel()])
predictions = predict(predictX, alphas, b, supportVectorsIndex, supportVectors)
predictions = predictions.reshape(xx1.shape)
plt.contour(xx1, xx2, predictions, [0.5], linewidths=5)
plt.show()
