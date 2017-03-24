# coding: utf-8
# algorithm_analysis/test_datasets_divide.py
"""数据集划分
"""
import linear_regression
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = loadmat('data/water.mat')
# 训练集
X = np.mat(data['X'])
# 为X添加偏置
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
y = np.mat(data['y'])
# 交叉验证集
Xval = np.mat(data['Xval'])
Xval = np.concatenate((np.ones((Xval.shape[0], 1)), Xval), axis=1)
yval = np.mat(data['yval'])
# 测试集
Xtest = np.mat(data['Xtest'])
Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)
ytest = np.mat(data['ytest'])

rate = 0.001
maxLoop = 5000
epsilon = 0.1
# initTheta = np.mat(np.ones((X.shape[1], 1)))
# result, timeConsumed = linear_regression.gradient(
#     X, y, rate=rate, maxLoop=maxLoop, epsilon=epsilon, initTheta=initTheta)
# theta, errors = result
#
# # 绘制拟合成果
# title = 'bgd: rate=%.3f, maxLoop=%d, epsilon=%.3f \n time: %.2fms, error=%.3f' % (
#     rate, maxLoop, epsilon, timeConsumed/1000.0, errors[-1])
# Xmin = X[:, 1].min()
# Xmax = X[:, 1].max()
# ymax = y[:, 0].max()
# ymin = y[:, 0].min()
# fitX = np.mat(np.linspace(Xmin, Xmax, 20).reshape(-1, 1))
# fitX = np.concatenate((np.ones((fitX.shape[0], 1)), fitX), axis=1)
# h = fitX * theta
# plt.xlim(Xmin, Xmax)
# plt.ylim(ymin, ymax)
# plt.title(title)
# # 绘制训练样本
# plt.scatter(X[:, 1].flatten().A[0], y[:, 0].flatten().A[0])
# # 绘制拟合曲线
# plt.plot(fitX[:, 1], h, color='g')
# plt.xlabel('Change in water level(x)')
# plt.ylabel('Water flowing out of the dam(y)')
# plt.show()

# 绘制随样本规模学习曲线
# m, n = X.shape
# trainErrors = np.zeros((1,m))
# valErrors = np.zeros((1,m))
# for i in range(m):
#     Xtrain = X[0:i+1]
#     ytrain = y[0:i+1]
#     res, timeConsumed = linear_regression.gradient(
#         Xtrain, ytrain, rate=rate, maxLoop=maxLoop, epsilon=epsilon)
#     theta, errors = res
#     trainErrors[0,i] = errors[-1]
#     valErrors[0,i] = linear_regression.J(theta, Xval, yval)
#
# plt.plot(np.arange(1,m+1).ravel(), trainErrors.ravel(), color='b', label='Training Error')
# plt.plot(np.arange(1,m+1).ravel(), valErrors.ravel(), color='g', label='Validation Error')
#
# plt.title('Learning curve for linear regression')
# plt.xlabel('Number of training examples')
# plt.ylabel('Error')
# plt.legend()
# plt.show()

# 多项式回归
poly = PolynomialFeatures(degree=8)
XX,XXval,XXtest = [linear_regression.normalize(np.mat(poly.fit_transform(data[:,1:]))) for data in [X,Xval,Xtest]]
initTheta = np.mat(np.ones((XX.shape[1], 1)))
# res, timeConsumed = linear_regression.gradient(XX, y, rate=1, maxLoop=5000, epsilon=0.01, theLambda=100)
# theta, errors = res
# print errors[-1]
#
#
# # 绘制拟合曲线
# fitX = np.mat(np.linspace(-60,45).reshape(-1, 1))
# fitX = np.concatenate((np.ones((fitX.shape[0], 1)), fitX), axis=1)
# fitXX = linear_regression.normalize(np.mat(poly.fit_transform(fitX[:, 1:])))
# h = fitXX * theta
# plt.scatter(X[:, 1].ravel(), y[:, 0].flatten().A[0])
# plt.plot(fitX[:, 1], h, color='g')
# plt.show()

theLambdas = [0, 0.001,0.003,0.01,0.003,0.1,0.3,1,3,10,100]
numTheLambdas = len(theLambdas)
trainErrors = np.zeros((1,numTheLambdas))
valErrors = np.zeros((1,numTheLambdas))
for idx, theLambda in enumerate(theLambdas):
    res, timeConsumed = linear_regression.gradient(XX, y, rate=0.3, maxLoop=500, epsilon=0.01, theLambda=theLambda)
    theta, errors = res
    trainErrors[0, idx] = errors[-1]
    valErrors[0,idx] = linear_regression.J(theta, XXval, yval, theLambda = theLambda)
print valErrors
# # 绘制随样本规模学习曲线
# trainErrors, valErrors = linear_regression.getLearningCurves(XX, y, XXval, yval, rate=0.1, maxLoop=5000, epsilon=0.01, theLambda=100)
# m,n = XX.shape
plt.plot(np.arange(1, numTheLambdas+1).ravel(), trainErrors.ravel(), color='b', label='Training Error')
plt.plot(np.arange(1, numTheLambdas+1).ravel(), valErrors.ravel(), color='g', label='Validation Error')
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.show()
