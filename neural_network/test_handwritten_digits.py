# coding: utf-8
# neural_network/test_handwritten_digits.py
"""手写字符集
"""
import nn
import numpy as np
from sklearn import datasets
from scipy.io import loadmat

# digits = datasets.load_digits()
#
#
# X = digits.images.reshape((len(digits.images), -1))
# y = digits.target.reshape(-1, 1)

data = loadmat('data/handwritten_digits.mat')
Thetas = loadmat('data/ex4weights.mat')
Thetas = [Thetas['Theta1'], Thetas['Theta2']]


X = np.mat(data['X'])
print X.shape
y = np.mat(data['y'])

res = nn.train(X,y,hiddenNum=1,unitNum=25,Thetas=Thetas, precision = 0.5)
print 'Error is: %.4f'%res['error']
