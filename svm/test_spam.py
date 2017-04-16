# coding: utf8
# svm/test_spam.py
import spam
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 垃圾邮件分类器
data = loadmat('data/spamTrain.mat')
X = np.mat(data['X'])
y = data['y']
m, n = X.shape
C = 0.1
tol = 1e-3

# 使用训练集训练分类器
clf = SVC(C=C, kernel='linear', tol=tol)
clf.fit(X, y.ravel())
predictions = np.mat([clf.predict(X[i, :])  for i in range(m)])
accuracy = 100 * np.mean(predictions == y)
print 'Training set accuracy: %0.2f %%' % accuracy

# 使用测试集评估训练结果
data = loadmat('data/spamTest.mat')
XTest = np.mat(data['Xtest'])
yTest = data['ytest']
mTest, _ = XTest.shape

clf.fit(XTest, yTest.ravel())
predictions = np.mat([clf.predict(XTest[i, :])  for i in range(mTest)])
accuracy = 100 * np.mean(predictions == yTest)
print 'Test set accuracy: %0.2f %%' % accuracy

# 获得最能标识垃圾邮件的词汇（在模型中获得高权值的）
weights = abs(clf.coef_.flatten())
top = 15
predictors = spam.getTopPredictors(weights, top)
print '\nTop %d predictors of spam:'%top
for word, weight in predictors:
    print '%-15s (%f)' % (word, weight)

# 使用邮件测试
def genExample(f):
    email = open(f).read()
    indices =  spam.processEmail(email)
    features =  spam.extractFeatures(indices)
    return features

files = [
    'data/emailSample1.txt',
    'data/emailSample1.txt',
    'data/spamSample1.txt',
    'data/spamSample2.txt'
]

emails = np.mat([genExample(f) for f in files], dtype=np.uint8)
labels = np.array([[0, 0, 1, 1]]).reshape(-1, 1)
predictions = np.mat([clf.predict(emails[i, :])  for i in range(len(files))])
accuracy = 100 * np.mean(predictions == labels)
print('\nTest set accuracy for own datasets: %0.2f %%' % accuracy)
