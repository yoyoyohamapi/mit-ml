# coding: utf-8
# linear_regression/test_multiple.py
import regression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

if __name__ == "__main__":
    srcX, y = regression.loadDataSet('data/houses.txt')

    # 新建特征
    m,n= srcX.shape
    X = regression.normalize(srcX.copy())
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    rate = 1
    maxLoop = 50
    epsilon = 1

    result, timeConsumed = regression.bgd(rate, maxLoop, epsilon, X, y)
    theta,errors = result

    print 'theta is:'
    print theta
    print '........'

    # 预测价格
    normalizedSize = (1650-srcX[:,0].mean(0))/srcX[:,0].std(0)
    normalizedBr = (3-srcX[:,1].mean(0))/srcX[:,1].std(0)
    predicateX = np.matrix([[1, normalizedSize, normalizedBr]])
    price = regression.h(theta, predicateX.T)
    print 'Predicted price of a 1650 sq-ft, 3 br house: $%.4f'%price
    print '........'

    # 打印拟合平面
    fittingFig = plt.figure(figsize=(16, 12))
    title = 'polynomial with bgd: rate=%.3f, maxLoop=%d, epsilon=%.3f \n time: %ds'%(rate,maxLoop,epsilon,timeConsumed)
    ax = fittingFig.add_subplot(111, projection='3d', title=title)

    xx = np.linspace(0,5000,25)
    yy = np.linspace(0,5,25)
    zz = np.zeros((25,25))
    for i in range(25):
        for j in range(25):
            normalizedSize = (xx[i]-srcX[:,0].mean(0))/srcX[:,0].std(0)
            normalizedSize = (xx[i]-srcX[:,0].mean(0))/srcX[:,0].std(0)
            x = np.matrix([[1,normalizedSize, normalizedBr]])
            zz[i,j] = regression.h(theta, x.T)
    xx, yy = np.meshgrid(xx,yy)
    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.rainbow, alpha=0.1, antialiased=True)

    xs = srcX[:, 0].flatten().A[0]
    ys = srcX[:, 1].flatten().A[0]
    zs = y[:, 0].flatten().A[0]
    ax.scatter(xs, ys, zs, c='b', marker='o')

    ax.set_xlabel('sq-ft of room')
    ax.set_ylabel('bedrooms')
    ax.set_zlabel('price')

    plt.show()

    # 打印误差曲线
    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')

    plt.show()
