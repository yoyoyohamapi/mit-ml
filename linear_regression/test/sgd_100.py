import regression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = regression.loadDataSet('data/ex0.txt');

    rate = 0.01
    maxLoop = 100
    epsilon = 0.0001

    result, timeConsumed = regression.sgd(rate, maxLoop, epsilon, X, y)
    theta,error,iterationCount = result

    fig = plt.figure()
    title = 'sgd: rate=%.2f, maxLoop=%d, epsilon=%.3f \n error: %.5f'%(rate,maxLoop,epsilon,error)
    ax = fig.add_subplot(111, title=title, )
    ax.scatter(X[:, 1].flatten().A[0], y[:,0].flatten().A[0])

    xCopy = X.copy()
    xCopy.sort(0)
    yHat = xCopy*theta
    ax.plot(xCopy[:,1], yHat)
    plt.show()



