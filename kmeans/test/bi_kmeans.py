# coding: utf-8
import kmeans
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataMat = np.mat(kmeans.loadDataSet('data/testSet2.txt'))
    centroids, clusterAssment = kmeans.kMeans(dataMat, 3)
    clusterCount = np.shape(centroids)[0]
    m = np.shape(dataMat)[0]
    # 绘制散点图
    patterns = ['o', 'D', '^']
    colors = ['b', 'g', 'y']
    fig = plt.figure()
    title = 'bi-kmeans with k=3'
    ax = fig.add_subplot(111, title=title)
    for k in range(clusterCount):
        # 绘制聚类中心
        ax.scatter(centroids[k,0], centroids[k,1], color='r', marker='+', linewidth=20)
        for i in range(m):
            # 绘制属于该聚类中心的样本
            ptsInCluster = dataMat[np.nonzero(clusterAssment[:, 0].A==k)[0]]
            ax.scatter(ptsInCluster[:, 0].flatten().A[0], ptsInCluster[:, 1].flatten().A[0], marker=patterns[k], color=colors[k])
    plt.show()
