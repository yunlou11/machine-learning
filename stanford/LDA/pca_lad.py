# -- coding:utf-8 --
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PLSample:
    def __init__(self):
        pass

    @staticmethod
    def two_class_data_1():
        path = "two_class_data_1.txt"
        if not os.path.exists(path):
            sample_size = 100
            X1 = []
            X2 = []
            X1.append(np.random.normal(loc=5.0, scale=0.05, size=sample_size))
            X1.append(np.random.normal(loc=5.0, scale=10.0, size=sample_size))
            X1.append([1] * sample_size)
            X1 = np.array(X1).T
            X2.append(np.random.normal(loc=6.0, scale=0.05, size=sample_size))
            X2.append(np.random.normal(loc=5.0, scale=10.0, size=sample_size))
            X2.append([2] * sample_size)
            X2 = np.array(X2).T
            data = np.vstack((X1, X2))
            np.random.shuffle(data)
            np.savetxt(path, data)
            print "new Data"
        else:
            data = np.loadtxt(path, dtype=float)
        return data

    @staticmethod
    def draw_data(data):
        plt.figure("data")
        c = data[:, 2]
        plt.scatter(data[:, 0], data[:, 1], c=c, cmap=plt.cm.Paired)

    @staticmethod
    def draw_pca_data(data, lable):
        plt.figure("pca data")
        sample_size = np.shape(data)[0]
        plt.scatter(x=data, y=[0] * sample_size, c=lable, cmap=plt.cm.Paired)


def main():
    pca = PCA(1)
    pl_sample = PLSample()
    data = pl_sample.two_class_data_1()
    pl_sample.draw_data(data)
    pca_data = pca.fit_transform(data[:, 0:2])
    pl_sample.draw_pca_data(pca_data, data[:, 2])
    print pca.components_
    plt.show()
if __name__ == '__main__':
    main()