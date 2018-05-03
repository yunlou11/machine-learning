# -- coding: utf-8 --
import heapq

from sklearn import datasets, preprocessing
import numpy as np
from scipy import linalg
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA


class MyPCA:
    def __init__(self, n_component=5):
        self.n_components_ = n_component
        self.n_components = n_component
        self.eigen_value = None
        self.eigen_vector = None
        self.explained_variance_ = None

    def eigenvector(self, data):
        """
        获取指定数据的特征值和特征向量
        :param data:
        :return: 排序后的特征值和特征向量
        """
        eigen_value, eigen_vector = np.linalg.eig(data)
        sorted_index = heapq.nlargest(self.n_components_, range(len(eigen_value)), eigen_value.take)
        sort_eigen_vector = []
        for i in sorted_index:
            sort_eigen_vector.append(eigen_vector[:, i])
        return eigen_value[sorted_index], np.array(sort_eigen_vector)

    def fit_transform(self, data):
        return self.fit(data)

    def fit(self, data):
        sample_size = np.shape(data)[0]
        data = preprocessing.scale(data, axis=0, with_mean=True, with_std=False)
        covariance = np.dot(data.T, data) / sample_size
        self.eigen_value, self.eigen_vector = self.eigenvector(covariance)
        data = self.transform(data)
        self.explained_variance_ = np.var(data, axis=0, ddof=1)
        return data

    def transform(self, data):
        reduction_dim_vector = []
        for i in range(self.n_components_):
            reduction_dim_vector.append(np.dot(data, self.eigen_vector[i]))
        return np.array(reduction_dim_vector).T


class MySvdPCA:
    def __init__(self):
        pass


def main():
    digits = datasets.load_digits()
    digits_data = digits.data
    digits_data = np.array([[1., 2., 3.],
                                [2., 4., 6.],
                                [1., 2., 1.],
                                [4., 8., 1.]], dtype=np.float64)
    my_pca = MyPCA(2)
    pca = PCA(2)
    my_pca_data = my_pca.fit_transform(digits_data.T)
    pca_data = pca.fit_transform(digits_data.T)
    print "my_pca data:\n", my_pca_data
    print "pca data:\n", pca_data
    print "my_pca explained_variance_:", my_pca.explained_variance_
    print "pca explained_variance_:", pca.explained_variance_


if __name__ == '__main__':
    main()