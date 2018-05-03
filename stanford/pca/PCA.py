# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()
"""
对样本的协方差做特征分解, 直观上讲 协方差能够表示样本各个分量之间的相关关系,其最大特征向量表明相关程度最高的方向
"""


def get_train_data():
    rng = np.random.RandomState(1)
    a = rng.rand(2, 2)
    b = rng.randn(2, 200)
    ab = np.dot(a, b).T
    return ab


def draw_train_data(X, alpha=1):
    plt.scatter(X[:, 0], X[:, 1], alpha=alpha)


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrow_props = dict(arrowstyle='->',
                       linewidth=2,
                       shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrow_props)


def draw_eigenvector(pca_mean, pca_explained_variance, pca_components):
    for length, vector in zip(pca_explained_variance, pca_components):
        v = vector * np.sqrt(length) * 3
        draw_vector(pca_mean, pca_mean + v)


def pca_train(X):
    pca = PCA(n_components=1)
    x_pca = pca.fit_transform(X)
    return pca, x_pca


def print_pca_coefficient(pca):
    print "主成分分解的特征向量:", pca.components_   # 主成分分解的特征向量
    print "特征方差", pca.explained_variance_   # 主成分分解的特征方差
    print "平均值:", pca.mean_
    print "方差百分比", pca.explained_variance_ratio_


def main():
    X = get_train_data()
    pca, x_pca = pca_train(X)
    print_pca_coefficient(pca)
    draw_eigenvector(pca.mean_, pca.explained_variance_, pca.components_)
    draw_train_data(X, 0.2)
    draw_train_data(pca.inverse_transform(x_pca), 0.8)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()

