# -- coding: utf-8 --
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt


def one_normal_distribute():
    """
    产生一元正态分布的样本数据,并绘制柱状图
    :return:
    """
    np.random.seed(0)
    mu = 2.0
    sigma = 0.1
    sample = np.random.normal(mu, sigma, 100)
    plt.hist(sample, 30, normed=True)
    return sample


def get_one_normal_from_sample(sample):
    """
    根据样本值估计正则分布的参数
    求方差无偏估计时 除以 sample_size -1, 在数据量多的时候无所谓减不减 1
    :param sample: 样本数据
    :return:
    """
    sample_size = np.shape(sample)[0]
    mu = sum(sample) / sample_size
    sigma_2 = np.dot((sample - mu), (sample - mu)) / (sample_size - 1)
    return mu, np.sqrt(sigma_2)


def two_normal_distribute(mu, sigma):
    """
    生成二元正态分布数据样本以及绘制图形
    cholesky: 楚列斯基分解, 用于对称矩阵的分解 对称矩阵 A = U*U.T
    生成二维正态分布数据也可以使用 np.random.multivariate_normal(mu, sigma, sample_size)
    使用如下方式的原理是 x~N(0,I), Y= U*A + mu, 则 Y~N(mu, U*U.T)
    :return:
    """

    u = cholesky(sigma)
    print "A=U*U.T:\n", np.dot(u, u.T)
    sample_standard = np.random.randn(1000, 2)
    sample = np.dot(sample_standard, u) + mu
    plt.plot(sample[:, 0], sample[:, 1], "og")
    return sample


def get_two_normal_from_sample(sample):
    sample_size = np.shape(sample)[0]
    mu = np.sum(sample, axis=0) / sample_size
    sigma = np.cov(sample.T)
    x1 = sample[:, 0]
    x2 = sample[:, 1]
    cov_x1_x1 = np.dot((x1 - mu[0, 0]).T, (x1 - mu[0, 0])) / (sample_size - 1)
    cov_x1_x2 = np.dot((x1 - mu[0, 1]).T, (x2 - mu[0, 1])) / (sample_size - 1)
    print "cov_1_2:", cov_x1_x1, cov_x1_x2
    return mu, sigma


def main():
    print 'hello'
    plt.figure("mode")
    plt.subplot(131)
    one_sample = one_normal_distribute()
    one_mu, one_sigma = get_one_normal_from_sample(one_sample)
    print one_mu, one_sigma
    sigma = np.mat([[1, 1.5],
                    [1.5, 3]])
    mu = [5, 7]
    plt.subplot(132)
    two_sample = two_normal_distribute(mu, sigma)
    two_mu, two_sigma = get_two_normal_from_sample(two_sample)
    print "two_mu:\n", two_mu, "\ntow_sigma:\n", two_sigma
    plt.subplot(133)
    two_normal_distribute(two_mu, two_sigma)
    plt.show()
if __name__ == '__main__':
    main()