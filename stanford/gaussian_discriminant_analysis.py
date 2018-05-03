# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts


def indicator(yi, j):
    """
    指示函数 I(2=2)=1
    :param yi: 训练数据的类别yi
    :param j: 对 theta_j 进行梯度下降时, theta_j对应的类别为 j类
    :return:  yi和 j相等返回1 , 否则返回 0
    """
    res = 0
    if yi == j:
        res = 1
    return res


def draw_scatter_data(data0, data1, color0, color1):
    plt.scatter(data0[:, 0], data0[:, 1], color=color0)
    plt.scatter(data1[:, 0], data1[:, 1], color=color1)


def get_data_by_class(data, category):
    sample_size = np.shape(data)[0]
    category_data = []
    for i in range(sample_size):
        if data[i, 2] == category:
            category_data.append(data[i, :])
    return np.array(category_data)


def load_data(path):
    raw_data = np.loadtxt(path, dtype=float)
    class0_x = get_data_by_class(raw_data, 0)
    class1_x = get_data_by_class(raw_data, 1)
    return class0_x, class1_x


def maximum_likelihood_estimation(class0_data, class1_data):
    class0_size = float(np.shape(class0_data)[0])
    class1_size = float(np.shape(class1_data)[0])
    class0_x = class0_data[:, 0:2]
    class1_x = class1_data[:, 0:2]
    phi = class1_size / (class0_size + class1_size)
    mu0 = np.sum(class0_x, axis=0) / class0_size
    mu1 = np.sum(class1_x, axis=0) / class1_size
    sigma0 = np.cov(class0_x.T)
    sigma1 = np.cov(class1_x.T)
    sigma = (sigma0 + sigma1) / 2
    return phi, mu0, mu1, sigma


def bernoulli_distribution(phi, category):
    return indicator(category, 1) * phi + indicator(category, 0) * (1 - phi)


def predict(x, phi, mu0, mu1, sigma):
    category = 1
    probability_y_0 = sts.multivariate_normal.pdf(x, mu0, sigma) * bernoulli_distribution(phi, 0)
    probability_y_1 = sts.multivariate_normal.pdf(x, mu1, sigma) * bernoulli_distribution(phi, 1)
    if probability_y_0 > probability_y_1:
        category = 0
    return category


def validate(data, phi, mu0, mu1, sigma):
    sample_size = np.shape(data)[0]
    correct_count = 0.0
    for i in range(sample_size):
        predict_y = predict(data[i, 0:2], phi, mu0, mu1, sigma)
        if predict_y == data[i, 2]:
            correct_count += 1
    return correct_count / sample_size


def main():
    print "lai a"
    plt.figure("mode")
    train_class0, train_class1 = load_data("../doc/soft_max_data.txt")
    validate1_0, validate1_1 = load_data("../doc/soft_max_test_data.txt")
    # 训练模型
    phi, mu0, mu1, sigma = maximum_likelihood_estimation(train_class0, train_class1)
    print "phi:", phi, "mu0:", mu0, "mu1:", mu1, "sigma:\n", sigma
    # 根据模型, 预测测试数据1, 满足正态分布, 正确率 0.965
    print "validate:", validate(np.row_stack((validate1_0, validate1_1)), phi, mu0, mu1, sigma)
    # 绘制数据的散点图
    plt.subplot(121)
    draw_scatter_data(train_class0, train_class1, "r", "g")
    plt.subplot(122)
    draw_scatter_data(validate1_0, validate1_1, "r", "g")
    plt.show()

if __name__ == '__main__':
    main()