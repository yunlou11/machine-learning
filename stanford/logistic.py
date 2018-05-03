# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt


def func_sigmoid(x):
    sig = 1.0 / (1 + np.exp(-x))
    return sig


def draw_split_line(theta):
    """
    hθ(z) = logistic(z)
    z = θT * x  z:特征空间  x:输入空间
    θT * X =0 唯一确定一个超平面L, 使数据线性可分, 此时 L平面上方数据对应 z > 0, L平面下方数据对应z < 0
    :param theta:
    :return:
    """
    x = np.linspace(-10, 10, 50)
    vector_x = np.row_stack((np.ones((1, 50)), np.mat(x)))
    y = -((np.dot(theta[0, 0:2], vector_x)) / theta[0, 2])
    plt.plot(x, y.T)


def draw_train_data(train_x, train_y):
    num_futures, num_samples = np.shape(train_x)
    for i in range(0, num_samples):
        color = 'or' if train_y[i, 0] == 1 else 'ob'
        plt.plot(train_x[1, i], train_x[2, i], color, markersize=5)


def get_train_data():
    train_x = []
    train_y = []
    data_file = open("../doc/logistic_train_data.txt")
    for line in data_file.readlines():
        line_array = line.strip().split()
        train_x.append([1.0, float(line_array[0]), float(line_array[1])])
        train_y.append(int(line_array[2]))
    return np.mat(train_x).transpose(), np.mat(train_y).transpose()


def train(theta, train_x, train_y, times, alpha, precision, train_type):
    """
    Parameters
    ----------
    theta: 目标参数
    train_x:训练集 x维度
    train_y:训练集 y维度
    times:最大循环次数
    alpha:步长
    precision:精度,满足精度时停止循环
    train_type: 训练模型, base: 全量梯度下降, incremental:随机梯度/增量梯度下降法
    ----------
    """
    for c in range(0, times):
        theta_bak = theta.copy()
        if train_type == 'base':
            theta = train_mode_base(theta, train_x, train_y, alpha)
        elif train_type == 'incremental':
            theta = train_mode_incremental(theta, train_x, train_y, alpha)
        error = theta_bak - theta
        # 判断精度是否满足要求
        if (np.fabs(error) < precision).any():
            print "precision:", precision, "count:", c, "error:", error
            break
    return theta


# 全量梯度下降法
def train_mode_base(theta, train_x, train_y, alpha):
    num_futures, num_samples = np.shape(train_x)
    for j in range(0, num_futures):
        step_sum = 0
        for i in range(0, num_samples):
            train_xi = train_x[:, i]
            train_xji = train_x[j, i]
            train_yi = train_y[i, 0]
            theta_x = np.dot(theta, train_xi)[0, 0]
            sig = func_sigmoid(theta_x)
            error = train_yi - sig
            step = error * train_xji
            step_sum += step
        theta[0, j] += alpha * step_sum
    return theta


# 增量/随机梯度下降法
def train_mode_incremental(theta, train_x, train_y, alpha):
    num_futures, num_samples = np.shape(train_x)
    for i in range(0, num_samples):
        for j in range(0, num_futures):
            train_xi = train_x[:, i]
            train_xji = train_x[j, i]
            train_yi = train_y[i, 0]
            theta_x = np.dot(theta, train_xi)[0, 0]
            sig = func_sigmoid(theta_x)
            error = train_yi - sig
            step = error * train_xji
            theta[0, j] += alpha * step
    return theta


def main():
    plt.figure("mode")
    train_x, train_y = get_train_data()
    # 绘制样本数据
    draw_train_data(train_x, train_y)
    # 训练
    theta = np.mat([1.0, 1.0, 1.0])
    theta = train(theta, train_x, train_y, 1000, 0.1, 0.001, 'base')
    print theta
    draw_split_line(theta)
    plt.show()

if __name__ == '__main__':
    main()
