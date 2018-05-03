# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import os


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


def load_data():
    """
    加载训练数据,保存格式为npy二进制格式,保存了数组的格式信息
    :return: 训练数据数组
    """
    path = "../doc/soft_max_data.npy"
    return np.load(path)


def load_test_data():
    """
    加载验证数据集,保存格式为txt,用于保存二维的数组,能够打开文件直接查看内容,以空格和\n分隔
    :return: 训练数据集数组
    """
    path = "../doc/soft_max_test_data.txt"
    return np.loadtxt(path, dtype=float)


def draw_train_data(data):
    """
    绘制数据集的分布,不同类别对应不同颜色,如 0 类别对应 'or',红色圆点
    :param data: 需要绘制的数据集
    :return:
    """
    sample_size, figure_size = np.shape(data)
    colors = ['or', 'ob', 'og', 'oy']
    for i in range(sample_size):
        plt.plot(data[i, 0], data[i, 1], colors[int(data[i, 2])])


def decision_boundaries(theta, j_0, j_1):
    x1 = np.linspace(0, 25, 10)
    theta0 = theta[j_0, 0] - theta[j_1, 0]
    theta1 = theta[j_0, 1] - theta[j_1, 1]
    theta2 = theta[j_0, 2] - theta[j_1, 2]
    x2 = -(theta0 + theta1 * x1) / theta2
    return x1, x2


def draw_decision_boundaries(theta):
    color = ['r', 'b', 'g', 'y']
    category_size = 4
    for j in range(4):
        j_0 = j
        j_1 = (j + 1) % category_size
        x1, x2 = decision_boundaries(theta, j_0, j_1)
        plt.plot(x1, x2, color[j])


def normalization_sum(theta, xi, mini_rate=1.0):
    """
    softmax函数的分母,用于归一化的所有分子之和
    :param theta: 所求的参数
    :param xi: 某一条样本数据
    :param mini_rate: 由于进行e的指数运算时,有可能数值溢出,分子分母同时以mini_rate比例进行缩减,最终的softmax结果不变
    :return: 某一条样本数据对应的softmax函数的分母,用于归一化
    """
    category_size, future_size = np.shape(theta)
    n_sum = 0
    for l in range(category_size):
        n_sum += mini_rate * np.exp(np.dot(theta[l], xi))
    return n_sum


def soft_max(theta, j, xi):
    """
    进行softmax函数的运算, xi属于j对应的类别的概率
    :param theta:
    :param j: 类别
    :param xi: 某条样本的特征变量向量
    :return: xi属于j类型的概率
    """
    mini_rate = 0.5
    element = mini_rate * np.exp(np.dot(theta[j], xi))
    n_sum = normalization_sum(theta, xi, mini_rate)
    element_percent = element / n_sum
    return element_percent


def soft_max_hypothesis(theta, xi):
    """
    soft_max的假设函数 H_theta(x) = XXX , 根据theta,判断xi样本属于哪一个类别的概率最大,返回概率最大的类别
    :param theta: 参数
    :param xi: 某条样本
    :return: xi所属的类别
    """
    category_size = np.shape(theta)[0]
    max_j = -1
    max_p = 0
    for j in range(category_size):
        probability = soft_max(theta, j, xi)
        if probability > max_p:
            max_j = j
            max_p = probability
    return max_j


def test_validate(theta, x, y, sample_size):
    """
    验证模型的准确度
    :param theta: 训练出来的参数
    :param x: 测试数据集x -> 代表特征
    :param y: 测试数据集y -> 代表类别
    :param sample_size: 测试数据集大小
    :return: 模型的分类准确度, 正确分类样本数/总样本数
    """
    correct_count = 0.0
    for i in range(sample_size):
        xi = x[:, i]
        predict = soft_max_hypothesis(theta, xi)
        if predict == y[i]:
            correct_count += 1
    return correct_count / sample_size


def train_mode_l2(x, y, category, theta, lmbda, alpha):
    """
    进行一次训练模型的过程, 使用 L2正则化方法防止"过拟合",使用全量梯度下降法,每次更新theta_j,循环所有样本
    注意(1) 虽然很多时候统一叫梯度下降法, 但是当进行极大似然估计时, 其实是逆向使用,应该叫梯度上升法
       (2) 截距项或偏置项,即 theta_0 不应该使用L2正则化优化, 会造成 theta_0更新过慢,单独对theta_0使用原本的更新方式
       (3) lmbda到底要不要除以样本数,有些会将lmbda/sample_size, 这里没有使用这种方式, 因为当样本数变大时,同时需要增大lmbda,才能保证
           L2正则优化的效果
       (4) 增加样本数据量能够防止"过拟合",但是样本数量很难获得,所以使用L2正则优化
    :param x: 训练数据集x
    :param y: 训练数据集y
    :param category: 类别集合
    :param theta: 所求的参数矩阵
    :param lmbda: L2正则化优化系数
    :param alpha: 每次跌倒步长系数
    :return: 一次训练后的theta矩阵
    """
    future_size, sample_size = np.shape(x)
    for j in category:
        x_sum = np.zeros(future_size)
        for i in range(sample_size):
            xi = x[:, i]
            yi = y[i]
            error = indicator(yi, j) - soft_max(theta, j, xi)
            x_sum += error * xi
        step = x_sum / sample_size
        b = theta[j, 0]
        theta[j] = (1 - alpha * lmbda) * theta[j] + alpha * step
        b = b + alpha * step[0]
        theta[j, 0] = b
    return theta


def train(theta, x, y, category, lmbda, alpha, max_iters):
    """
    按照指定迭代次数训练模型, 将训练出来的theta矩阵保存文件中,如果文件存在,直接加载theta,方便其他的调试,不用每次都重新训练theta
    :param theta: 所求参数矩阵
    :param x: 训练数据集x
    :param y: 训练数据集y
    :param category: 类别集合
    :param lmbda: L2正则优化系数
    :param alpha: 步长系数
    :param max_iters: 迭代次数
    :return: 训练后的 theta
    """
    theta_save_path = '../doc/soft_max_theta.txt'
    if os.path.exists(theta_save_path):
        theta = np.loadtxt(theta_save_path, dtype=float)
        print 'theta has saved in ', theta_save_path, 'and will load it'
    else:
        for i in range(max_iters):
            theta = train_mode_l2(x, y, category, theta, lmbda, alpha)
        np.savetxt(theta_save_path, theta)
    return theta


def xy_data(raw_data):
    """
    从原始数据中分离出x数据集和y数据集
    :param raw_data:  原始数据
    :return: x和y数据集
    """
    sample_size = np.shape(raw_data)[0]
    intercept = np.full((sample_size, 1), 20)
    train_data = np.column_stack((intercept, raw_data))
    y = train_data[:, 3]
    x = train_data[:, 0:3].T
    return x, y


def main():
    theta = np.array([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]])
    raw_data = load_data()
    test_data = load_test_data()
    sample_size = np.shape(raw_data)[0]
    x, y = xy_data(raw_data)
    x_test, y_test = xy_data(test_data)
    category = [0, 1, 2, 3]
    plt.figure("mode")
    plt.subplot(121)
    draw_train_data(raw_data)
    plt.subplot(122)
    draw_train_data(test_data)
    theta = train(theta, x, y, category, 1e3, 1e-4, 1000)
    # draw_decision_boundaries(theta)
    print theta
    print test_validate(theta, x_test, y_test, sample_size)
    plt.show()

if __name__ == '__main__':
    main()