# --coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts


def load_train_data():
    path = '../doc/soft_max_data.txt'
    return np.loadtxt(path, dtype=float)[:, 0:2]


def draw_train_data(data):
    plt.scatter(data[:, 0], data[:, 1])


def draw_label(data, labels, centers):
    colors = ["r", "g", "y", "c"]
    center_size, future_size = np.shape(centers)
    sample_size = np.shape(data)[0]
    for i in range(sample_size):
        plt.scatter(data[i, 0], data[i, 1], marker="o", color=colors[labels[i]])
    for i in range(center_size):
        plt.scatter(centers[i, 0], centers[i, 1], marker="x", color="k")


def norm_tmp(x, mu, sigma):
    future_size = np.shape(x)[0]
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    exp_index = -1.0 / 2 * np.dot(np.dot((x - mu).T, sigma_inv), x - mu)
    return 1.0 / np.sqrt(((2 * np.pi) ** future_size) * sigma_det) * np.exp(exp_index)


def norm_pdf(x, mu, sigma):
    return sts.multivariate_normal.pdf(x, mu, sigma)


def get_Q_zi(xi, phi, mu, sigma):
    """
    若mu和sigma设置不合适,可能会出现property_i=[0, 0, 0, 0]
    某一点的概率太小,计算机已经无法表示,趋近于0
    :param xi:
    :param phi:
    :param mu:
    :param sigma:
    :return:
    """
    class_size = np.shape(phi)[0]
    Q_zi = []
    for i in range(class_size):
        property_i = norm_pdf(xi, mu[i], sigma[i]) * phi[i]
        Q_zi.append(property_i)
    sum_Q_zi = np.sum(Q_zi)
    return Q_zi / sum_Q_zi


def e_step(data, phi, mu, sigma):
    sample_size = np.shape(data)[0]
    Q_z = []
    for i in range(sample_size):
        Q_zi = get_Q_zi(data[i], phi, mu, sigma)
        Q_z.append(Q_zi)
    return np.array(Q_z)


def m_step(Q_z, data):
    sample_size, future_size = np.shape(data)
    class_size = np.shape(Q_z)[1]
    phi = []
    mu = []
    sigma = []
    for j in range(class_size):
        wj = Q_z[:, j].reshape(sample_size, 1)
        sum_wj = np.sum(wj)
        phi.append(np.sum(wj) / sample_size)
        mu.append(np.sum(data * wj, axis=0) / sum_wj)
        sigma_j = np.zeros((2, 2))
        for i in range(sample_size):
            # 如果不reshape, error.T的shape依然为(4,), 矩阵乘法的结果为一个浮点数
            error = (data[i] - mu[j]).reshape(1, future_size)
            cov_i = error.T * error
            sigma_j = sigma_j + wj[i] * cov_i
        sigma.append(sigma_j / sum_wj)
    return np.array(phi), np.array(mu), np.array(sigma)


def get_array_max_index(array):
    length = np.shape(array)[0]
    max_value = array[0]
    max_index = 0
    for i in range(1, length):
        if array[i] > max_value:
            max_value = array[i]
            max_index = i
    return max_index


def get_labels(Q_z):
    sample_size = np.shape(Q_z)[0]
    labels = []
    for i in range(sample_size):
        labels.append(get_array_max_index(Q_z[i]))
    return labels


def train(data, phi, mu, sigma, iter_max, precision):
    Q_z_old = []
    Q_z = []
    for i in range(iter_max):
        Q_z = e_step(data, phi, mu, sigma)
        phi, mu, sigma = m_step(Q_z, data)
        if i > 0 and np.sum(np.abs(Q_z - Q_z_old)) <= precision:
            print "precision satisfied:", i
            break
        Q_z_old = Q_z
    return phi, mu, sigma, Q_z


def main():
    plt.figure("mode")
    train_data = load_train_data()
    phi = np.array([0.25, 0.25, 0.25, 0.25])
    # 注意,对初始mu和sigma设置有要求,如果设置参数太离谱,某些点可能对于所有分模型的概率都为0,因为偏离mu太远的点,计算机无法继续表示精度,只能表示为0
    mu = np.array([[0, 20], [25, 0], [50, 25], [25, 50]])
    sigma = np.array([[[1.0, 0.0], [0.0, 1.0]],
                      [[1.0, 0.0], [0.0, 1.0]],
                      [[1.0, 0.0], [0.0, 1.0]],
                      [[1.0, 0.0], [0.0, 1.0]]])
    draw_train_data(train_data)
    phi, mu, sigma, Q_z = train(train_data, phi, mu, sigma, 500, 1e-2)
    print phi, mu, sigma
    labels = get_labels(Q_z)
    draw_label(train_data, labels, mu)
    plt.show()
if __name__ == '__main__':
    main()