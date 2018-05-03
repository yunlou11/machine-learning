# -- coding: utf-8 --
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


def grads(o, xdata, y, j):
    dsum = 0
    for i in (0, 49):
        x_i = xdata[:, i]
        dsum += ((np.dot(o, x_i) - y[i]) * x_i[j])
    return dsum


if __name__ == "__main__":
    random.seed(42)
    o = np.array([[1.0, 2.0]])
    x = np.linspace(0, 100, 50)
    x_data = np.row_stack((np.ones((1, 50)), x))
    y1 = np.dot(o, x_data)
    y = 5 * x + 80
    y_data = y + 50 * np.random.normal(scale=1, size=50)
    plt.figure("mode")
    plt.plot(x, y, label='y')
    plt.scatter(x, y_data, label='y2')
    plt.plot(x, y1.T, label='y1')
    for c in range(0, 100):
        o[0, 0] = (o[0, 0] - 0.1 * grads(o, x_data, y_data, 0))
        print o[0, 0]
        tmp = (o[0, 1] - 0.00001 * grads(o, x_data, y_data, 1))
        o[0, 1] = tmp
    print o
    plt.plot(x, np.transpose(np.dot(o, x_data)), label=c)
    plt.show()
