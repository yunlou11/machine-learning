import math
import numpy as np
import Function as func
import os


def logistic_data():
    x = np.linspace(-10, 10, 1000)
    y = func.sigmoid(x)
    return x, y


def random_between(mean, size):
    x2 = np.random.normal(mean, 10, size)
    return x2


def generate_soft_max_data():
    np.random.seed(23)
    size = 100
    y1 = np.full(size, 0)
    y2 = np.full(size, 1)
    y3 = np.full(size, 2)
    y4 = np.full(size, 3)
    x1_1 = random_between(0, size)
    x1_2 = random_between(22, size)
    x1_3 = random_between(57, size)
    x1_4 = random_between(23, size)
    x2_1 = random_between(28, size)
    x2_2 = random_between(0, size)
    x2_3 = random_between(26, size)
    x2_4 = random_between(55, size)
    x1 = np.row_stack((x1_1, x2_1))
    x2 = np.row_stack((x1_2, x2_2))
    x3 = np.row_stack((x1_3, x2_3))
    x4 = np.row_stack((x1_4, x2_4))
    xy1 = np.row_stack((x1, y1)).T
    xy2 = np.row_stack((x2, y2)).T
    xy3 = np.row_stack((x3, y3)).T
    xy4 = np.row_stack((x4, y4)).T
    return np.row_stack((xy1, xy2, xy3, xy4))


def soft_max_data():
    path = "../doc/soft_max_test_data.txt"
    if os.path.exists(path):
        print "data load"
        return np.load(path)
    else:
        data = generate_soft_max_data()
        np.savetxt(path, data)
    return data