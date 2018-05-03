# -- coding: utf-8 --
import numpy as np


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


def x_value(raw_x):
    """
    --------------------------------------------------------
    x1,x2 |x<-20|-20<=x<-10|-10<=x<0|0<=x<10|10<=x<=20|20<x
    --------------------------------------------------------
    value |  0  |     1    |    2   |    3   |    4    |  5
    --------------------------------------------------------
    :param raw_x:
    :return:
    """
    value = 0
    if -20 <= raw_x < -10:
        value = 1
    elif -10 <= raw_x < 0:
        value = 2
    elif 0 <= raw_x < 10:
        value = 3
    elif 10 <= raw_x < 20:
        value = 4
    elif 20 < raw_x:
        value = 5
    return value


def maximum_likelihood_estimation(data):
    sample_size = np.shape(data)[0]
    p_x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_x1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    k = np.shape(p_x0)[0]
    for i in range(sample_size):
        x0 = x_value(data[i, 0])
        p_x0[x0] += 1
        x1 = x_value(data[i, 1])
        p_x1[x1] += 1
    p_x0 = (p_x0 + 1) / (sample_size + k)
    p_x1 = (p_x1 + 1) / (sample_size + k)
    return p_x0, p_x1


def train(train_class0, train_class1):
    """

    :param train_class0:
    :param train_class1:
    :return:
    """
    class0_size = float(np.shape(train_class0)[0])
    class1_size = float(np.shape(train_class1)[0])
    py_1 = class1_size / (class0_size + class1_size)
    px0_y0, px1_y0 = maximum_likelihood_estimation(train_class0)
    px0_y1, px1_y1 = maximum_likelihood_estimation(train_class1)
    return py_1, px0_y0, px1_y0, px0_y1, px1_y1


def predict(x, py_1, px0_y0, px1_y0, px0_y1, px1_y1):
    category = 0
    py_0 = 1 - py_1
    x0 = x_value(x[0])
    x1 = x_value(x[1])
    probability_0 = px0_y0[x0] * px1_y0[x1] * py_0
    probability_1 = px0_y1[x0] * px1_y1[x1] * py_1
    if probability_1 > probability_0:
        category = 1
    return category


def validate(data, py_1, px0_y0, px1_y0, px0_y1, px1_y1):
    sample_size = np.shape(data)[0]
    correct_count = 0.0
    for i in range(sample_size):
        category = predict(data[i], py_1, px0_y0, px1_y0, px0_y1, px1_y1)
        if category == data[i, 2]:
            correct_count += 1
    return correct_count / sample_size


def main():
    print "zou qi"
    train_class0, train_class1 = load_data("../doc/soft_max_data.txt")
    validate_date = np.row_stack(load_data("../doc/soft_max_test_data.txt"))
    py_1, px0_y0, px1_y0, px0_y1, px1_y1 = train(train_class0, train_class1)
    print py_1, px0_y0, px1_y0, px0_y1, px1_y1
    print validate(validate_date, py_1, px0_y0, px1_y0, px0_y1, px1_y1)

if __name__ == '__main__':
    main()