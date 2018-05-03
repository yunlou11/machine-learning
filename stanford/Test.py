import numpy
from sklearn.decomposition import PCA

import Function as Func
import DataSource as Ds
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

from scipy import linalg
from scipy import linalg
import numpy as np
from sklearn.decomposition import PCA


def test_dot():
    a = numpy.mat([49, 70, 90])
    b = numpy.mat([1.0, -0.017612, 14.053064]).transpose()
    ab = numpy.dot(a, b)
    print ab


def test_numpy_compare():
    try:
        if ((numpy.mat([[0.11102094, 0.05340839, 0.5746013]])) < 1).any():
            print("success")
        else:
            print("failed")
    except Exception, e:
        print e.message


def test_softmax_data():
    data = Ds.soft_max_data()


def test_get_matrix():
    y = np.array([[1], [2]])
    theta = np.array([[1, 2, 3],
                     [4, 5, 6]])
    print np.shape(y)
    print y[0, 0]
    print theta[0, :]


def load_theta():
    theta = np.loadtxt("../doc/soft_max_theta.txt", dtype=float)
    print theta
    text_data = np.loadtxt("../doc/soft_max_test_data.txt", dtype=float)
    print text_data


def save_soft_max_data():
    data = np.load("../doc/soft_max_data.npy")
    np.savetxt("../doc/soft_max_data.txt", data, fmt='%.2f %.2f %d')


def np_array_add():
    p_x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print np.shape(p_x0)
    print p_x0[0]


def array_multiply():
    a = np.array([1, 2, 3, 4])
    b = np.array([-1, 1, -1, 1])
    data = np.loadtxt("../doc/hard_svm_data.txt", dtype=float)
    alpha = np.ones(3)
    b = 0
    x = data[:, 0:2]
    y = data[:, 2]
    kernel = np.dot(x, x[0])
    print sum(alpha * y * kernel) + b
    print y[0] * np.dot(x[0], x[0]) + y[1] * np.dot(x[1], x[0]) + y[2] * np.dot(x[2], x[0])


def false():
    a = True
    b = [1, 2, 3]
    b = np.array(b) - 1
    print b
    if not a:
        print "false"
    if a:
        print "true"


def float_int():
    for i in range(100):
        s = 1.0
        print s == 1


def abs():
    a = np.array([1, 2, 3])
    b = np.array([1, 4, 5])
    print np.sum(np.abs(a - b))


def devide():
    a = np.array([[4, 4],
                 [2, 2]])
    b = np.array([[4],
                 [2]])
    c = []
    d = np.float64(2.0)
    c.append(2.1)
    c.append(4.5)
    c.append(6.0)
    print c / d


def pow():
    print 2 ** 3
    print np.sqrt(4)


def norm():
    x = np.array([1, 1])
    mu = np.array([1, 1])
    sigma = np.array([[1, 0],
                      [0, 1]])
    future_size = np.shape(x)[0]
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    exp_index = -1.0 / 2 * np.dot(np.dot((x - mu).T, sigma_inv), x - mu)
    print 1.0 / np.sqrt(((2 * np.pi) ** future_size) * sigma_det) * np.exp(exp_index)


def array_add():
    a=[]
    a.append(1)
    a.append(2)
    print a


def norm():
    print sts.norm.pdf(100, 0, 1)


def ss():
    a = np.array([2, 4])
    b = np.array([4, 6])
    e = np.mean(b)
    print e
    print np.dot((a-e), (a-e).T)
    print np.cov(a , b)


def test_image():
    plt.figure("test")
    k_range = (0, 1, 2, 3)
    k_list = [5, 4, 8.222, 9]
    plt.plot(k_range, k_list)
    ax = plt.gca()
    for k in zip(k_range, k_list):
        ax.annotate("(%s, %.2f)" % k, xy=k)
    plt.show()


def test_dot():
    X = np.array([[1., 2., 3.],
                  [2., 4., 7.],
                  [1., 3., 1.],
                  [4., 8., 1.]], dtype=np.float64)
    a = np.array([0.5, 0.5, 0.5, 0.5])
    print "aX:\n",np.dot(a, X)


if __name__ == '__main__':
    # test_dot()
    # test_plt()
    # test_numpy_compare()
    #pow()
    test_dot()