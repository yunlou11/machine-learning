from scipy import linalg
import numpy as np
from sklearn.decomposition import PCA


class MySvd:
    def __init__(self):
        self.mean_ = None
        pass

    def svd(self, X):
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X)
        return U, S, V


class MyPP(PCA):
    def fit_full(self, X, n_c):
        return self._fit_full(X, n_c)


def main():
    pca = MyPP(3)
    my_svd = MySvd()
    X = np.array([[1., 2., 3.],
                  [2., 4., 7.],
                  [1., 3., 1.],
                  [4., 8., 1.]], dtype=np.float64)
    print np.ndim(X)
    x_mean = np.mean(X, axis=0)
    X_mean = X - x_mean
    A = np.dot(X_mean.T, X_mean)
    A_T = np.dot(X_mean, X_mean.T)
    print "A", A_T
    print "A.ndim", np.ndim(A)
    eigen_value, eigen_vector = np.linalg.eig(A)
    print linalg.eig(A)
    U, S, Vh = my_svd.svd(X)
    print "U:\n",U
    print "S:\n", S
    print "V:\n", Vh
    U_2, S_2, V_2h = pca.fit_full(X, 2)
    print "U2:\n",U_2
    print "S2:\n", S_2
    print "V2:\n", V_2h
    print "eigen_vector:\n", eigen_vector
    print "eigen_value:\n", eigen_value
    print "s**2:\n", S**2
    print "A eigen_vector:\n", np.dot(A, eigen_vector)
    print "Av:\n", np.dot(A, V_2h.T)
    print U[:, 3]
    print "UT X:\n", np.dot(X.T, U)
    print "S^2 V:\n", S**2 * V_2h

if __name__ == '__main__':
    main()


