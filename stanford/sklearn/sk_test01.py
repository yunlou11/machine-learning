# --coding: utf-8 --
import heapq
import unittest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
import scipy as sp
from sklearn import datasets, svm, linear_model, neighbors, cluster, preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction import grid_to_graph
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline


class SkTestCase(unittest.TestCase):
    __name__ = "SkTestCase"

    @unittest.skip
    def test_hello(self):
        print "hello world"

    @unittest.skip
    def test_iris_svm(self):
        # todo 参照SVM GUI，下载svm_gui.py;通过鼠标左右键设置两类数据点，拟合模型并改变参数和数据。
        """
        kernel = RBF :exp(-gamma ||x-x'||^2). gamma is specified by keyword gamma, must be greater than 0.
            C : 惩罚系数, 越大,则对错误分类越敏感,表示更倾向于正确分类,容易过拟合, 越小, 则分类界面越平滑,容易欠拟合
            gamma: = 1/(2*theta^2) =>
        :return:
        """
        digits = datasets.load_digits()
        sample_size = np.shape(digits.target[:])[0]
        clf = svm.SVC(gamma=0.001, C=100)
        clf.fit(digits.data[:1700], digits.target[:1700])
        p_target = clf.predict(digits.data[1700:])
        c = 0
        for i in range(1700, sample_size):
            if p_target[i - 1700] == digits.target[i]:
                c += 1
        print float(c) / (sample_size - 1700)
        p_target_0 = clf.predict(digits.data[1701].reshape(1, -1))
        print p_target_0, digits.target[1701]

    @unittest.skip
    def test_linear_mode(self):
        diabets = datasets.load_diabetes()
        x_train = diabets.data[:-20]
        y_train = diabets.target[:-20]
        x_test = diabets.data[-20:]
        y_test = diabets.target[-20:]
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)
        print (regr.coef_)
        # the mean square error
        print np.mean((regr.predict(x_test) - y_test)**2)
        # Explained variance score: 1 is perfect prediction
        # and 0 means that there is no linear relationship
        # between X and Y.
        print regr.score(x_test, y_test)

    @unittest.skip
    def test_linear_radge(self):
        """
        岭回归, 即 L2正则化
        :return:
        """
        X = np.c_[.5, 1].T
        y = [.5, 1]
        test = np.c_[0, 2].T
        regr = linear_model.Ridge(alpha=0.1)
        plt.figure()
        np.random.seed(0)
        for _ in range(6):
            this_x = 0.1 * np.random.normal(size=(2, 1)) + X
            regr.fit(this_x, y)
            plt.plot(test, regr.predict(test))
            plt.scatter(this_x, y, s=3)
        plt.show()

    @unittest.skip
    def test_linear_lasso(self):
        diabets = datasets.load_diabetes()
        print diabets.DESCR
        x_train = diabets.data[:-20]
        y_train = diabets.target[:-20]
        x_test = diabets.data[-20:]
        y_test = diabets.target[-20:]
        alphas = np.logspace(-4, -1, 6)
        regr = linear_model.Lasso()
        scores = [regr.set_params(alpha=alpha).fit(x_train, y_train).score(x_test, y_test)
                  for alpha in alphas]
        print "scores:", scores
        best_alpha = alphas[scores.index(max(scores))]
        print "best_alpha:", best_alpha
        regr.alpha = best_alpha
        regr.fit(x_train, y_train)
        print regr.coef_

    @unittest.skip
    def test_logistic_regression(self):
        iris = datasets.load_iris()
        print np.shape(iris.data)
        x_train = iris.data[:-20]
        y_train = iris.target[:-20]
        x_test = iris.data[-20:]
        y_test = iris.target[-20:]
        logistic = linear_model.LogisticRegression(C=1)
        logistic.fit(x_train, y_train)
        print logistic.score(x_test, y_test)

    @unittest.skip
    def test_KNN(self):
        # todo 研究一下 KNN
        digits = datasets.load_digits()
        n_train_samples = int(len(digits.data) * 0.9)
        x_train = digits.data[:n_train_samples]
        y_train = digits.target[:n_train_samples]
        x_test = digits.data[n_train_samples:]
        y_test = digits.target[n_train_samples:]
        knn = neighbors.KNeighborsClassifier()
        print "knn scores:", knn.fit(x_train, y_train).score(x_test, y_test)

    @unittest.skip
    def test_cross_validation(self):
        k_fold = KFold(n_splits=3)
        digits = datasets.load_digits()
        x_digits = digits.data
        y_digits = digits.target
        svc = svm.SVC(gamma=0.001, C=100)
        for train, test in k_fold.split(x_digits):
            svc.fit(x_digits[train], y_digits[train])
            print svc.score(x_digits[test], y_digits[test])

    @unittest.skip
    def test_grid_search(self):
        digits = datasets.load_digits()
        x_digits = digits.data
        y_digits = digits.target
        svc = svm.SVC(gamma=0.001, C=100)
        svm.SVC()
        Cs = np.logspace(-6, -1, 10)
        clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),
                           n_jobs=-1)
        clf.fit(x_digits[:1000], y_digits[:1000])
        print clf.best_score_
        print clf.best_estimator_.C
        print clf.score(x_digits[1000:], y_digits[1000:])

    @unittest.skip
    def test_k_mean_face(self):
        face = misc.face(gray=True)
        x = face.reshape((-1, 1))
        k_means = cluster.KMeans(n_clusters=10, n_init=1)
        k_means.fit(x)
        cluster_centers = k_means.cluster_centers_
        values = cluster_centers.squeeze()
        labels = k_means.labels_
        face_compressed = np.choose(labels, values)
        face_compressed.shape = face.shape
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(face)
        ax = fig.add_subplot(122)
        ax.imshow(face_compressed)
        plt.show()

    @unittest.skip
    def test_connectivity_constrained(self):
        face = misc.face(gray=True)
        face = sp.misc.imresize(face, 0.10) / 255.0
        x = face.reshape((-1, 1))
        connectivity = grid_to_graph(*face.shape)
        print connectivity
        ward = AgglomerativeClustering(n_clusters=15, linkage='ward', connectivity=connectivity).fit(x)
        labels = ward.labels_
        print labels
        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.imshow(face)
        # plt.show()

    @unittest.skip
    def test_feature_agglomeration(self):
        pass

    @unittest.skip
    def test_pip(self):
        logistic = linear_model.LogisticRegression()
        pca = PCA()
        pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        ###############################################################################
        # Plot the PCA spectrum
        pca.fit(X_digits)
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        explained_variance = pca.explained_variance_
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_')
        ###############################################################################
        # Prediction
        n_components = [20, 40, 64]
        Cs = np.logspace(-4, 4, 3)
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        estimator = GridSearchCV(pipe,
                                 dict(pca__n_components=n_components,
                                      logistic__C=Cs))
        estimator.fit(X_digits, y_digits)
        plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                    linestyle=':', label='n_components chosen')
        plt.legend(prop=dict(size=12))
        plt.show()

    @unittest.skip
    def test_3d(self):
        iris = datasets.load_iris()
        x_train = iris.data
        y_train = iris.target
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = np.array(['r', 'b', 'y'])
        ax.scatter3D(x_train[:, 0], x_train[:, 1], y_train, c=colors[y_train])
        plt.show()

    @unittest.skip
    def test_scala(self):
        data = np.array([[1, 1, 2, 4, 2],
                         [1, 3, 3, 4, 4]])
        print preprocessing.scale(data, axis=1, with_mean=True, with_std=False)

    # @unittest.skip
    def test_order(self):
        a = np.array([3, 2, 1, 4])
        b = [2, 1, 3, 0]
        print np.sort(a)
        print a[b]
        print heapq.nlargest(3, range(4), a.take)


if __name__ == '__main__':
        unittest.main()

