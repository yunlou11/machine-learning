# -- coding: utf-8 --
from sklearn import datasets as d, preprocessing


def load_dataset():
    boston = d.load_boston()
    # print boston.DESCR
    X, y = boston.data, boston.target
    X_2 = preprocessing.scale(X)
    print X_2.mean(axis=0), X_2.var(axis=0)


def make_data():
    reg_data = d.make_regression()
    complex_reg_data = d.make_regression(1000, 10, 5, 2, 1.0)   # 生成复杂点的回归数据－1000＊10的矩阵，5个有效变量，2个目标变量，1.0的偏差
    classification_set = d.make_classification(weight=[0.1])    # 生成分类数据
    blobs = d.make_blobs()  # 生成聚类数据


def main():
    load_dataset()
if __name__ == '__main__':
    main()