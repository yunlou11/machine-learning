# -- coding: utf-8 --
import os

import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt

from kaggle.digit_recognizer import data_source

base_dir = "D:\\kaggle\\digit_recognizer\\"
pca_path = base_dir + 'digit_recognizer_pca.pkl'


def pca_transform(data):
    pca = PCA(n_components=200)
    joblib.dump(pca, pca_path)
    return pca.fit_transform(data), pca


def plot_pca(pca):
    pca_ex = np.cumsum(pca.explained_variance_ratio_)
    np.savetxt(base_dir + "pca_ex.txt", pca_ex)
    plt.plot(pca_ex)
    plt.xlabel("number of component")
    plt.ylabel("cumulative explained variance")


def fit():
    data, label = data_source.load_train_data()
    pca_data, pca = pca_transform(data)
    return pca_data, label, pca


def main():
    print "pca"
    plt.figure("pca")
    plot_pca()
    plt.show()


if __name__ == '__main__':
    main()

