# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()


def get_train_digit():
    return datasets.load_digits()


def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='Reds', interpolation='nearest',
                  clim=(0, 16))


def plot_pca_digits(pca_x_data, y):
    plt.scatter(pca_x_data[:, 0], pca_x_data[:, 1],
                c=y, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()


def plot_explained_variance_ratio(x_data):
    pca = PCA()
    pca.fit(x_data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of component")
    plt.ylabel("cumulative explained variance")


def pca_fit_transform(x_data):
    pca = PCA(2)
    return pca.fit_transform(x_data), pca


def main():
    digits = get_train_digit()
    dd = digits.data
    plot_digits(digits.data)
    pca_x_data, pca = pca_fit_transform(digits.data)
    print pca.n_components_
    print pca_x_data[0], "-===="
    print (pca.inverse_transform(pca_x_data))[0]
    plt.figure("pca")
    plot_pca_digits(pca_x_data, digits.target)
    plt.figure("explained variance ratio")
    plot_explained_variance_ratio(digits.data)
    plt.show()
if __name__ == '__main__':
    main()