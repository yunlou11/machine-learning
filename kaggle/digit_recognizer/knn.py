# -- coding: utf-8 --
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

from kaggle.digit_recognizer import data_source
from kaggle.digit_recognizer import pca
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

predict_file = data_source.base_dir + "result.txt"


def train(train_data, label, n_neighbors=15, weight="uniform"):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(train_data, label)
    return clf


def predict(clf, pca_mode):
    pca_test_data = pca_mode.transform(data_source.load_test_data())
    result = clf.predict(pca_test_data).reshape((-1, 1))
    sample_size = np.shape(result)[0]
    image_ids = np.arange(1, sample_size + 1, 1, dtype=int).reshape((-1, 1))
    df = pd.DataFrame(np.hstack((image_ids, result)), columns=list(("ImageId", "Label")), dtype=int)
    df.to_csv(data_source.base_dir + "result.csv", index=False)


def k_scores(train_data, label):
    k_range = range(1, 10)
    k_scores_list = []
    for k in k_range:
        print "k", k
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train_data, label, cv=5, scoring="accuracy")
        k_scores_list.append(scores.mean())
    return k_range, k_scores_list


def plot_k_scores(train_data, label):
    k_range, k_scores_list = k_scores(train_data, label)
    print k_scores_list
    plt.figure("k scores")
    plt.plot(k_range, k_scores_list)
    ax = plt.gca()
    for k_tuple in zip(k_range, k_scores_list):
        ax.annotate("(%s, %.4f)" % k_tuple, xy=k_tuple)
    plt.xlabel("value of k for KNN")
    plt.ylabel("Cross-Validated Accuracy")
    plt.savefig(data_source.base_dir + "k_scores_accuracy.png")
    plt.show()


def main():
    print "knn"
    train_data, label, pca_mode = pca.fit()
    plot_k_scores(train_data, label)
    # clf = train(train_data, label, n_neighbors=4, weight="distance")
    # predict(clf, pca_mode)


if __name__ == '__main__':
    main()