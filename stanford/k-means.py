# --coding: utf-8 --
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_train_data():
    path = '../doc/soft_max_data.txt'
    return np.loadtxt(path, dtype=float)[:, 0:2]


def draw_train_data(data):
    plt.scatter(data[:, 0], data[:, 1])


def draw_label(data, labels, centers):
    colors = ["r", "g", "y", "c"]
    center_size, future_size = np.shape(centers)
    sample_size = np.shape(data)[0]
    for i in range(sample_size):
        plt.scatter(data[i, 0], data[i, 1], marker="o", color=colors[labels[i]])
    for i in range(center_size):
        plt.scatter(centers[i, 0], centers[i, 1], marker="x", color="k")


def classify_label(centers, xi):
    max_distance = sys.maxint
    label = -1
    centers_size, future_size = np.shape(centers)
    for j in range(centers_size):
        tmp_distance = np.sum(np.abs(xi - centers[j])) / future_size
        if tmp_distance < max_distance:
            max_distance = tmp_distance
            label = j
    return label


def get_labels(data, centers):
    labels = []
    sample_size = np.shape(data)[0]
    for i in range(sample_size):
        labels.append(classify_label(centers, data[i]))
    return labels


def get_centers(data, labels):
    total_count = np.zeros((4, 1))
    sample_size = np.shape(data)[0]
    centers = np.zeros((4, 2))
    for i in range(sample_size):
        label = labels[i]
        centers[label] += data[i]
        total_count[label] += 1
    return centers / total_count


def train(train_data, centers, iter_max, precision):
    labels = []
    for i in range(iter_max):
        labels = get_labels(train_data, centers)
        tmp_centers = get_centers(train_data, labels)
        changes = abs(np.sum(centers - tmp_centers))
        centers = tmp_centers
        if changes <= precision:
            print "precision satisfied:", i
            break
    return centers, labels


def main():
    plt.figure("mode")
    train_data = load_train_data()
    plt.subplot(121)
    draw_train_data(train_data)
    centers = np.array([[1, 2],
                        [50, 1],
                        [50, 50],
                        [1, 50]])
    plt.subplot(122)
    centers, labels = train(train_data, centers, 100, 1e-4)
    draw_label(train_data, labels, centers)
    plt.show()
if __name__ == '__main__':
    main()