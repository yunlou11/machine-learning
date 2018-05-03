# -- coding: utf-8 --
import pandas as pd
import numpy as np


base_dir = "D:\\kaggle\\digit_recognizer\\"
train_data_percent = 0.3


def load_train_data():
    data = pd.read_csv(base_dir + "train.csv")
    sample_size = np.shape(data.values)[0]
    data = data.values[0: int(sample_size * train_data_percent), :]
    print np.shape(data)[0]
    return data[:, 1:], data[:, 0]


def load_test_data():
    data = pd.read_csv(base_dir + "test.csv")
    return data.values

