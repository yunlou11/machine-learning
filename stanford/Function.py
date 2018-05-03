import numpy


def sigmoids(x):
    sig = 1.0 / (1 + numpy.exp(-x))
    return sig