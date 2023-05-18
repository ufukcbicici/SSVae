import numpy as np


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def produce_2d_circular_gaussian_data(sample_count):
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]  # diagonal unit covariance
        samples = np.random.multivariate_normal(mean, cov, sample_count)
        A_ = samples / 10.0
        B_ = samples / np.linalg.norm(samples, axis=1)[:, np.newaxis]
        C_ = A_ + B_
        return C_


