import numpy as np


class environment():
    def __init__(self):
        X = np.arange(-3, 3, 0.023)
        Y = np.arange(-3, 3, 0.023)
        self.meshgrid = np.meshgrid(X, Y)

    def sample(self, coordinates):
        return np.sinc(coordinates[0])+np.sin(coordinates[1])


def env_sample(coordinates):
    return np.sinc(coordinates[0])+np.sin(coordinates[1])+2
