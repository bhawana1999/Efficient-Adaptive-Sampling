import numpy as np


class environment():
    def __init__(self):
        X, Y = np.round(np.arange(-3, 3, 0.025, dtype=np.float), 3), \
            np.round(np.arange(-3, 3, 0.025, dtype=np.float), 3)
        self.meshgrid = np.meshgrid(X, Y)

    def sample(self, coordinates):
        return np.sin(coordinates[0])+np.cos(coordinates[1])


def env_sample(coordinates):
    return np.sinc(coordinates[0])+np.sin(coordinates[1])+2
