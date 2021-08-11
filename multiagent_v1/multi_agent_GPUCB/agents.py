import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class GPUCB_agent():
    def __init__(self, mesh, env_sample, beta, n_samples, directory):
        self.directory = directory
        self.env = env_sample
        self.meshgrid = mesh

        self.beta = beta
        self.n_samples = n_samples

        self.visited = []
        self.sampled_depths = []
        self.kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel)  # , n_restarts_optimizer=5

        self.samples = 0

    def learn(self, coordinates):
        self.samples += 1
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.visited, self.sampled_depths)
        self.meshgrid.mu, self.meshgrid.sigma = self.gpr.predict(
            self.meshgrid.grid, return_std=True)

    def sample(self, coordinates):
        return self.env(coordinates)
