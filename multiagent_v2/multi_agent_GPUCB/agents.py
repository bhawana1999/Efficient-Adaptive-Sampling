import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class GPUCB_agent():
    def __init__(self, env, beta, n_samples, directory):
        self.directory = directory
        self.env = env
        self.meshgrid = self.env.meshgrid
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.beta = beta
        self.n_samples = n_samples

        self.visited = []
        self.sampled_depths = []
        self.kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel)  # , n_restarts_optimizer=5

        self.samples = 0
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])

    def learn(self, coordinates):
        self.samples += 1
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.visited, self.sampled_depths)
        self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)
        self.plot()
        return self.get_cor_list()

    def sample(self, coordinates):
        return self.env.sample(coordinates)

    def get_cor_list(self):
        mu = self.mu.copy()
        sigma = self.sigma.copy()

        cor = []
        for _ in range(50):
            idx = np.argmax(mu + sigma * np.sqrt(self.beta))
            cor.append(self.grid[idx])
            mu[idx] = np.NINF
            for __ in range(2000):
                mu[np.argmax(mu + sigma * np.sqrt(self.beta))] = np.NINF

        return cor

    def plot(self):
        fig = plt.figure(figsize=(10, 10))
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
                          self.env.sample(self.meshgrid), alpha=0.5, color='b')
        ax.scatter([x[0] for x in self.visited], [x[1] for x in self.visited], self.sampled_depths, c='r',
                   marker='o', alpha=1.0)
        plt.savefig(self.directory+"/"+str(self.samples)+".png")
