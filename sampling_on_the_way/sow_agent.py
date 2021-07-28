import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class SOW_agent():
    def __init__(self, env, beta, n_samples, osp, directory):
        self.directory = directory
        self.env = env
        self.meshgrid = self.env.meshgrid
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.beta = beta
        self.n_samples = n_samples
        self.osp = osp

        self.visited = []
        self.sampled_depths = []
        self.kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel) #, n_restarts_optimizer=5

        self.samples = 0
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])

    def learn(self, coordinates):
        self.samples += 1
        if (len(self.visited) > 0 and self.osp > 0):
            self.sample_on_the_way(self.visited[-1], coordinates)
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.visited, self.sampled_depths)
        self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)
        self.plot()
        return (self.grid[np.argmax(self.mu + self.sigma * np.sqrt(self.beta))], self.samples==self.n_samples)

    def get_y(self, P, Q, x):
        value = (x - P[0])*(Q[1] - P[1])/(Q[0]-P[0]) + P[1]
        return value

    def get_x(self, P, Q, y):
        value = (y - P[1])*(Q[0] - P[0])/(Q[1]-P[1]) + P[0]
        return value

    def sample_on_the_way(self, P, Q):
        if (abs(P[0]-Q[0]) >= abs(P[1]-Q[1])):
            if P[0] < Q[0]:
                i = P[0]
                while (i + 0.25/self.osp < Q[0]):
                    i += 0.25/self.osp
                    j = self.get_y(P, Q, i)
                    self.visited.append([i, j])
                    self.sampled_depths.append(self.sample([i, j]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)
            else:
                i = Q[0]
                while (i + 0.25/self.osp < P[0]):
                    i += 0.25/self.osp
                    j = self.get_y(P, Q, i)
                    self.visited.append([i, j])
                    self.sampled_depths.append(self.sample([i, j]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)

        else:
            if P[1] < Q[1]:
                i = P[1]
                while (i + 0.25/self.osp < Q[1]):
                    i += 0.25/self.osp
                    j = self.get_x(P, Q, i)
                    self.visited.append([j, i])
                    self.sampled_depths.append(self.sample([j,i]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)

            else:
                i = Q[1]
                while (i + 0.25/self.osp < P[1]):
                    i += 0.25/self.osp
                    j = self.get_x(P, Q, i)
                    self.visited.append([j, i])
                    self.sampled_depths.append(self.sample([j,i]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)

    def sample(self, coordinates):
        return self.env.sample(coordinates)

    def plot(self):
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle("Iteration %02d" %self.samples)

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
            self.mu.reshape(self.meshgrid[0].shape), alpha=0.5, color='g')
        ax.plot_wireframe(self.meshgrid[0], self.meshgrid[1],
            self.env.sample(self.meshgrid), alpha=0.5, color='b')
        ax.scatter([x[0] for x in self.visited], [x[1] for x in self.visited], self.sampled_depths, c='r',
            marker='o', alpha=1.0)
        ax1 = fig.add_subplot(1, 2, 2)
        ax1 = plt.subplot(122)
        ax1.pcolormesh(self.meshgrid[0], self.meshgrid[1],
            self.mu.reshape(self.meshgrid[0].shape), shading='auto')

        plt.savefig(self.directory+"/"+str(self.samples)+".png")