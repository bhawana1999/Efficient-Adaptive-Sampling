import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class AR_agent():
    def __init__(self, env, beta, n_samples, init_radius, directory, threshold_gain, threshold_loss, deltaParam):
        self.directory = directory
        self.rad = init_radius
        self.env = env
        self.meshgrid = self.env.meshgrid
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.beta = beta
        self.n_samples = n_samples
        self.thresh_gain = threshold_gain
        self.thresh_loss = threshold_loss
        self.deltaParam = deltaParam
        self.visited = []
        self.sampled_depths = []
        self.kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel) #, n_restarts_optimizer=5

        self.samples = 0
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])

    def get_next_coordinates(self):
        # min_mu = min(self.mu)
        # min_sigma = min(self.sigma)
        # optimise the code here
        max_h = np.NINF
        for i in range(len(self.mu)):
            if (np.hypot((self.curr_coor[0]-self.grid[i][0]), (self.curr_coor[1]-self.grid[i][1])) < self.rad):
                if (self.mu[i] + self.sigma[i] * np.sqrt(self.beta) > max_h):
                    print(max_h)
                    max_h = self.mu[i] + self.sigma[i] * np.sqrt(self.beta)
                    idx =  i

        predicted = self.mu[idx]
        ground_truth = self.sample(self.grid[idx])
        if (abs(ground_truth - predicted) > self.thresh_gain):
            self.rad -= self.deltaParam
        elif (abs(ground_truth - predicted) < self.thresh_loss):
            self.rad += self.deltaParam

        return self.grid[idx]

    def learn(self, coordinates):
        self.samples += 1
        self.curr_coor = coordinates
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.visited, self.sampled_depths)
        self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)
        self.plot()
        return (self.get_next_coordinates(), self.samples==self.n_samples)

    
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