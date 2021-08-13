import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class AR_agent():
    def __init__(self, mesh, env_sample, beta, n_samples, init_radius, directory,
                 threshold_gain, threshold_loss, deltaParam):
        self.directory = directory
        self.env = env_sample
        self.meshgrid = mesh

        self.rad = init_radius
        self.beta = beta
        self.n_samples = n_samples
        self.thresh_gain = threshold_gain
        self.thresh_loss = threshold_loss
        self.deltaParam = deltaParam

        self.visited = []
        self.sampled_depths = []

        self.kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
        # , n_restarts_optimizer=5
        self.samples = 0

    def find_idx(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        idx = [0, 0]

        for i in range(self.meshgrid.meshgrid[0].shape[1]):
            if x == self.meshgrid.meshgrid[0][0][i]:
                idx[1] = i
                break

        for j in range(self.meshgrid.meshgrid[1].shape[0]):
            if y == self.meshgrid.meshgrid[1][j][0]:
                idx[0] = j
                break
        return idx

    def get_next_coordinates(self):
        idx = self.find_idx(self.curr_coor)
        max_h = np.NINF

        for i in range(idx[0]-self.rad, idx[0]+self.rad+1):
            for j in range(idx[1]-self.rad, idx[1]+self.rad+1):
                if (0 <= i < self.meshgrid.meshgrid[0].shape[0] and 0 <= j < self.meshgrid.meshgrid[0].shape[1]):
                    if (self.meshgrid.meshgrid[2][i][j] + (self.meshgrid.meshgrid[3][i][j]*np.sqrt(self.beta)) > max_h):
                        max_h = self.meshgrid.meshgrid[2][i][j] + \
                            (self.meshgrid.meshgrid[3][i][j]*np.sqrt(self.beta))
                        new_idx = [i, j]

        return [self.meshgrid.meshgrid[0][new_idx[0]][new_idx[1]], self.meshgrid.meshgrid[1][new_idx[0]][new_idx[1]]]

    def update_radius(self, pred, truth):
        if (abs(truth - pred) > self.thresh_gain):
            self.rad -= self.deltaParam
        elif (abs(truth - pred) < self.thresh_loss):
            self.rad += self.deltaParam

    def learn(self, coordinates):
        self.samples += 1
        self.curr_coor = coordinates
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))

        idx = self.find_idx(coordinates)
        self.predicted = self.meshgrid.meshgrid[2][idx[0]][idx[1]]

        self.meshgrid.visited.append(coordinates)
        self.meshgrid.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.meshgrid.visited, self.meshgrid.sampled_depths)
        self.meshgrid.mu, self.meshgrid.sigma = self.gpr.predict(self.meshgrid.grid, return_std=True)
        self.meshgrid.update_mesh()

        self.ground_truth = self.sample(coordinates)
        return self.predicted, self.ground_truth

    def sample(self, coordinates):
        return self.env(coordinates)
