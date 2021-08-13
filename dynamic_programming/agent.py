import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


class cell():
    def __init__(self, x, y, mu, sigma):
        self.x = x
        self.y = y
        self.mu = mu
        self.sigma = sigma
        self.coor = [self.x, self.y]


class DP_agent:
    def __init__(self, env, beta, n_samples, directory, osp, init_radius):
        self.directory = directory
        self.env = env
        self.n_samples = n_samples
        self.meshgrid = self.env.meshgrid.copy()
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])
        self.kernel = ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e4))
        self.gpr = GaussianProcessRegressor(kernel=self.kernel)
        # , n_restarts_optimizer=5

        self.rad = init_radius
        self.beta = beta
        self.osp = osp
        self.visited = []
        self.sampled_depths = []
        self.samples = 0

        self.update_mesh()

    def get_y(self, P, Q, x):
        value = (x - P[0])*(Q[1] - P[1])/(Q[0]-P[0]) + P[1]
        return value

    def get_x(self, P, Q, y):
        value = (y - P[1])*(Q[0] - P[0])/(Q[1]-P[1]) + P[0]
        return value

    def sample(self, coordinates):
        return self.env.sample(coordinates)

    def find_idx(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        idx = [0, 0]

        for i in range(self.meshgrid[0].shape[1]):
            if x == self.meshgrid[0][0][i]:
                idx[1] = i
                break

        for j in range(self.meshgrid[1].shape[0]):
            if y == self.meshgrid[1][j][0]:
                idx[0] = j
                break
        return idx

    def update_mesh(self):
        self.mu = self.mu.reshape(self.meshgrid[0].shape)
        self.sigma = self.sigma.reshape(self.meshgrid[0].shape)
        if self.samples == 0:
            self.meshgrid.append(self.mu)
            self.meshgrid.append(self.sigma)
        else:
            self.meshgrid[2] = self.mu
            self.meshgrid[3] = self.sigma

    def get_next_coordinates(self):
        idx = self.find_idx(self.curr_coor)
        max_var = np.NINF

        for i in range(idx[0]-self.rad, idx[0]+self.rad+1):
            for j in range(idx[1]-self.rad, idx[1]+self.rad+1):
                if (0 <= i < self.meshgrid[0].shape[0] and 0 <= j < self.meshgrid[0].shape[1]):
                    if(self.variance_of_the_path(self.curr_coor, [i, j]) > max_var):
                        max_var = self.variance_of_the_path(
                            self.curr_coor, [i, j])
                        new_idx = [i, j]

        return [self.meshgrid[0][new_idx[0]][new_idx[1]], self.meshgrid[1][new_idx[0]][new_idx[1]]]

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
                    self.mu, self.sigma = self.gpr.predict(
                        self.grid, return_std=True)
                    self.update_mesh()
            else:
                i = Q[0]
                while (i + 0.25/self.osp < P[0]):
                    i += 0.25/self.osp
                    j = self.get_y(P, Q, i)
                    self.visited.append([i, j])
                    self.sampled_depths.append(self.sample([i, j]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(
                        self.grid, return_std=True)
                    self.update_mesh()

        else:
            if P[1] < Q[1]:
                i = P[1]
                while (i + 0.25/self.osp < Q[1]):
                    i += 0.25/self.osp
                    j = self.get_x(P, Q, i)
                    self.visited.append([j, i])
                    self.sampled_depths.append(self.sample([j, i]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(
                        self.grid, return_std=True)
                    self.update_mesh()

            else:
                i = Q[1]
                while (i + 0.25/self.osp < P[1]):
                    i += 0.25/self.osp
                    j = self.get_x(P, Q, i)
                    self.visited.append([j, i])
                    self.sampled_depths.append(self.sample([j, i]))
                    self.gpr.fit(self.visited, self.sampled_depths)
                    self.mu, self.sigma = self.gpr.predict(
                        self.grid, return_std=True)
                    self.update_mesh()

    def variance_of_the_path(self, P, Q):
        var = 0

        if (abs(P[0]-Q[0]) >= abs(P[1]-Q[1])):

            if P[0] < Q[0]:
                i = P[0]
                while (i + 0.25/self.osp < Q[0]):
                    i += 0.25/self.osp
                    j = self.get_y(P, Q, i)
                    mu, sigma = self.gpr.predict(
                        np.array([[i, j]]), return_std=True)
                    # var += (self.mu[self.findloc(self.X_grid, [i, j])]+np.sqrt(self.beta)*self.sigma[self.findloc(self.X_grid, [i, j])])
                    var += mu + np.sqrt(self.beta)*sigma

            else:
                i = Q[0]
                while (i + 0.25/self.osp < P[0]):
                    i += 0.25/self.osp
                    j = self.get_y(P, Q, i)
                    mu, sigma = self.gpr.predict(
                        np.array([[i, j]]), return_std=True)
                    # var += (self.mu[self.findloc(self.X_grid, [i, j])]+np.sqrt(self.beta)*self.sigma[self.findloc(self.X_grid, [i, j])])
                    var += mu + np.sqrt(self.beta)*sigma

        else:

            if P[1] < Q[1]:
                i = P[1]
                while (i + 0.25/self.osp < Q[1]):
                    i += 0.25/self.osp
                    j = self.get_x(P, Q, i)
                    mu, sigma = self.gpr.predict(
                        np.array([[i, j]]), return_std=True)
                    # var += (self.mu[self.findloc(self.X_grid, [i, j])]+np.sqrt(self.beta)*self.sigma[self.findloc(self.X_grid, [i, j])])
                    var += mu + np.sqrt(self.beta)*sigma

            else:
                i = Q[1]
                while (i + 0.25/self.osp < P[1]):
                    i += 0.25/self.osp
                    j = self.get_x(P, Q, i)
                    mu, sigma = self.gpr.predict(
                        np.array([[i, j]]), return_std=True)
                    # var += (self.mu[self.findloc(self.X_grid, [i, j])]+np.sqrt(self.beta)*self.sigma[self.findloc(self.X_grid, [i, j])])
                    var += mu + np.sqrt(self.beta)*sigma
        return var

    def learn(self, coordinates):
        self.samples += 1
        self.curr_coor = coordinates
        if (len(self.visited) > 0 and self.osp > 0):
            self.sample_on_the_way(self.visited[-1], coordinates)
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.visited, self.sampled_depths)
        self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)
        self.update_mesh()
        self.plot()
        return (self.get_next_coordinates(), self.samples == self.n_samples)

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
