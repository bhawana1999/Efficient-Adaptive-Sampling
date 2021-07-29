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
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5) #, n_restarts_optimizer=5

        self.samples = 0
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])
        self.update_mesh()

    def find_idx(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        idx = [0, 0]
        
        for i in range(self.meshgrid[0].shape[1]):
            if x==self.meshgrid[0][0][i]:
                idx[1] = i
                break
            
        for j in range(self.meshgrid[1].shape[0]):
            if y==self.meshgrid[1][j][0]:
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
        max_h = np.NINF

        for i in range(idx[0]-self.rad, idx[0]+self.rad+1):
            for j in range(idx[1]-self.rad, idx[1]+self.rad+1):
                if (0<=i<self.meshgrid[0].shape[0] and 0<=j<self.meshgrid[0].shape[1]):
                    if (self.meshgrid[2][i][j] + (self.meshgrid[3][i][j]*np.sqrt(self.beta)) > max_h):
                        max_h = self.meshgrid[2][i][j] + (self.meshgrid[3][i][j]*np.sqrt(self.beta))
                        new_idx = [i,j]

        predicted = self.meshgrid[2][new_idx[0]][new_idx[1]]
        ground_truth = self.sample([self.meshgrid[0][new_idx[0]][new_idx[1]], self.meshgrid[1][new_idx[0]][new_idx[1]]])
        if (abs(ground_truth - predicted) > self.thresh_gain):
            self.rad -= self.deltaParam
        elif (abs(ground_truth - predicted) < self.thresh_loss):
            self.rad += self.deltaParam

        return [self.meshgrid[0][new_idx[0]][new_idx[1]], self.meshgrid[1][new_idx[0]][new_idx[1]]]

    def learn(self, coordinates):
        self.samples += 1
        self.curr_coor = coordinates
        self.visited.append(coordinates)
        self.sampled_depths.append(self.sample(coordinates))
        self.gpr.fit(self.visited, self.sampled_depths)
        self.mu, self.sigma = self.gpr.predict(self.grid, return_std=True)
        self.update_mesh()
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