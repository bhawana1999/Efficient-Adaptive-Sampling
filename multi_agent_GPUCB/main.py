import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
# from env import env_sample
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
# from agents import GPUCB_agent
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples to be taken by the agent",
                    default=20, type=int)
parser.add_argument("--beta", help="hyperparameter that dictates exploration vs. exploitation",
                    default=100., type=float)
parser.add_argument("--name", help="give a name to the experiment",
                    default=None, type=str)
args = parser.parse_args()

directory = "output/" + args.name + "_samples_" + str(args.n)\
            if args.name != None else "output/" + "samples_" + str(args.n)
os.makedirs(directory, exist_ok=True)


class mesh:
    def __init__(self):
        X, Y = np.round(np.arange(-3, 3, 0.025, dtype=np.float),
                        3), np.round(np.arange(-3, 3, 0.025, dtype=np.float), 3)
        self.meshgrid = np.meshgrid(X, Y)
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])


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


def env_sample(coordinates):
    return np.sin(coordinates[0])+np.cos(coordinates[1])

def plot(meshgrid, agent):
    fig = plt.figure(figsize=(10,10))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
        meshgrid.mu.reshape(meshgrid.meshgrid[0].shape), alpha=0.5, color='g')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
        env_sample(meshgrid.meshgrid), alpha=0.5, color='b')
    markers = ['o', '^', 's']
    color = ['black', 'lightcoral', 'magenta']
    for idx, a in enumerate(agent):
        ax.scatter([x[0] for x in a.visited], [x[1] for x in a.visited], a.sampled_depths, c=color[idx],
            marker=markers[idx], alpha=1.0)
    # plt.savefig(self.directory+"/"+str(self.samples)+".png")
    plt.show()

def get_cors(meshgrid, agents, beta):
    grid = meshgrid.grid.copy()
    mu = meshgrid.mu.copy()
    sigma = meshgrid.sigma.copy()

    idx = []
    cor = []
    for _ in range(len(agents)):
        idx.append(np.argmax(mu + sigma * np.sqrt(beta)))
        mu[idx[-1]] = np.NINF
        cor.append(grid[idx[-1]])
        for __ in range(2000):
            mu[np.argmax(mu + sigma * np.sqrt(beta))] = np.NINF

    return cor

def allocate_cors(agents, cors):
    new_cors = []
    cors = cors.copy()
    for a in agents:
        x, y = a.visited[-1][0], a.visited[-1][1]
        dist = []
        for c in cors:
            dist.append(np.hypot((x-c[0]), (y-c[1])))
        idx = np.argmin(dist)
        new_cors.append(cors[idx])
        _ = cors.pop(idx)
    return new_cors


init_cor = [[-3, -3], [-2.5, -3], [-2, -3]]

meshgrid = mesh()
agents = [GPUCB_agent(mesh=meshgrid, env_sample=env_sample, beta=args.beta, n_samples=args.n, directory=directory)]*3


for _ in range(args.n):
    for i in range(len(agents)):
        agents[i].learn(init_cor[i])
    init_cor = get_cors(meshgrid, agents, args.beta)
    plot(meshgrid, agents)