import numpy as np
from numpy.random import default_rng
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from env import env_sample
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from agents import GPUCB_agent
import argparse
import os
import warnings
from utils import plot_sigma, plot, get_cors


class mesh:
    def __init__(self):
        X, Y = np.round(np.arange(-3, 3, 0.025, dtype=np.float),
                        3), np.round(np.arange(-3, 3, 0.025, dtype=np.float), 3)
        self.meshgrid = np.meshgrid(X, Y)
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])
        self.visited = []
        self.sampled_depths = []


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="number of samples to be taken by the agent",
                        default=20, type=int)
    parser.add_argument(
        "--n_agents", help="number of agents", default=3, type=int)
    parser.add_argument("--beta", help="hyperparameter that dictates exploration vs. exploitation",
                        default=100., type=float)
    parser.add_argument("--name", help="give a name to the experiment",
                        default=None, type=str)
    args = parser.parse_args()

    directory = "output/" + args.name + "_samples_" + str(args.n)\
                if args.name != None else "output/" + "samples_" + str(args.n)
    directory_sigma = "output/" + args.name + "_samples_" + str(args.n)+"_sigma"\
        if args.name != None else "output/" + "samples_" + str(args.n)+"_sigma"
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory_sigma, exist_ok=True)

    init_cor = [[-3, -3], [-2.5, -3], [-2, -3]]
    # init_cor = [[-3, -3], [0, -3], [2.9, -3]]

    meshgrid = mesh()
    agents = []
    for i in range(args.n_agents):
        agents.append(GPUCB_agent(mesh=meshgrid, env_sample=env_sample,
                      beta=args.beta, n_samples=args.n, directory=directory, id=i))

    for _ in range(args.n):
        rng = default_rng()
        idxs = rng.choice(3, size=3, replace=False)
        for i in idxs:
            agents[i].learn(init_cor[i])
        init_cor = get_cors(meshgrid, agents, args.beta)
        plot(env_sample, meshgrid, agents, _, directory)
        plot_sigma(meshgrid, agents, _, directory)
