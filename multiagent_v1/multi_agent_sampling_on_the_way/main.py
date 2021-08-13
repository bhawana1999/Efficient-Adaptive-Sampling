import numpy as np
from sow_agent import SOW_agent
from env import env_sample
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import argparse
import os
import warnings


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


def plot(meshgrid, agent, n_sample):
    fig = plt.figure(figsize=(10, 10))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
                      meshgrid.mu.reshape(meshgrid.meshgrid[0].shape), alpha=0.5, color='g')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
                      env_sample(meshgrid.meshgrid), alpha=0.5, color='b')
    markers = ['o', '^', 's']
    color = ['black', 'lightcoral', 'magenta']
    # for idx, a in enumerate(agent):
    #     ax.scatter([x[0] for x in a.visited], [x[1] for x in a.visited], a.sampled_depths, c=color[idx],
    #                marker=markers[idx], alpha=1.0)
    for idx in range(len(agent)):
        ax.scatter([x[0] for x in agent[idx].visited], [x[1] for x in agent[idx].visited], agent[idx].sampled_depths, c=color[idx],
                   marker=markers[idx], alpha=1.0, s=70)
    plt.savefig(directory+"/"+str(n_sample)+".png", bbox_inches='tight',pad_inches = 0)
    # plt.show()


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

    cor = allocate_cors(agents, cor)

    return cor


def allocate_cors(agents, cors):
    n = len(agents)
    dist_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x, y = cors[j][0], cors[j][1]
            X, Y = agents[i].visited[-1][0], agents[i].visited[-1][1]
            dist_mat[i][j] = np.hypot(x-X, y-Y)

    new_cor = [0]*n
    for _ in range(n):
        i, j = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
        new_cor[i] = cors[j]
        dist_mat[i,:] = 1e6
        dist_mat[:,j] = 1e6
    
    return new_cor

if __name__=="__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="number of samples to be taken by the agent", 
                        default=20, type=int)
    parser.add_argument(
        "--n_agents", help="number of agents", default=3, type=int)
    parser.add_argument("--osp", help="On the way sampling parameter", default=0.25, type=float)
    parser.add_argument("--beta", help="hyperparameter that dictates exploration vs. exploitation", 
                        default=100., type=float)
    parser.add_argument("--name", help="give a name to the experiment", 
                        default=None, type=str)
    args = parser.parse_args()

    directory = "output/" + args.name + "_samples_" + str(args.n)\
                if args.name != None else "output/" + "samples_" + str(args.n)
    os.makedirs(directory, exist_ok = True)

    init_cor = [[-3, -3], [-2.5, -3], [-2, -3]]

    meshgrid = mesh()
    agents = []
    for i in range(args.n_agents):
        agents.append(SOW_agent(mesh=meshgrid, env_sample=env_sample, beta=args.beta, n_samples=args.n, directory=directory, osp=args.osp))

    for _ in range(args.n):
        for i in range(len(agents)):
            agents[i].learn(init_cor[i])
        init_cor = get_cors(meshgrid, agents, args.beta)
        plot(meshgrid, agents, _)
