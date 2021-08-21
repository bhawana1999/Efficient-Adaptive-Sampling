import numpy as np
from numpy.random import default_rng
from ar_agent_mod import AR_agent
from env import env_sample
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import argparse
import os
import warnings


class mesh():
    def __init__(self):
        X, Y = np.round(np.arange(-3, 3, 0.025, dtype=np.float),
                        3), np.round(np.arange(-3, 3, 0.025, dtype=np.float), 3)
        self.meshgrid = np.meshgrid(X, Y)
        self.grid = np.c_[self.meshgrid[0].ravel(), self.meshgrid[1].ravel()]
        self.mu = np.zeros(self.grid.shape[0])
        self.sigma = 0.5 * np.ones(self.grid.shape[0])

        mu = self.mu.copy()
        sigma = self.sigma.copy()
        mu = mu.reshape(self.meshgrid[0].shape)
        sigma = sigma.reshape(self.meshgrid[0].shape)

        self.meshgrid.append(mu)
        self.meshgrid.append(sigma)
        self.visited = []
        self.sampled_depths = []

        self.occupied_coordinates=[]

    def update_mesh(self):
        mu = self.mu.copy()
        sigma = self.sigma.copy()
        mu = mu.reshape(self.meshgrid[0].shape)
        sigma = sigma.reshape(self.meshgrid[0].shape)
        
        self.meshgrid[2] = mu
        self.meshgrid[3] = sigma

    def clean(self):
        self.occupied_coordinates = []


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
    plt.title("Iteration "+str(n_sample)+" Prediction vs. ground truth")
    plt.savefig(directory+"/"+str(n_sample)+".png", bbox_inches='tight',pad_inches = 0)
    # plt.show()

def plot_sigma(meshgrid, agent, n_sample):
    fig = plt.figure(figsize=(10, 10))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
                      meshgrid.sigma.reshape(meshgrid.meshgrid[0].shape), alpha=0.5, color='g')
    ax.set_zlim3d(0, 2)
    markers = ['o', '^', 's']
    color = ['black', 'lightcoral', 'magenta']
    for idx in range(len(agent)):
        ax.scatter([x[0] for x in agent[idx].visited], [x[1] for x in agent[idx].visited], c=color[idx],
                   marker=markers[idx], alpha=1.0, s=70)
    plt.title("Iteration "+str(n_sample)+" Standard deviation")
    plt.savefig(directory+"_sigma/"+str(n_sample)+".png", bbox_inches='tight',pad_inches = 0)
    # plt.show()



if __name__=="__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help="number of samples to be taken by the agent", 
                        default=20, type=int)
    parser.add_argument("--n_agents", help="number of agents", default=3, type=int)
    parser.add_argument("--beta", help="hyperparameter that dictates exploration vs. exploitation", 
                        default=500., type=float)
    parser.add_argument("--thresh_gain", help="threshold for information gain", default=0.50, type=float)
    parser.add_argument("--thresh_loss", help="threshold for information loss", default=0.25, type=float)
    parser.add_argument("--del_rad", help="change to be made in the radius", default=4, type=int)
    parser.add_argument("--init_rad", help="initial radius", default=2, type=int)
    parser.add_argument("--name", help="give a name to the experiment", 
                        default=None, type=str)
    args = parser.parse_args()

    directory = "output/" + args.name + "_samples_" + str(args.n)\
                if args.name != None else "output/" + "samples_" + str(args.n)

    directory_sigma = "output/" + args.name + "_samples_" + str(args.n)+"_sigma"\
                if args.name != None else "output/" + "samples_" + str(args.n)+"_sigma"
    os.makedirs(directory, exist_ok = True)
    os.makedirs(directory_sigma, exist_ok=True)

    # init_cor = [[-3, -3], [-2.5, -3], [-2, -3]]
    init_cor = [[-3, -3], [0, -3], [2.9, -3]]

    meshgrid = mesh()
    agents = []
    for i in range(args.n_agents):
        agents.append(AR_agent(mesh=meshgrid, env_sample=env_sample, beta=args.beta, 
                        n_samples=args.n, directory=directory, init_radius=args.init_rad,
                        threshold_gain=args.thresh_gain, threshold_loss=args.thresh_loss, 
                        deltaParam=args.del_rad))
    
    
    for sample in range(args.n):
        # print(sample)
        preds = []
        truths = []
        meshgrid.clean()
        for i in range(args.n_agents):
            predicted, gt = agents[i].learn(init_cor[i])
            preds.append(predicted)
            truths.append(gt)

        if sample == 0:
            init_cor = []
            for i in range(args.n_agents):
                init_cor.append(agents[i].get_next_coordinates())
                # print(init_cor[-1])

        else:
            init_cor = []
            for i in range(args.n_agents):
                agents[i].update_radius(preds[i], truths[i])
                init_cor.append(agents[i].get_next_coordinates())
                # print(init_cor[-1])
        plot(meshgrid, agents, sample)
        plot_sigma(meshgrid, agents, sample)
