import numpy as np
from gp_ucb import GPUCB_agent
from env import environment
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import argparse
import os

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
os.makedirs(directory, exist_ok = True)

env = environment()
agent = GPUCB_agent(env=env, beta=args.beta, n_samples=args.n, directory=directory)

done = False
init_cor = [env.meshgrid[0][0][0], env.meshgrid[1][0][0]]
while not done:
    init_cor, done = agent.learn(init_cor)
