import numpy as np
from ar_agent_mod import AR_agent
from env import environment
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples to be taken by the agent", 
                    default=20, type=int)
parser.add_argument("--beta", help="hyperparameter that dictates exploration vs. exploitation", 
                    default=100., type=float)
parser.add_argument("--thresh_gain", help="threshold for information gain", default=0.50, type=float)
parser.add_argument("--thresh_loss", help="threshold for information loss", default=0.25, type=float)
parser.add_argument("--del_rad", help="change to be made in the radius", default=4, type=int)
parser.add_argument("--init_rad", help="initial radius", default=2, type=int)
parser.add_argument("--name", help="give a name to the experiment", 
                    default=None, type=str)
args = parser.parse_args()

directory = "output/" + args.name + "_samples_" + str(args.n)\
            if args.name != None else "output/" + "samples_" + str(args.n)
os.makedirs(directory, exist_ok = True)

env = environment()
agent = AR_agent(env=env, beta=args.beta, n_samples=args.n, directory=directory, init_radius=args.init_rad,
                    threshold_gain=args.thresh_gain, threshold_loss=args.thresh_loss, deltaParam=args.del_rad)

done = False
init_cor = [env.meshgrid[0][0][0], env.meshgrid[1][0][0]]
while not done:
    init_cor, done = agent.learn(init_cor)
