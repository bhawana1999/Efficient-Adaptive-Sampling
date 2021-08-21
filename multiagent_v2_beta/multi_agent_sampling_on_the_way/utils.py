import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math


def plot(env_sample, meshgrid, agent, n_sample, directory):
    fig = plt.figure(figsize=(10, 10))
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
                      meshgrid.mu.reshape(meshgrid.meshgrid[0].shape), alpha=0.5, color='g')
    ax.plot_wireframe(meshgrid.meshgrid[0], meshgrid.meshgrid[1],
                      env_sample(meshgrid.meshgrid), alpha=0.5, color='b')
    markers = ['o', '^', 's']
    color = ['black', 'lightcoral', 'magenta']

    for idx in range(len(agent)):
        ax.scatter([x[0] for x in agent[idx].visited], [x[1] for x in agent[idx].visited], agent[idx].sampled_depths, c=color[idx],
                   marker=markers[idx], alpha=1.0, s=70)
    plt.title("Iteration "+str(n_sample)+" Prediction vs. ground truth")
    plt.savefig(directory+"/"+str(n_sample)+".png",
                bbox_inches='tight', pad_inches=0)
    # plt.show()


def plot_sigma(meshgrid, agent, n_sample, directory):
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
    plt.savefig(directory+"_sigma/"+str(n_sample) +
                ".png", bbox_inches='tight', pad_inches=0)
    # plt.show()


def get_cors(meshgrid, Agents, beta):
    agents = deepcopy(Agents)
    # grid = meshgrid.grid.copy()
    # grid = grid.reshape(meshgrid.meshgrid[0].shape)
    mu = meshgrid.mu.copy()
    sigma = meshgrid.sigma.copy()

    heuristic = mu + sigma*np.sqrt(beta)
    heuristic = heuristic.reshape(meshgrid.meshgrid[0].shape)
    goals = [0]*len(Agents)
    backCors = [0]*len(Agents)
    for _ in range(len(agents)):
        idx = np.unravel_index(np.argmax(heuristic), heuristic.shape)
        cor = meshgrid.meshgrid[0][idx], meshgrid.meshgrid[1][idx] 

        dist = []
        for i in range(len(agents)):
            x, y = cor[0], cor[1]
            X, Y = agents[i].visited[-1][0], agents[i].visited[-1][1]
            dist.append(np.hypot(x-X, y-Y))

        agent_idx = np.argmin(np.array(dist))
        goals[agents[agent_idx].id] = cor
        start_idx = find_idx(meshgrid, agents[agent_idx].visited[-1])
        heuristic, back = dijkstra(heuristic, start_idx, idx)

        back_temp = []
        for b in back:
            back_temp.append([meshgrid.meshgrid[0][b[0],b[1]], meshgrid.meshgrid[1][b[0],b[1]]])

        backCors[agents[agent_idx].id] = back_temp
        __ = agents.pop(agent_idx)
    
    return goals, backCors


def dijkstra(heuristic, start, goal):
    H = -heuristic
    costMap = np.ones(heuristic.shape)*np.Infinity
    costMap[start[0], start[1]] = 0
    backTrack = np.ones(heuristic.shape, dtype=int)*np.nan
    visited = np.zeros(heuristic.shape, dtype=bool)
    done = False
    x, y = start[0], start[1]
    Xmax = heuristic.shape[0]
    Ymax = heuristic.shape[1]

    while not done:
        # move to x+1, y
        if x < Xmax-1:
            if H[x+1, y]+costMap[x, y] < costMap[x+1, y] and not visited[x+1, y]:
                costMap[x+1, y] = H[x+1, y]+costMap[x, y]
                backTrack[x+1, y] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x-1, y
        if x > 0:
            if H[x-1, y]+costMap[x, y] < costMap[x-1, y] and not visited[x-1, y]:
                costMap[x-1, y] = H[x-1, y]+costMap[x, y]
                backTrack[x-1, y] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x, y+1
        if y < Ymax-1:
            if H[x, y+1]+costMap[x, y] < costMap[x, y+1] and not visited[x, y+1]:
                costMap[x, y+1] = H[x, y+1]+costMap[x, y]
                backTrack[x, y+1] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x, y-1
        if y > 0:
            if H[x, y-1]+costMap[x, y] < costMap[x, y-1] and not visited[x, y-1]:
                costMap[x, y-1] = H[x, y-1]+costMap[x, y]
                backTrack[x, y-1] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x+1, y+1
        if x < Xmax-1 and y < Ymax-1:
            if H[x+1, y+1]+costMap[x, y] < costMap[x+1, y+1] and not visited[x+1, y+1]:
                costMap[x+1, y+1] = H[x+1, y+1]+costMap[x, y]
                backTrack[x+1, y+1] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x-1, y-1
        if x > 0 and y > 0:
            if H[x-1, y-1]+costMap[x, y] < costMap[x-1, y-1] and not visited[x-1, y-1]:
                costMap[x-1, y-1] = H[x-1, y-1]+costMap[x, y]
                backTrack[x-1, y-1] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x+1, y-1
        if x < Xmax-1 and y > 0:
            if H[x+1, y-1]+costMap[x, y] < costMap[x+1, y-1] and not visited[x+1, y-1]:
                costMap[x+1, y-1] = H[x+1, y-1]+costMap[x, y]
                backTrack[x+1, y-1] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        # move to x-1, y+1
        if x > 0 and y < Ymax-1:
            if H[x-1, y+1]+costMap[x, y] < costMap[x-1, y+1] and not visited[x-1, y+1]:
                costMap[x-1, y+1] = H[x-1, y+1]+costMap[x, y]
                backTrack[x-1, y+1] = np.ravel_multi_index([x, y], (Xmax, Ymax))

        visited[x, y] = True
        costMapTemp = costMap.copy()
        costMapTemp[np.where(visited)] = np.Infinity
        min_idx = np.unravel_index(np.argmin(costMapTemp), costMapTemp.shape)
        x, y = min_idx[0], min_idx[1]
        if x == goal[0] and y == goal[1]:
            done = True

    # backtracking
    # x, y = goal[0], goal[1]
    backCors = []
    heuristic[x, y] = 0
    while True:
        next_id = backTrack[np.int(x), np.int(y)]
        if math.isnan(next_id):
            break
        else:
            next_id = np.int(next_id)
        idx = np.unravel_index(next_id, (Xmax, Ymax))
        x, y = idx[0], idx[1]
        backCors.append([x, y])
        heuristic[x, y] = 0

    return heuristic, backCors


def find_idx(meshgrid, coordinates):
    x = coordinates[0]
    y = coordinates[1]
    idx = [0, 0]

    for i in range(meshgrid.meshgrid[0].shape[1]):
        if x == meshgrid.meshgrid[0][0][i]:
            idx[1] = i
            break

    for j in range(meshgrid.meshgrid[1].shape[0]):
        if y == meshgrid.meshgrid[1][j][0]:
            idx[0] = j
            break
    return idx
