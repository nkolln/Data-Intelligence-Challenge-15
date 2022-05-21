import copy
import json
import time

import pickle
import os
from environment import Robot

import robot_configs.MonteCarlo as MC
from itertools import product
import pandas as pd

epsilons = [0.2,0.5,0.8]
grid_files = ['empty.grid', 'example-random-house-0.grid', 'rooms-with-furniture.grid'] #81, 89, 90

results = {'epsilon':[], 'grid':[], 'efficiency':[], 'cleaned':[]}

i = 0
for epsilon, grid_file in product(epsilons, grid_files):
    i += 1
    print(f'training bot {i} of 9')
    results['epsilon'].append(epsilon)
    results['grid'].append(grid_file)
    # get the grid
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # initialize the robot
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.1, battery_drain_lam=2)
    # train the robot
    policy = MC.train(robot, episodes=1000, epsilon=epsilon)
    
    # re-initialize the robot
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # test the efficiency of the policy
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.1, battery_drain_lam=2)

    efficiency, clean = MC.test_pi(robot, policy, iter_count=5)
    results['efficiency'].append(efficiency)
    results['cleaned'].append(clean)
    
df = pd.DataFrame.from_dict(results)
df.to_csv('results-MC-hyperparameter-search.csv')

