
import copy
import json
import time

import pickle
import os
from environment import Robot

import robot_configs.MonteCarlo as MC

with open(f'test_grids/1example-random-house-0.grid', 'rb') as f:
    grid = pickle.load(f)

robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.25, battery_drain_lam=2)

policy = MC.train_monte_carlo_e_soft(robot, episodes=10000)

with open(f'test_grids/1example-random-house-0.grid', 'rb') as f:
    grid = pickle.load(f)




robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.25, battery_drain_lam=2)

print("result of testing: ", MC.test_policy(robot, policy, iter_count=300))


