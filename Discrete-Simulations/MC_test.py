
import copy
import json
import time

import pickle
import os
from environment import Robot

import robot_configs.MonteCarlo as MC
# get the grid
with open(f'test_grids/custom2_corridor.grid', 'rb') as f:
    grid = pickle.load(f)
# initialize the robot
robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.1, battery_drain_lam=2)
# train the robot
policy = MC.train(robot, episodes=10000, epsilon=0.1)

# re-initialize the robot
with open(f'test_grids/1example-random-house-0.grid', 'rb') as f:
    grid = pickle.load(f)
# test the efficiency of the policy
robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.1, battery_drain_lam=2)
print("result of testing: ", MC.test_pi(robot, policy, iter_count=300))


