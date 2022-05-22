import copy
import json
import time

import pickle
import os
from environment import Robot

import robot_configs.MonteCarlo as MC
from itertools import product
import pandas as pd
from sys import argv
import multiprocessing

if len(argv) > 1 and argv[1]=='hyperparameter-search':
    #run tests to determine best epsilon value, takes about 1h
    epsilons = [0.2,0.5,0.8]
    nr_bots = 9
    test_iters = 5

elif len(argv) > 1 and argv[1]=='test':
    #run tests with the best epsilon value, takes about 20 min
    epsilons = [0.5]
    nr_bots = 3
    test_iters = 30
    
elif len(argv) > 1:
    raise Exception(f"Unknown argument {argv[1]}")
    
else:
    raise Exception(f"Please provide 1 argument, {len(argv)-1} detected")
    
grid_files = ['empty.grid', 'example-random-house-0.grid', 'rooms-with-furniture.grid'] #81, 89, 90
i = 0

def progress_update(*args):
    global i
    global nr_bots
    i += 1
    print(f"finished bot {i} of {nr_bots}")

def train_and_test(x):
#for epsilon, grid_file in product(epsilons, grid_files):
    results = {}
    epsilon = x[0]
    grid_file = x[1]
    #i += 1
    #print(f'training bot {i} of {nr_bots}')
    #results['epsilon'].append(epsilon)
    #results['grid'].append(grid_file)
    results['epsilon'] = epsilon
    results['grid'] = grid_file
    # get the grid
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # initialize the robot
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.1, battery_drain_lam=2)
    # train the robot
    policy = MC.train(robot, episodes=1000, epsilon=epsilon)
    
    #print(f'testing bot {i} of {nr_bots}')
    # re-initialize the robot
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # test the efficiency of the policy
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.1, battery_drain_lam=2)

    efficiency, clean, runtime = MC.test_pi(robot, policy, iter_count=test_iters)
    #results['efficiency'].append(efficiency)
    #results['cleaned'].append(clean)
    results['efficiency'] = efficiency
    results['cleaned'] = clean
    results['runtime'] = runtime
    return results
    

if __name__=='__main__':
    results = {'epsilon':[], 'grid':[], 'efficiency':[], 'cleaned':[], 'runtime':[]}
    
    parameters = product(epsilons, grid_files)
    
    with multiprocessing.Pool(3) as pool:
        processes = [pool.apply_async(train_and_test, args=(x,), callback=progress_update) for x in parameters]
        test_results = [p.get() for p in processes]
    
    for k in results.keys():
        results[k] = [d[k] for d in test_results]
    
    df = pd.DataFrame.from_dict(results)
    df.to_csv(f'results-MC-{argv[1]}.csv', index=False)

