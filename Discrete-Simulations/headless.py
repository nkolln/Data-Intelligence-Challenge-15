# Import our robot algorithm to use in this simulation:
import robot_configs.greedy_random_robot as grr
import robot_configs.sarsa as sarsa
import robot_configs.qlearning as ql
import pickle
from environment import Robot
#import matplotlib.pyplot as plt

import time
import pandas as pd
from itertools import product
from tqdm import tqdm
import multiprocessing
from sys import argv

if len(argv) > 1 and argv[1] == 'test':
    bot_funcs = [grr.robot_epoch]
    bot_labels = ['greedy-random-robot']
    grid_files = ['empty.grid']
    gammas = [0]
    alphas = [0]
    epsilons = [0]
    runs = list(range(20))
    nr_iters = 20

elif len(argv) > 1 and argv[1] == 'hyperparameter-search':
    #about 1.5h runtime (45 min per algorithm)
    bot_funcs = [ql.robot_epoch, sarsa.robot_epoch]
    bot_labels = ['Q-learning', 'SARSA']
    grid_files = ['empty.grid', 'example-random-house-0.grid', 'rooms-with-furniture.grid'] #81, 89, 90
    gammas = [0.2, 0.5, 0.8]
    alphas = [0.2, 0.5, 0.8]
    epsilons = [0.2, 0.5, 0.8]
    runs = list(range(5))
    nr_iters = 810
    
elif len(argv) > 1 and argv[1] == 'MC':
    pass

elif len(argv) > 1 and argv[1] == 'QL-SARSA-test':
    #about 20 min runtime (10 min per algorithm)
    bot_funcs = [ql.robot_epoch, sarsa.robot_epoch]
    bot_labels = ['Q-learning', 'SARSA']
    grid_files = ['empty.grid', 'example-random-house-0.grid', 'rooms-with-furniture.grid'] #81, 89, 90
    gammas = [0]
    alphas = [0]
    epsilons = [0]
    runs = list(range(30))
    nr_iters = 180

elif len(argv) > 1:
    raise Exception(f"Unknown argument {argv[1]}")
    
else:
    raise Exception(f"Please provide 1 argument, {len(argv)-1} detected.")


progress_bar = tqdm(total=nr_iters)

def progress_update(*args):
    global progress_bar
    progress_bar.update()

def run_test(i):
    global bot_labels
    # Cleaned tile percentage at which the room is considered 'clean':
    stopping_criteria = 100

    # Keep track of some statistics:
    #efficiencies = []
    #n_moves = []
    deaths = 0
    #cleaned = []
    
    #bot_labels = ['Q-learning', 'SARSA']
    results = {}
    bot_func = i[0]
    bot_label = bot_labels[bot_funcs.index(bot_func)]
    results['bot'] = bot_label
    grid_file = i[1]
    results['grid'] = grid_file
    
    if i[2] == 0 and bot_label == "Q-learning":
        gamma = 0.5
    elif i[2] == 0 and bot_label == "SARSA":
        gamma = 0.8
    else:
        gamma = i[2]
    results['gamma'] = gamma
    
    if i[3] == 0 and bot_label == "Q-learning":
        alpha = 0.8
    elif i[3] == 0 and bot_label == "SARSA":
        alpha = 0.2
    else:
        alpha = i[3]
    results['alpha'] = alpha
    
    if i[4] == 0 and (bot_label == "Q-learning" or bot_label == "SARSA"):
        epsilon = 0.5
    else:
        epsilon = i[4]
    results['epsilon'] = epsilon
    
    run = i[5]
    results['run'] = run
    
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    # if battery:
    #     robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.25, battery_drain_lam=2)
    # else:
    robot = Robot(grid, (1, 1), orientation='n')
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    
    start = time.time()
    while time.time() - start <= 25:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        bot_func(robot, alpha, gamma, epsilon)
        # Stop this simulation instance if robot died :( :
        if not robot.alive:
            deaths += 1
            break
        # Calculate some statistics:
        clean = (grid.cells == 0).sum()
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= stopping_criteria and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    # Keep track of the last statistics for each simulation instance:
    end = time.time()
    runtime = end - start
    results['runtime'] = runtime
    results['efficiency'] = float(efficiency)
    #n_moves.append(len(robot.history[0]))
    results['cleaned'] = clean_percent
    return results

if __name__=='__main__':
    
    results = {'bot': [], 'grid':[], 'gamma':[], 'alpha':[], 'epsilon':[],
               'run':[], 'efficiency':[], 'runtime':[], 'cleaned':[]}

    parameters = product(bot_funcs, grid_files, gammas, alphas, epsilons, runs)
    
    #run tests concurrently on 4 cores
    with multiprocessing.Pool(4) as pool:
        processes = [pool.apply_async(run_test, args=(x,), callback=progress_update) for x in parameters]
        test_results = [p.get() for p in processes]
    
    #combine all dictionaries in test_results into one big dict
    for k in results.keys():
        results[k] = [d[k] for d in test_results]
        
    df = pd.DataFrame.from_dict(results)
    df.to_csv(f'results-{argv[1]}.csv', index=False)

# Make some plots:
# plt.hist(cleaned)
# plt.title('Percentage of tiles cleaned.')
# plt.xlabel('% cleaned')
# plt.ylabel('count')
# plt.show()

# plt.hist(efficiencies)
# plt.title('Efficiency of robot.')
# plt.xlabel('Efficiency %')
# plt.ylabel('count')
# plt.show()
