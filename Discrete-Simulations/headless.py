# Import our robot algorithm to use in this simulation:
import robot_configs.greedy_random_robot as grr
import robot_configs.idle_robot as idr
import pickle
from environment import Robot
#import matplotlib.pyplot as plt

import time
import pandas as pd
from itertools import product

results = {'bot': [], 'grid':[], 'battery':[], 'run':[], 'efficiency':[], 'runtime':[], 'cleaned':[]}

bot_funcs = [grr.robot_epoch]
bot_labels = ['greedy-random-robot']
grid_files = ['empty.grid', 'wall-furniture.grid',
              'example-random-house-0.grid', 'rooms-with-furniture.grid']
battery = [True, False]
runs = list(range(30))

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []
deaths = 0
cleaned = []

# Run 30 times:
for i in product(bot_funcs, grid_files, battery, runs):
    bot_func = i[0]
    results['bot'].append(bot_labels[bot_funcs.index(bot_func)])
    grid_file = i[1]
    results['grid'].append(grid_file)
    battery_status = i[2]
    results['battery'].append(battery_status)
    run = i[3]
    results['run'].append(run)
    
    start = time.time()
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    if battery:
        robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
    else:
        robot = Robot(grid, (1, 1), orientation='n')
    # Keep track of the number of robot decision epochs:
    n_epochs = 0
    while True:
        n_epochs += 1
        # Do a robot epoch (basically call the robot algorithm once):
        bot_func(robot)
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
    results['runtime'].append(runtime)
    results['efficiency'].append(float(efficiency))
    #n_moves.append(len(robot.history[0]))
    results['cleaned'].append(clean_percent)
    
df = pd.DataFrame.from_dict(results)
df.to_csv('results.csv', index=False)

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
