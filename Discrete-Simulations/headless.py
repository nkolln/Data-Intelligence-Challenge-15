# Import our robot algorithm to use in this simulation:
# from robot_configs.greedy_random_robot import robot_epoch
import copy
import json
import time

import robot_configs.value_iteration2_weighting as VI
import robot_configs.policy_iteration as PI

import pickle
import os
from environment import Robot
import matplotlib.pyplot as plt
from statistics import mean


def calc_grid_avg_eff(stats: dict, robot_name, grid_name):
    robot_stats = stats.get(robot_name)
    efficiency_array = robot_stats.get(grid_name).get("efficiencies")
    avg = mean(efficiency_array)
    return avg


def calc_overall_avg_eff(stats: dict, robot_name):
    avg = 0
    robot_stats = stats.get(robot_name)

    for grid_name in robot_stats.keys():
        temp_avg = calc_grid_avg_eff(stats, robot_name, grid_name)
        avg += temp_avg

    avg = avg/len(robot_stats.keys())
    return avg


grid_files = os.listdir("test_grids")
# print(grid_files)
# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100
iter_count = 5

# Keep track of some statistics:
statistics = {"VI": {},
              # "PI": {}
              }
# initialize the dictionary
for grid_name in grid_files:
    statistics.get("VI").update({grid_name: {}})
    # statistics.get("PI").update({grid_name: {}})
# print(statistics)

for robot_name in statistics.keys():
    robot_start = time.time()
    # run for each grid file
    for i in range(len(grid_files)):
        grid_start = time.time()

        grid_file = grid_files[i]
        # get the grid file as a grid
        with open(f'grid_configs/{grid_file}', 'rb') as f:
            original_grid = pickle.load(f)
        print("grid: ", grid_file)
        # initialize statistics
        efficiencies = []
        n_moves = []
        deaths = 0
        cleaned = []
        # Run iter_count times for each grid :
        for j in range(iter_count):
            grid = copy.deepcopy(original_grid)
            iter_start = time.time()
            print("iteration ", j)

            # Calculate the total visitable tiles:
            n_total_tiles = (grid.cells >= 0).sum()
            # Spawn the robot at (1,1) facing north with battery drainage enabled:
            robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
            # Keep track of the number of robot decision epochs:
            n_epochs = 0
            while True:
                # print("lol")
                n_epochs += 1
                # Do a robot epoch (basically call the robot algorithm once):
                # if robot_name == "VI":
                VI.robot_epoch(robot)
                # else:
                #     PI.robot_epoch(robot)
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
            print("eff: ", efficiency)
            efficiencies.append(float(efficiency))
            n_moves.append(len(robot.history[0]))
            cleaned.append(clean_percent)
            print("one iter took: ", time.time()-iter_start, " seconds")

        # if robot_name == "VI":
        statistics.get("VI").get(grid_file).update({"efficiencies": efficiencies,
                                                    "n_moves": n_moves,
                                                    "cleaned": cleaned})
        # else:
        #     statistics.get("PI").get(grid_file).update({"efficiencies": efficiency,
        #                                                 "n_moves": n_moves,
        #                                                 "cleaned": cleaned})
        print("one grid took: ", time.time() - grid_start, " seconds")

    print("one robot took: ", time.time() - robot_start, " seconds")

with open("results.json", "w") as json_file:
    json.dump(statistics, json_file)

print("efficiency of VI: ", calc_overall_avg_eff(statistics, "VI"))

# Make some plots:
# plt.figure()
# plt.bar()


# plt.hist(cleaned)
# plt.title('Percentage of tiles cleaned.')
# plt.xlabel('% cleaned')
# plt.ylabel('count')
# plt.show()
#
# plt.hist(efficiencies)
# plt.title('Efficiency of robot.')
# plt.xlabel('Efficiency %')
# plt.ylabel('count')
# plt.show()
