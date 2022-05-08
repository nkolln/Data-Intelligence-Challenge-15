import matplotlib.pyplot as plt
import json
from statistics import mean

"""
data looks like:
{VI: grid1{theta-gamma pair1:{efficiencies:[],n_moves:[],cleaned_percentage[]},theta-gama pair2:{}},
     grid2{} }  
"""


def calc_overall_avg_eff(data, robot_name):
    avg = 0
    robot = data.get(robot_name)
    for grid in robot.keys():
        grid_dict = robot.get(grid)

        for tg_pair in grid_dict.keys():
            temp_avg = mean(grid_dict.get(tg_pair).get("efficiencies"))
            avg += temp_avg
    avg = avg/(12*4)
    return avg

def plot_efficiencies(data, robot_name, grid_name):

    tg_pairs = data.get(robot_name).get(grid_name).keys()

    tg_avg_eff = [mean(data.get(robot_name).get(grid_name).get(tg).get("efficiencies")) for tg in tg_pairs]

    tg_pairs_trimmed = [tg.replace("theta ", "").replace("gamma ", "") for tg in tg_pairs]
    plt.plot(tg_pairs_trimmed, tg_avg_eff)
    plt.title(grid_name)
    plt.xlabel("theta-gamma pairs")
    plt.ylabel("average efficiency")
    plt.show()



with open("results.json") as file:
    data = json.load(file)
print("overall eff: ", calc_overall_avg_eff(data, "VI"))

for grid in data.get("VI").keys():
    plot_efficiencies(data, "VI", grid)


