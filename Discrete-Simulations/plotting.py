import numpy as np
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
    
    a = np.zeros((3,4))

    tg_pairs = data.get(robot_name).get(grid_name).keys()

    tg_avg_eff = [mean(data.get(robot_name).get(grid_name).get(tg).get("efficiencies")) for tg in tg_pairs]

    print(tg_avg_eff)

    #tg_pairs_trimmed = [tg.replace("theta ", "").replace("gamma ", "") for tg in tg_pairs]
    
    for i in range(12):
        x = i//4
        y = i%4
        a[x,y] = tg_avg_eff[i]
    
    plt.imshow(a, cmap='hot')
    plt.title(grid_name)
    plt.ylabel("theta")
    plt.yticks([0,1,2], ['0.2', '1.0', '5.0'])
    plt.xlabel("gamma")
    plt.xticks([0,1,2,3], ['0.2','0.5','0.8','1.0'])
    plt.colorbar()
    plt.show()



with open("PI_results.json") as file:
    data = json.load(file)
print("overall eff: ", calc_overall_avg_eff(data, "PI"))

for grid in data.get("PI").keys():
    plot_efficiencies(data, "PI", grid)


