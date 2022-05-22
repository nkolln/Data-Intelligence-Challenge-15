import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from sys import argv

if len(argv) > 1 and argv[1]=='hyperparameters':
    hyperparameters = True
    main_results = False
elif len(argv) > 1 and argv[1]=='main-results':
    hyperparameters = False
    main_results = True
elif len(argv) > 1:
    raise Exception(f"Do not recognize argument {argv[1]}")
else:
    raise Exception("Please give 1 argument. Possible arguments are 'hyperparameters' and 'main-results'")


#%% Find best hyperparameters
if hyperparameters:
    per_bot = pd.read_csv('results-hyperparameter-search.csv').groupby('bot')
    Ql = per_bot.get_group('Q-learning')
    sarsa = per_bot.get_group('SARSA')
    
    print("Best performing combination of hyperparameters per algorithm:")
    mean_metrics_ql = Ql.groupby(['gamma','alpha','epsilon']).mean()
    mean_metrics_sarsa = sarsa.groupby(['gamma','alpha','epsilon']).mean()
    print("Q-learning (gamma, alpha, epsilon):")
    print(mean_metrics_ql['efficiency'].idxmax()) #gamma: 0.5, alpha: 0.8, epsilon: 0.5
    print("SARSA (gamma, alpha, epsilon):")
    print(mean_metrics_sarsa['efficiency'].idxmax()) #gamma: 0.8, alpha: 0.2, epsilon: 0.5
    
    MC = pd.read_csv('results-MC-hyperparameter-search.csv')
    mean_metrics_MC = MC.groupby('epsilon').mean()
    print('Monte Carlo (epsilon):')
    print(mean_metrics_MC['efficiency'].idxmax()) #epsilon: 0.5

#%% Plot performance per algorithm per grid

if main_results:
    #Divide data per grid:
    QL_sarsa = pd.read_csv('results-QL-SARSA-test.csv').groupby('bot')
    QL = QL_sarsa.get_group('Q-learning')
    sarsa = QL_sarsa.get_group('SARSA')
    MC = pd.read_csv('results-MC-test.csv')
    grr = pd.read_csv('results-baseline.csv')
    
    #MC aready has mean values, but we still need them for sarsa, QL and grr:
    mean_QL = QL.groupby('grid').mean()
    mean_sarsa = sarsa.groupby('grid').mean()
    mean_grr = grr.groupby('grid').mean()
    
    #plot efficiencies:
    plt.figure(0,figsize=(10,6))
    plt.bar(x = [0.7, 1.7, 2.7], height = mean_grr['efficiency'], width=0.2, color='white',edgecolor='black', label = "Greedy Random Robot")
    plt.bar(x = [0.9, 1.9, 2.9], height = MC['efficiency'], width=0.2, color='silver', edgecolor='black', label = 'MC')
    plt.bar(x = [1.1, 2.1, 3.1], height = mean_QL['efficiency'], width=0.2, color='dimgray', edgecolor='black', label = "Q-learning")
    plt.bar(x = [1.3, 2.3, 3.3], height = mean_sarsa['efficiency'], width=0.2, color='black', edgecolor='black', label = "SARSA")
    plt.xticks(ticks = [1,2,3], labels = ['empty-room', 'empty-house', 'furnished-house'], fontsize='large')
    plt.xlabel("grids", fontsize='x-large')
    plt.ylabel("efficiency (%)", fontsize='x-large')
    plt.yticks(fontsize='large')
    plt.ylim(0,101)
    plt.title("Mean efficiency per grid and algorithm", fontsize='xx-large')
    plt.savefig("./figure-2a.png")
    
    #plot cleanliness:
    plt.figure(1,figsize=(10,6))
    plt.bar(x = [0.7, 1.7, 2.7], height = mean_grr['cleaned'], width=0.2, color='white',edgecolor='black', label = "Greedy Random Robot")
    plt.bar(x = [0.9, 1.9, 2.9], height = MC['cleaned'], width=0.2, color='silver', edgecolor='black', label = 'MC')
    plt.bar(x = [1.1, 2.1, 3.1], height = mean_QL['cleaned'], width=0.2, color='dimgray', edgecolor='black', label = "Q-learning")
    plt.bar(x = [1.3, 2.3, 3.3], height = mean_sarsa['cleaned'], width=0.2, color='black', edgecolor='black', label = "SARSA")
    plt.xticks(ticks = [1,2,3], labels = ['empty-room', 'empty-house', 'furnished-house'], fontsize='large')
    plt.xlabel("grids", fontsize='x-large')
    plt.ylabel("tiles cleaned (%)", fontsize='x-large')
    plt.yticks(fontsize='large')
    plt.ylim(0,101)
    plt.title("Mean tile percentage cleaned per grid and algorithm", fontsize='xx-large')
    plt.legend(fontsize='x-large')
    plt.savefig('./figure-2b.png')

#%% Comparison assignment 1

if main_results:
    with open("PI_results.json") as file:
        PI = json.load(file)
        #extract the grid we re-used this time as well as best performing hyperparameters for comparison
        PI = PI['PI']['1example-random-house-0.grid']['theta 0.2-gamma 1']
        
    with open("VI_results.json") as file:
        VI = json.load(file)
        VI = VI['VI']['1example-random-house-0.grid']['theta 0.2-gamma 0.5']
        
    PI_efficiency = np.mean(PI['efficiencies'])
    PI_clean = np.mean(PI['cleaned'])
    
    VI_efficiency = np.mean(VI['efficiencies'])
    VI_clean = np.mean(VI['cleaned'])
    
    heights_eff = [PI_efficiency, VI_efficiency, MC['efficiency'][1], 
                   mean_QL['efficiency'][1], mean_sarsa['efficiency'][1]]
    heights_clean = [PI_clean, VI_clean, MC['cleaned'][1],
                     mean_QL['cleaned'][1], mean_sarsa['cleaned'][1]]
    
    #plot efficiencies
    plt.figure(2)
    plt.bar(x = [1,2,3,4,5], height = heights_eff, width=0.6, color='grey', edgecolor='black')
    plt.xticks(ticks = [1,2,3,4,5], labels = ['PI', 'VI', 'MC', 'Q-l', 'SARSA'], fontsize='large')
    plt.xlabel("algorithms", fontsize='x-large')
    plt.ylabel("efficiency (%)", fontsize='x-large')
    plt.yticks(fontsize='large')
    plt.ylim(0,101)
    plt.title("Mean efficiencies per algorithm\non the empty-house grid", fontsize='x-large')
    plt.savefig('./figure-3a.png')
    
    #plot cleaned tile percentage
    plt.figure(3)
    plt.bar(x = [1,2,3,4,5], height = heights_clean, width=0.6, color='grey', edgecolor='black')
    plt.xticks(ticks = [1,2,3,4,5], labels = ['PI', 'VI', 'MC', 'Q-l', 'SARSA'], fontsize='large')
    plt.xlabel("algorithms", fontsize='x-large')
    plt.ylabel("tiles cleaned (%)", fontsize='x-large')
    plt.yticks(fontsize='large')
    plt.ylim(0,101)
    plt.title("Mean percentage of cleaned tiles per algorithm\non the empty-house grid", fontsize='x-large')
    plt.savefig('./figure-3b.png')