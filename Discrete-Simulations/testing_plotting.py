import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st

bots = ['greedy-random-robot']
grids = ['empty.grid', 'wall-furniture.grid',
         'example-random-house-0.grid', 'rooms-with-furniture.grid']

df = pd.read_csv('results.csv')
grouped_by_bots = df.groupby(df.bot)
battery_runs = df.groupby(df.battery).get_group(True)
battery_runs_grouped_by_bots = battery_runs.groupby(df.bot)

#%% average efficiency per bot
#calculate mean efficiencies per bot
efficiencies = [np.mean(grouped_by_bots.get_group(bot)['efficiency']) for bot in bots]
#calculate standard error per bot (needed for confidence intervals)
sems = [st.sem(grouped_by_bots.get_group(bot)['efficiency']) for bot in bots]
#calculate 95% confidence intervals
CIs = np.array([st.norm.interval(alpha=.95, loc=efficiencies[i], scale=sems[i])[0] for i in range(len(bots))])
CIs = abs(CIs - np.array(efficiencies))

plt.figure(0)
plt.bar(x=range(len(bots)), height=efficiencies, yerr=CIs, capsize=20)
plt.xticks(ticks=range(len(bots)), labels = bots)
plt.ylabel('efficiency (%)')
plt.title("Mean efficiency over all runs for each algorithm")

#%% average runtime per bot
#calculate mean runtime per bot
runtimes = [np.mean(grouped_by_bots.get_group(bot)['runtime']) for bot in bots]
#calculate standard error per bot (needed for confidence intervals)
sems = [st.sem(grouped_by_bots.get_group(bot)['runtime']) for bot in bots]
#calculate 95% confidence intervals
CIs = np.array([st.norm.interval(alpha=.95, loc=runtimes[i], scale=sems[i])[0] for i in range(len(bots))])
CIs = abs(CIs - np.array(runtimes))

plt.figure(0)
plt.bar(x=range(len(bots)), height=runtimes, yerr=CIs, capsize=20)
plt.xticks(ticks=range(len(bots)), labels = bots)
plt.ylabel('runtime (s)')
plt.title("Mean runtime over all runs for each algorithm")

#%% average cleanliness per bot
#calculate mean tiles cleaned per bot
cleanliness = [np.mean(battery_runs_grouped_by_bots.get_group(bot)['cleaned']) for bot in bots]
#calculate standard error per bot (needed for confidence intervals)
sems = [st.sem(battery_runs_grouped_by_bots.get_group(bot)['cleaned']) for bot in bots]
#calculate 95% confidence intervals
CIs = np.array([st.norm.interval(alpha=.95, loc=cleanliness[i], scale=sems[i])[0] for i in range(len(bots))])
CIs = abs(CIs - np.array(cleanliness))

plt.figure(0)
plt.bar(x=range(len(bots)), height=cleanliness, yerr=CIs, capsize=20)
plt.xticks(ticks=range(len(bots)), labels = bots)
plt.ylabel('tiles cleaned (%)')
plt.title("Mean cleanliness over all battery-runs for each algorithm")