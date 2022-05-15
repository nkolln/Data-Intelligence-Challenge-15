import numpy as np
import pandas as pd
import gym

Q = np.zeros((10, 4))
dct_map = {}
print(Q)
count = 0
for i in range(3):
    for j in range(4):
        dct_map.update({(i,j):count})
        count+=1
print(dct_map)
print(dct_map[(0,0)])
