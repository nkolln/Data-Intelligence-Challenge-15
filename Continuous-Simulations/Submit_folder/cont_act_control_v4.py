import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from pygame_env import Environment, StaticObstacle, Robot
import math

class direction_control():
    def __init__(self,environment,alpha = .5, gamma = 0.5,coord=(0,0),further_step=0,size_rand=300,step_size=2,mode=0,vis_bool=False, col_avg='average',neighbors=5,range_coord=[-1,1],range_val=[0,100],column=0,data_present = False):
        self.coord=coord
        self.gamma = gamma
        self.alpha=alpha
        self.x = coord[0]
        self.y = coord[1]
        self.size_rand=size_rand
        self.further_step=further_step
        self.env = environment
        self.mode=mode
        self.sz = step_size
        self.vb = vis_bool
        self.col_avg = col_avg
        self.neighbors = neighbors
        self.rc1=range_coord[0]
        self.rc2=range_coord[1]
        self.rv1=range_val[0]
        self.rv2=range_val[1]
        self.column = column
        self.data_present = data_present

    #Formula to calculate the variant Q
    def update_Q(self,reward,next_reward):
        target = reward + self.gamma * next_reward
        r_curr = self.alpha * target
        return(r_curr)

    #First generate all the possible random moves
    #Then sample surrounding area in an even 8 area split
    def generate_data(self):
        #Generates a list of movements
        l_x,l_y = np.random.uniform(self.rc1,self.rc2,self.size_rand),np.random.uniform(self.rc1,self.rc2,self.size_rand)
        #List of 4 moves to sample around initial move
        l_x_q,l_y_q = [1,0,-1,0],[0,1,0,-1]
        #Greedy robot mode
        if self.mode == -2:
            #lst to store scores
            l_z = []
            for i,(a,b) in enumerate(zip(l_x,l_y)):
                #Calculates the scale needed to convert the distance as the same distance
                scale = (self.sz/(a**2+ b**2))**0.5
                #Updates array with scaled value
                l_x[i],l_y[i] = a*scale,b*scale
                a,b = l_x[i],l_y[i]
                #finds nearby reward
                reward_tot, _, _, _,_ = self.env.cont_step(a, b, False)
                #saves reward
                l_z.append(reward_tot)
                #Reverts copy back to original position
                self.env.revert_copy()

        #Q Learning Mode
        elif self.mode==-1:
            #normal generation mode
            l_z = []
            for i,(a,b) in enumerate(zip(l_x,l_y)):
                #Calculates the scale needed to convert the distance as the same distance
                scale = (self.sz/(a**2+ b**2))**0.5
                l_x[i],l_y[i] = a*scale,b*scale
                a,b = l_x[i],l_y[i]
                reward_old = 0
                #cycles through the four surrounding moves
                for c,d in zip(l_x_q,l_y_q):
                    #This must repeat the first ttwo steps for all cases unfortunately. This is due to not being able to set the system to a specific coordinate
                    #Instead, I could reset it to the original postion, but would have to repeat the first step to get the score again and have the system move appropiately
                    reward_curr, _, _, _,_ = self.env.cont_step(a, b, False)
                    reward, _, _,_, _ = self.env.cont_step(c,d, False)
                    #Calculate Q value
                    reward_tot = self.update_Q(reward_curr,reward)
                    if reward_tot < 0:
                        #Extension mode, if it has no good values nearby, take another step to see if anything changes
                        reward, _, _,_, _ = self.env.cont_step(c,d, False) 
                        reward_tot = self.update_Q(reward_tot,reward)
                    #Saves the best one
                    if reward_tot>reward_old:
                        reward_old=reward_tot
                    #Reverts copy
                    self.env.revert_copy()
                #Stores the best one
                l_z.append(reward_old)
                self.env.revert_copy()
                
        #Format for Geometry for geopandas
        geometry = [Point(a +self.x, b+self.y) for a, b in zip(l_x, l_y)]
        gpd_data = gpd.GeoDataFrame(l_z,geometry=geometry)
        return(gpd_data)


    #Function to find the closest points to eachother
    def closest_points(self):
        #Gets the generated Data
        gpd_data = self.generate_data()
        lst_return = []
        #For each point, find the distance to every other point
        for a,b in zip(gpd_data['geometry'].x,gpd_data['geometry'].y):
            gpd_dist = gpd_data.distance(Point(a,b))
            #Stores the closest neighbors amount
            lst_return.append(gpd_dist.sort_values()[:self.neighbors].index.to_numpy())
        return(lst_return,gpd_data)

    #Averages the closest neighbor amount of points
    def average_values(self):
        #Gets the closet point and data
        lst_sim,gpd_data = self.closest_points()
        lst_return = []
        #Gets the average values for the list of ids
        for row in lst_sim:
            lst_return.append(gpd_data.iloc[row][self.column].mean())
        gpd_data[self.col_avg] = lst_return
        return(gpd_data)

    #Calculates the final vector
    def generate_vector(self):
        #Gets data
        gpd_data = self.average_values()
        #m,b = np.polyfit(gpd_data_cut['geometry'].x, gpd_data_cut['geometry'].y, 1)
        #Gets the id of the largest point
        _,idx = self.get_max_val(gpd_data)
        #stores the coordinate of largest point
        coord_best = gpd_data.iloc[idx]['geometry']

        return(coord_best.x-self.x,coord_best.y-self.y)

    #Calculates the maximum vector
    def get_max_val(self,data):
        gpd_data = data
        #gets the max id
        max_id = gpd_data['average'].idxmax()
        ret = gpd_data.iloc[max_id]['geometry']
        return(ret,max_id)


    def visualize_data_in_circle_avg(self,data = None):
        if  self.data_present:
            gpd_data = data
            self.data_present = False
        else:
            gpd_data = self.average_values()
        gpd_data.plot(column = self.col_avg,cmap='hot')
        return(gpd_data)

    def visualize_data_in_circle(self):
        gpd_data = self.generate_data()
        gpd_data.plot(column = self.column,cmap='hot')
        return(gpd_data)