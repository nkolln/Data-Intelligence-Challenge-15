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

    
    #Focmula to calculate the variant Q
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
                reward_tot, _, _, _,_ = self.env.cont_step(a, b, False)
                l_z.append(reward_tot)
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
                #for j in range(size):
                for c,d in zip(l_x_q,l_y_q):
                    reward_curr, _, _, _,_ = self.env.cont_step(a, b, False)
                    reward, _, _,_, _ = self.env.cont_step(c,d, False)
                    reward_tot = self.update_Q(reward_curr,reward)
                    if reward_tot < 0:
                        reward, _, _,_, _ = self.env.cont_step(c,d, False) 
                        reward_tot = self.update_Q(reward_tot,reward)
                    if reward_tot>reward_old:
                        reward_old=reward_tot
                    self.env.revert_copy()
                l_z.append(reward_old)
                self.env.revert_copy()

        #Modes Beneath were made for testing before the environment was created
        elif self.mode==0:
            l_z = [np.random.uniform(self.rv1+60,self.rv2,1) if ((a>0.5)and(b>0.5))or((a>1)and(b>1)) else np.random.uniform(self.rv1,self.rv2-60,1) for a,b in zip(l_x,l_y)]
        elif self.mode==1:
            l_z = [np.random.uniform(self.rv1+60,self.rv2,1) if ((a>0)and(b>0))or((a>1)and(b>1)) else np.random.uniform(self.rv1,self.rv2-60,1) for a,b in zip(l_x,l_y)]
        elif self.mode==2:
            l_z = [np.random.uniform(self.rv1+60,self.rv2,1) if ((a>0)and(b>0))or((a>0)and(b<0)) else np.random.uniform(self.rv1,self.rv2-60,1) for a,b in zip(l_x,l_y)]
        #l_ratio = [(self.sz/(a**2+ b**2))**0.5 for a,b in zip(l_x,l_y)]
        #print(l_z)
        geometry = [Point(a +self.x, b+self.y) for a, b in zip(l_x, l_y)]
        gpd_data = gpd.GeoDataFrame(l_z,geometry=geometry)
        return(gpd_data)


    
    def smallest_points(self):
        gpd_data = self.generate_data()
        lst_return = []
        for a,b in zip(gpd_data['geometry'].x,gpd_data['geometry'].y):
            gpd_dist = gpd_data.distance(Point(a,b))
            lst_return.append(gpd_dist.sort_values()[:self.neighbors].index.to_numpy())
        return(lst_return,gpd_data)

    def average_values(self):
        lst_sim,gpd_data = self.smallest_points()
        lst_return = []
        for row in lst_sim:
            lst_return.append(gpd_data.iloc[row][self.column].mean())
        gpd_data[self.col_avg] = lst_return
        return(gpd_data)

    def scale_circle(self):
        gpd_data = self.average_values()
        
        lst_norm_scale = []
        for a,b,r in zip(gpd_data['geometry'].x,gpd_data['geometry'].y,(gpd_data[self.col_avg])):
            lst_norm_scale.append(Point((a-self.x)*r,((b-self.y)*r)))
            
        gpd_data['geometry_old'] = gpd_data['geometry']
        gpd_data['geometry'] = lst_norm_scale
        
        point,_ = self.get_max_val(gpd_data)
        gpd_dist = gpd_data.distance(point)
        lst_ids = gpd_dist.sort_values()[:int(self.neighbors)].index.to_numpy()
        gpd_data_cut = gpd_data.iloc[lst_ids]

        if self.vb:#!=True:
            #self.visualize_data_in_circle_avg()
            gpd_data.plot(column = self.col_avg,cmap='hot')
            plt.plot(np.unique(gpd_data_cut['geometry'].x), np.poly1d(np.polyfit(gpd_data_cut['geometry'].x, gpd_data_cut['geometry'].y, 1))(np.unique(gpd_data_cut['geometry'].x)))
            #self.vb = False

        return(gpd_data,gpd_data_cut)

    def generate_vector(self):
        gpd_data,gpd_data_cut = self.scale_circle()
        #m,b = np.polyfit(gpd_data_cut['geometry'].x, gpd_data_cut['geometry'].y, 1)
        coords = self.get_max_val1(gpd_data)
        #coord_best = gpd_data.iloc[idx]['geometry_old']
        #x_c = coord_best.x-self.x
        #y_c = -x_c/m
        #calculate ratio
        #r = (self.sz/(x_c**2+ y_c**2))**0.5
        #vec = (x_c*r,y_c*r)
        return(coords[0],coords[1])
        return(vec[0],vec[1])

    def get_max_val(self,data):
        gpd_data = data
        max_id = gpd_data['average'].idxmax()
        ret = gpd_data.iloc[max_id]['geometry']
        return(ret,max_id)

    def get_max_val1(self,data):
        gpd_data = data
        #max_id = gpd_data['average'].nlargest(2).index
        max_id = gpd_data['average'].idxmax()
        ret0 = gpd_data.iloc[max_id]['geometry_old']
        gpd_dist = gpd_data.distance(Point(ret0.x,ret0.y))
        ids = gpd_dist.nlargest(3).index
        ids = gpd_data.iloc[ids]['average'].nlargest(2).index[1]
        #ids = gpd_data.iloc[ids]['average'].idxmax()
        ret1 = gpd_data.iloc[ids]['geometry_old']
        a,b = ret0.x+ret1.x, ret0.y+ret1.y
        r = (self.sz/((a)**2+ (b)**2))**0.5
        a,b = a*r,b*r
        return(a,b)


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