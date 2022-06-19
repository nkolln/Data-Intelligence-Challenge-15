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

    #First generate all the possible random moves
    #Then sample surrounding area in an even 8 area split

    def update_Q(self,reward,next_reward):
        target = reward + self.gamma * next_reward
        r_curr = self.alpha * target
        return(r_curr)

    def generate_data(self):
        l_x,l_y = np.random.uniform(self.rc1,self.rc2,self.size_rand),np.random.uniform(self.rc1,self.rc2,self.size_rand)
        if self.mode==-1:
            #normal generation mode
            #l_x_q,l_y_q = [1,math.sqrt(0.5),0,-math.sqrt(0.5),-1,-math.sqrt(0.5),0,math.sqrt(0.5)],[0,math.sqrt(0.5),1,math.sqrt(0.5),0,-math.sqrt(0.5),-1,-math.sqrt(0.5)]
            l_x_q,l_y_q = [1,0,-1,0],[0,1,0,-1]
            l_z = []
            for i,(a,b) in enumerate(zip(l_x,l_y)):
                scale = (self.sz/(a**2+ b**2))**0.5
                l_x[i],l_y[i] = a*scale,b*scale
                a,b = l_x[i],l_y[i]
                reward_old = 0
                size = self.further_step#np.random.randint(0,self.further_step,1)[0]
                #for j in range(size):
                for c,d in zip(l_x_q,l_y_q):
                    reward_curr, _, _, _ = self.env.cont_step(a, b, False)
                    reward, _, _, _ = self.env.cont_step(c,d, False)
                    print(f'{reward_curr}  {reward}')
                    reward_tot = self.update_Q(reward_curr,reward)
                    print(reward_tot)
                    if reward_tot>reward_old:
                        reward_old=reward_tot
                    self.env.revert_copy()
                """l_x_temp, l_y_temp = np.random.uniform(self.rc1,self.rc2,size),np.random.uniform(self.rc1,self.rc2,size)
                for i,(a,b) in enumerate(zip(l_x_temp,l_y_temp)):
                    #a,b = np.random.uniform(self.rc1,self.rc2,1),np.random.uniform(self.rc1,self.rc2,self.size_rand)
                    scale = (self.sz/(a**2+ b**2))**0.5
                    a,b = a*scale,b*scale
                    reward, done, score, efficiency = self.env.cont_step(a, b, False)
                    reward_tot = 
                    #reward_tot = reward + reward_tot
                    #alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])"""
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
        print(lst_sim,gpd_data)
        print('-'*100)
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
        lst_ids = gpd_dist.sort_values()[:int(self.neighbors/2)].index.to_numpy()
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
        _,idx = self.get_max_val(gpd_data)
        coord_best = gpd_data.iloc[idx]['geometry_old']
        #x_c = coord_best.x-self.x
        #y_c = -x_c/m
        #calculate ratio
        #r = (self.sz/(x_c**2+ y_c**2))**0.5
        #vec = (x_c*r,y_c*r)

        return(coord_best.x-self.x,coord_best.y-self.y)
        return(vec[0],vec[1])


    def get_max_val(self,data):
        gpd_data = data
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