import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString

class direction_control():
    def __init__(self,coord,size_rand,step_size,mode=0,vis_bool=False,col_avg='average',neighbors=5,range_coord=[-1,1],range_val=[0,100],column=0,data_present = False):
        self.coord=coord
        self.x = coord[0]
        self.y = coord[1]
        self.size_rand=size_rand
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

    def generate_data_random(self):
        l_x,l_y = np.random.uniform(self.rc1,self.rc2,self.size_rand),np.random.uniform(self.rc1,self.rc2,self.size_rand)
        l_z = np.random.uniform(self.rv1,self.rv2,self.size_rand)
        l_ratio = [(1/(a**2+ b**2))**0.5 for a,b in zip(l_x,l_y)]
        geometry = [Point(a*r, b*r) for a, b,r in zip(l_x, l_y,l_ratio)]
        gpd_data = gpd.GeoDataFrame(l_z,geometry=geometry)
        return(gpd_data)

    def generate_data(self):
        l_x,l_y = np.random.uniform(self.rc1,self.rc2,self.size_rand),np.random.uniform(self.rc1,self.rc2,self.size_rand)
        if self.mode==-1:
            #normal generation mode
        elif self.mode==0:
            l_z = [np.random.uniform(self.rv1+60,self.rv2,1) if ((a>0.5)and(b>0.5))or((a>1)and(b>1)) else np.random.uniform(self.rv1,self.rv2-60,1) for a,b in zip(l_x,l_y)]
        elif self.mode==1:
            l_z = [np.random.uniform(self.rv1+60,self.rv2,1) if ((a>0)and(b>0))or((a>1)and(b>1)) else np.random.uniform(self.rv1,self.rv2-60,1) for a,b in zip(l_x,l_y)]
        elif self.mode==2:
            l_z = [np.random.uniform(self.rv1+60,self.rv2,1) if ((a>0)and(b>0))or((a>0)and(b<0)) else np.random.uniform(self.rv1,self.rv2-60,1) for a,b in zip(l_x,l_y)]
        l_ratio = [(self.sz/(a**2+ b**2))**0.5 for a,b in zip(l_x,l_y)]
        geometry = [Point(a*r +self.x, b*r+self.y) for a, b,r in zip(l_x, l_y,l_ratio)]
        gpd_data = gpd.GeoDataFrame(l_z,geometry=geometry)
        #gpd_data['x'] = gpd_data['geometry'].x
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
        m,b = np.polyfit(gpd_data_cut['geometry'].x, gpd_data_cut['geometry'].y, 1)
        _,idx = self.get_max_val(gpd_data)
        coord_best = gpd_data.iloc[idx]['geometry_old']
        x_c = coord_best.x-self.x
        y_c = -x_c/m
        #calculate ratio
        r = (self.sz/(x_c**2+ y_c**2))**0.5
        vec = (x_c*r,y_c*r)

        if self.vb:
            #plt.plot([0,x_c*100],[0,y_c*100])
            gpd_data_copy = gpd_data.copy()

            #Vis 1
            gpd_data_copy['geometry'] = gpd_data_copy['geometry_old']
            self.visualize_data_in_circle_avg(gpd_data_copy)
            plt.plot([0+self.x,vec[0]+self.x],[0+self.y,vec[1]+self.y])
            
            #Vis 2
            self.data_present = True
            max_coord, _ = self.get_max_val(gpd_data_copy)
            plt.plot([0+self.x,max_coord.x],[0+self.y,max_coord.y])
            self.vb = False
        return(vec)



    def get_max_val(self,data):
        gpd_data = data
        max_id = gpd_data['average'].idxmax()
        ret = gpd_data.iloc[max_id]['geometry']
        return(ret,max_id)


    def max_vector(self):
        gpd_data = self.average_values()
        max_id = gpd_data['average'].idxmax()
        vec_x, vec_y = gpd_data.iloc[max_id]['geometry'].x-self.x,gpd_data.iloc[max_id]['geometry'].y-self.y
        print(vec_x,vec_y)

        if self.vb:
            self.visualize_data_in_circle_avg()
            self.vb = False

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



"""import operator

possible_tiles = {(1,1):1,(2,0):1,(1,0):1,(0,1):3,(0,2):1,(-1,1):0,(-1,0):2,(0,-1):1} #robot.possible_tiles_after_move()
#possible_tiles_good = {k:v for k,v in possible_tiles.items() if float(v) >= 1}
possible_tiles_one = {k:v for k,v in possible_tiles.items() if abs(k[0])+abs(k[1])==1}
print(possible_tiles_one)
#[sum(abs(k[0])+sum(abs(k[1]))) for k in possible_tiles.keys()]
farthest_step_vision = max([abs(k[0])+abs(k[1]) for k in possible_tiles.keys()])
print(farthest_step_vision)
#cap at 2 for now
if farthest_step_vision > 2:
    farthest_step_vision = 2
#farthest_step_vision = max([abs(k[0])+abs(k[1]) for k in possible_tiles.keys()])
for i in range(1,farthest_step_vision):
    for key, val in possible_tiles_one.items():
        #dct_val = {}
        lst_val = []
        possible_tiles_iter = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) + abs(move[1]) == i + 1}
        for key_it,val_it in possible_tiles_iter.items():
            key_new = tuple(map(operator.sub, key_it, key))
            if(abs(key_new[0])+ abs(key_new[1]) == i):
                lst_val.append(val_it)
                print(key_new, key_it, key)
        print(lst_val)
        if lst_val:
            if len(lst_val)!=1:
                max_val = max(lst_val)
            else:
                max_val = lst_val[0]
        else:
            max_val = 0
        
        if max_val <1:
            max_val = 0
        possible_tiles_one.update({key:max_val+val})
        print(possible_tiles_one)

print(max(possible_tiles_one, key=possible_tiles_one.get))
print(list(possible_tiles_one.keys())[list(possible_tiles_one.values()).index(1.0)])

#-----------------------------------
possible_tiles = {(1, -1): -2, (1, 0): 1, (0, 1): 1, (-1, 0): -1}
print({k:v for k,v in possible_tiles.items() if float(v) < 1})
print(list(possible_tiles.keys()))
print(list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)])

possible_tiles = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) < 2 and abs(move[1]) < 2}
print(possible_tiles)"""