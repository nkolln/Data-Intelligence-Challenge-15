import numpy as np
from scipy.spatial import distance_matrix

def reward_func(state, obs, action):
    '''
    calculate reward based on the given state and action
    
    :param state: 2d n x 2 numpy array containing the coordinates of each previous step
    :param obs: 2d n x 2 numpy array of obstacle outline points
    :param action: tuple containing the coordinates of the robot after performing the move
    '''
    reward = 0
    #1. obstacle check: action should not be closer than 0.5 (robot radius) to an obstacle
    distances_from_obstacles = distance_matrix(obs, action.reshape(1,2)) #calculate euclidean distances from obstacles
    
    #how many obstacles would the robot collide with when taking this action?
    collisions = distances_from_obstacles < 0.5
    nr_of_collisions = np.sum(collisions)
    reward -= nr_of_collisions * 5000 #colliding should be absolutely impossible!!
    
    #how many obstacles is it flush against when taking this action?
    perf_obstacles = distances_from_obstacles == 0.5
    nr_of_perf_obstacles = np.sum(perf_obstacles)
    reward += nr_of_perf_obstacles * 5 #encourage getting into nooks and crannies

    #2. check it doesn't re-visit tiles: action should not be closer than 1 (robot diameter) to a previous position
    distances_from_previous = distance_matrix(state, action.reshape(1,2)) #euclidean distances from previous positions
    
    #How much would the robot overlap with previously cleaned tiles when taking this action?
    overlap = distances_from_previous < 1
    overlap_weights = distances_from_previous[overlap]
    nr_of_overlap = np.sum(overlap)
    reward -= nr_of_overlap * 10 #discourage overlap, needs to be higher than nooks and crannies reward!
    
    #To what extent is it perfectly against the border of cleaned areas?
    perf_previous = distances_from_previous == 1
    nr_of_perf_prev = np.sum(perf_previous)
    reward += nr_of_perf_prev * 5 #encourage not leaving strips of uncleaned floor
    
    #3. how much of the room is traversed so far?
    how_clean = state.shape[0] - np.sum(overlap_weights)
    #reward += how_clean
    efficiency = how_clean / state.shape[0]
        
    return reward, efficiency
            

def robot_epoch(robot, alpha=0.6, gamma=0.5, epsilon=0.5):
    moves = ['n','e','s','w','nw','ne','sw','se']
    moves_actual = [(0,-1),(1,0),(0,1),(-1,0),(-1,-1),(1,-1),(-1,1),(1,1)]
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape 
    
    