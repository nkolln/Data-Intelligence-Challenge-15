import numpy as np
from scipy.spatial import distance_matrix
import math

'''
TO DO (10 June): 
    - Finish Q-learning Main loop
    - Establish terminal state (probably based on cleanliness obtained from environment)
'''

'''
We use Q-dictionary of format:
     {state: {action: Q-value,
              action: Q-value,
              ...},
      state: {action: Q-value,
              ...},
      ...}
'''

ROBOT_SIZE = (33,33)
Q = {}
state = []

def get_robot_checkpoints(center, ROBOT_SIZE):
    '''
    Takes center point of a square robot and returns checkpoints used to test overlap with obstacles
    '''
    x,y = center
    half_width = ROBOT_SIZE[0]//2
    half_height = ROBOT_SIZE[1]//2
    quart_width = half_width//2
    quart_height = half_height//2
    roomba_points = [(x,y),(x+half_width,y),(x-half_width,y),(x,y+half_height),(x,y-half_height),
                     (x+half_width,y+half_height),(x-half_width,y+half_height),(x+half_width,y-half_height),(x-half_width,y-half_height),
                     (x+quart_width,y+quart_height),(x-quart_width,y+quart_height),(x+quart_width,y-quart_height),(x-quart_width,y-quart_height)]
    return roomba_points

def get_robot_surrounding(center, ROBOT_SIZE):
    '''
    get some pixels surrounding the robot to check its surroundings
    '''
    x,y = center
    half_width = ROBOT_SIZE[0]//2
    half_height = ROBOT_SIZE[1]//2
    roomba_surrounding = [(x+half_width+1,y),(x-half_width-1,y),(x,y+half_height+1),(x,y-half_height-1),
                        (x+half_width+1,y+half_height+1),(x-half_width-1,y+half_height+1),
                        (x+half_width+1,y-half_height-1),(x-half_width-1,y-half_height-1)]
    return roomba_surrounding


def reward_func(state, env, future_pos):
    '''
    calculate reward based on the given state and action
    
    :param state: 2d n x 2 numpy array containing the coordinates of each previous step
    :param obs_vic: list of vicinities of obstacles near the future coordinate
    :param action: tuple containing the coordinates of the robot after performing the move
    
    :return: reward for the state-action pair, efficiency so far
    '''
    reward = 0
    #1. obstacle check: future_pos should not overlap an obstacle, preferably
    #check 8 points surrounding the roomba (corners and middle edges), and 5 extra center points to cover corner cases
    roomba_points = get_robot_checkpoints(future_pos, ROBOT_SIZE)
    for rp in roomba_points: #check if roomba points do not intercept obstacles
        if env.is_obstacle(rp):
            reward -= 5000
            
    #check the parameter of the roomba for obstacles, to encourage getting into nooks and crannies
    roomba_surrounding = get_robot_surrounding(future_pos, ROBOT_SIZE)
    for rs in roomba_surrounding:
        if env.is_obstacle(rs):
            reward += 5
    
    edges = np.array(roomba_points[1:5])
    corners = np.array(roomba_points[5:9])
    #2. check it doesn't re-visit tiles: future_pos should not be closer than 1 (robot diameter) to a previous position
    for prev_point in state:
        if any(abs(edges - prev_state)[:,0] <= 1) and any(abs(edges - prev_state)[:,1] <= 1): #25% overlap
            reward -= 5
        if any(abs(corners - prev_state)[:,0] <= 1) and any(abs(corners - prev_state)[:,0] <= 1): #50% overlap
            reward -= 10
        elif all(abs(np.array(prev_point) - future_pos) <= 1): #100% overlap: complete revisit
            reward -= 20
    
    # distances_from_previous = distance_matrix(state, action.reshape(1,2)) #euclidean distances from previous positions
    
    # #How much would the robot overlap with previously cleaned tiles when taking this action?
    # overlap = distances_from_previous < 1
    # overlap_weights = distances_from_previous[overlap]
    # nr_of_overlap = np.sum(overlap)
    # reward -= nr_of_overlap * 10 #discourage overlap, needs to be higher than nooks and crannies reward!
    
    # #To what extent is it perfectly against the border of cleaned areas?
    # perf_previous = distances_from_previous == 1
    # nr_of_perf_prev = np.sum(perf_previous)
    # reward += nr_of_perf_prev * 5 #encourage not leaving strips of uncleaned floor
    
    #3. how much of the room is traversed so far?
    #how_clean = state.shape[0] - np.sum(overlap_weights)
    #reward += how_clean
    #efficiency = how_clean / state.shape[0]
    return reward #, efficiency

            
def select_action(state_Q, policy, epsilon):
    '''
    If policy==True, uses epsilon-greedy policy to select an action
    If policy==False, selects the action with the max Q-value. If more than one action has the
    max Q-value, one is randomly chosen from these actions.
    
    :param state_Q: Q-values for the state in which we need to act
    :param policy: Boolean indicating whether to follow the epsilon-greedy policy or not
    :param epsilon: used in the epsilon-greedy policy, indicates probability of exploration
    '''
    #epsilon-greedy policy
    if policy and np.random.uniform(0, 1) < epsilon:
        best_action = np.random.choice(list(state_Q.keys())) #choose randomly with Q-value weights
    else: #choose action with highest Q-value
        max_actions = [key for key, value in state_Q.items() if value == max(state_Q.values())]
        best_action = np.random.choice(max_actions)
    return best_action

def take_action(current_pos, action, robot_size):
    '''
    Takes the robot's current position, action and size to determine the coordinate it arrives at after performing the action
    '''
    x_move,y_move,dist = action
    robot_width, robot_height = robot_size
    x,y = current_pos
    
    x += x_move*dist*robot_size
    y += y_move*dist*robot_size
    
    return x,y

def robot_epoch(env, alpha=0.6, gamma=0.5, epsilon=0.5):
    global state
    global Q
    #update the state
    current_position = env.robot_location()[0]
    state.append(current_position)
    #add state to Q-dictionary with arbitrary value
    moves = [(0,-1,1),(1,0,1),(0,1,1),(-1,0,1),(-1,-1,1),(1,-1,1),(-1,1,1),(1,1,1), #ADD ROBOT_SIZE
             (0,-1,0.5),(1,0,0.5),(0,1,0.5),(-1,0,0.5),(-1,-1,0.5),(1,-1,0.5),(-1,1,0.5),(1,1,0.5)]
    #add current state to Q-dictionary
    Q[state] = {m:0 for m in moves}
    
    #START Q-LEARNING MAIN LOOP
    for _ in range(200):
        next_state = state.copy()
        next_position = current_position
        terminal = False
        while not terminal:
            #choose an action using the policy and Q-values
            action = select_action(Q[next_state], True, epsilon) 
            #find the position the robot is in after taking the action
            next_position = take_action(next_position, action, ROBOT_SIZE)
            #compute the reward
            reward = reward_func(next_state, env, next_position)
            #update the state
            next_state = next_state.append(next_position)
            #find the best possible next action that could be taken
            next_action = select_action(Q[next_state], False, epsilon)
            
            #update the Q-dictionary
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            
            if 
            
    
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape 
    
    