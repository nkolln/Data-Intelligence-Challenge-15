import numpy as np
from scipy.spatial import distance_matrix

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

Q = {}
state = []

def reward_func(state, obs_vic, action):
    '''
    calculate reward based on the given state and action
    
    :param state: 2d n x 2 numpy array containing the coordinates of each previous step
    :param obs_vic: list of vicinities of obstacles near the action coordinate
    :param action: tuple containing the coordinates of the robot after performing the move
    
    :return: reward for the state-action pair, efficiency so far
    '''
    reward = 0
    #1. obstacle check: action should not be closer than 0.5 (robot radius) to an obstacle
    for o in obs_vic:
        if o < 0.5:         #if the robot would bump into the obstacle when making this move
            reward -= 5000  #discourage! No bumping!
        if o == 0.5:        #if the robot is flush against the obstacle
            reward += 5     #encourage getting nooks and crannies
    
    # distances_from_obstacles = distance_matrix(obs, action.reshape(1,2)) #calculate euclidean distances from obstacles
    
    # #how many obstacles would the robot collide with when taking this action?
    # collisions = distances_from_obstacles < 0.5
    # nr_of_collisions = np.sum(collisions)
    # reward -= nr_of_collisions * 5000 #colliding should be absolutely impossible!!
    
    # #how many obstacles is it flush against when taking this action?
    # perf_obstacles = distances_from_obstacles == 0.5
    # nr_of_perf_obstacles = np.sum(perf_obstacles)
    # reward += nr_of_perf_obstacles * 5 #encourage getting into nooks and crannies

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
    return str(best_action)

def robot_epoch(robot, alpha=0.6, gamma=0.5, epsilon=0.5):
    global state
    global Q
    #update the state
    state.append((robot.x, robot.y))
    #add state to Q-dictionary with arbitrary value
    moves = [(0,-1,1),(1,0,1),(0,1,1),(-1,0,1),(-1,-1,1),(1,-1,1),(-1,1,1),(1,1,1),
             (0,-1,0.5),(1,0,0.5),(0,1,0.5),(-1,0,0.5),(-1,-1,0.5),(1,-1,0.5),(-1,1,0.5),(1,1,0.5)]
    #add current state to Q-dictionary
    Q[state] = {m:0 for m in moves}
    
    #START Q-LEARNING MAIN LOOP
    for _ in range(200):
        current_state = state.copy()
        for t in range(100):
            action = select_action(Q[current_state], True, epsilon)
        
    
    
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape 
    
    