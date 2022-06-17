import numpy as np
from scipy.spatial import distance_matrix
import math
import pygame
from pygame_env import Environment, StaticObstacle, Robot, MovingHorizontalObstacle, MovingVerticalObstacle, ChargingDock
import random


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


def reward_func(state, env, future_pos, overlap):
    '''
    calculate reward based on the given state and action
    
    :param state: 2d n x 2 numpy array containing the coordinates of each previous step
    :param obs_vic: list of vicinities of obstacles near the future coordinate
    :param action: tuple containing the coordinates of the robot after performing the move
    
    :return: reward for the state-action pair, efficiency so far
    '''
    global ROBOT_SIZE
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
    
    #2. check it doesn't re-visit tiles: future_pos should not be closer than 1 (robot diameter) to a previous position
    edges = np.array(roomba_points[1:5])
    corners = np.array(roomba_points[5:9])
    for prev_point in state:
        if any(abs(edges - prev_point)[:,0] <= 1) and any(abs(edges - prev_point)[:,1] <= 1): #25% overlap
            reward -= 5
            overlap += 0.25
        elif any(abs(corners - prev_point)[:,0] <= 1) and any(abs(corners - prev_point)[:,0] <= 1): #50% overlap
            reward -= 10
            overlap += 0.5
        elif all(abs(np.array(prev_point) - future_pos) <= 1): #100% overlap: complete revisit
            reward -= 20
            overlap += 1
    
    #3. how much of the room is traversed so far?
    total_area = env.display_height * env.display_width
    obstacle_area = sum([obs.size[0]*obs.size[1] for obs in env.obstacles])
    area_to_be_cleaned = total_area - obstacle_area
    cleaned_area = (len(state)+1 - overlap)*(ROBOT_SIZE[0]*ROBOT_SIZE[1])
    cleanliness = cleaned_area / area_to_be_cleaned
    
    return reward, cleanliness, overlap

            
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
    if policy and np.random.uniform(0, 1) < epsilon and (min(list(state_Q.values())) != 0 and max(list(state_Q.values())) != 0):
        if min(list(state_Q.values())) <= 0:
            weights = list(state_Q.values()).copy()
            for i in range(len(weights)):
                weights[i] - min(list(state_Q.values()))
        else:
            weights = list(state_Q.values())
        choice = random.choices(list(range(len(list(state_Q.keys())))), weights=weights) #choose randomly with Q-value weights
        best_action = list(state_Q.keys())[choice]
    else: #choose action with highest Q-value
        max_actions = [key for key, value in state_Q.items() if value == max(state_Q.values())]
        best_action = random.choices(max_actions)
    return best_action[0]

def take_action(current_pos, action, robot_size):
    '''
    Takes the robot's current position, action and size to determine the coordinate it arrives at after performing the action
    '''
    print(action)
    x_move,y_move,dist = action
    robot_width, robot_height = robot_size
    x,y = current_pos
    
    x += x_move*dist*robot_width
    y += y_move*dist*robot_height
    
    return x,y

def robot_epoch(env, alpha=0.6, gamma=0.5, epsilon=0.5):
    global state
    global Q
    global ROBOT_SIZE
    #update the state
    current_position = env.robot_location()[0]
    state.append(current_position)
    #add state to Q-dictionary with arbitrary value
    moves = [(1,0,1),(1,-1,1),(0,-1,1),(-1,-1,1),(-1,0,1),(-1,1,1),(0,1,1),(1,1,1),
             (1,0,0.5),(1,-1,0.5),(0,-1,0.5),(-1,-1,0.5),(-1,0,0.5),(-1,1,0.5),(0,1,0.5),(1,1,0.5)]
    #add current state to Q-dictionary
    if tuple(state) not in Q.keys(): #initialize if not yet in dict
        Q[tuple(state)] = {m:0 for m in moves}
    
    #START Q-LEARNING MAIN LOOP
    for _ in range(200): #200 episodes
        next_state = state.copy()
        if tuple(next_state) not in Q.keys(): #initialize if not yet in dict
            Q[tuple(next_state)] = {m:0 for m in moves}
        next_position = current_position
        overlap = 0
        terminal = False
        t = 0
        while not terminal and t < 100:
            t += 1
            #choose an action using the policy and Q-values
            action = select_action(Q[tuple(next_state)], True, epsilon)
            #find the position the robot is in after taking the action
            next_position = take_action(next_position, action, ROBOT_SIZE)
            #compute the reward
            reward, cleaned, overlap = reward_func(next_state, env, next_position, overlap)
            #update the state
            next_state.append(next_position)
            #find the best possible next action that could be taken
            next_action = select_action(Q[tuple(next_state)], False, epsilon)
            
            #update the Q-dictionary
            Q[tuple(state)][action] += alpha * (reward + gamma * Q[tuple(next_state)][next_action] - Q[tuple(state)][action])
            
            if cleaned >= 1:
                terminal = True
                
    #find best move
    max_actions = [key for key, value in Q[tuple(state)].items() if value == max(Q[tuple(state)].values())]
    
    if len(max_actions) == 1:
        max_action = max_actions[0]
    else:
        max_action = np.random.choice(max_actions)
    
    bool_action = [0,0,0,0,0,0,0,0]
    bool_action[moves.index(max_action)%8] = 1
    
    if max_action[2] == 1:
        env.discrete_step(bool_action)
        
    elif max_action[2] == 0.5:
        env.robot.speed /= 2
        env.discrete_step(bool_action)
        env.robot.speed *= 2
        

if __name__=='__main__':
    screen = pygame.display.set_mode((800, 600))

    all_sprites = pygame.sprite.Group()
    collision_sprites = pygame.sprite.Group()

    # obstacle setup, random generation will be implemented
    obs1 = StaticObstacle(pos=(100, 500), size=(100, 50), groups=[all_sprites, collision_sprites])
    obs2 = StaticObstacle((400, 400), (100, 200), [all_sprites, collision_sprites])
    obs3 = StaticObstacle((200, 200), (200, 100), [all_sprites, collision_sprites])
    obs4 = StaticObstacle((300, 100), (200, 300), [all_sprites, collision_sprites])
    obs5 = StaticObstacle((1, 1), (200, 100), [all_sprites, collision_sprites])
    obs6 = StaticObstacle((700, 1), (50, 400), [all_sprites, collision_sprites])
    obs7 = MovingHorizontalObstacle((0, 300), (50, 50), [all_sprites, collision_sprites], max_left=0, max_right=300, speed=5)
    obs7 = MovingVerticalObstacle((0, 300), (50, 50), [all_sprites, collision_sprites], max_up=0, max_down=300, speed=5)

    charging_dock = ChargingDock((25, 554), (50,50), [all_sprites])

    robot = Robot(all_sprites, collision_sprites, screen, 0.1, 5, 20)
    game = Environment(robot, [obs1, obs2, obs3, obs4, obs5, obs6, obs7], charging_dock, all_sprites, collision_sprites, screen)
    
    clean_percent = 0
    battery_percent = 100
    while clean_percent < 100 and battery_percent > 0:
        robot_epoch(game)
        clean_percent = game.calc_clean_percentage()
        battery_percent = game.robot.battery_percentage
        
    print(clean_percent)
    print(game.calculate_efficiency())
        