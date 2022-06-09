import numpy as np
import operator
import pandas as pd

def list_creator(shape_w, current_world):
    valid_states = []
    lst_end = []
    lst_goal = []
    lst_taboo = []
    lst_wall = []
    all_states = []
    lst_clean = []
    lst_dirty = []
    dct_lsts = {-2: lst_taboo, -1: lst_wall, 0: lst_clean, 1: lst_dirty, 2: lst_goal, 3: lst_end}
    # current_world = pd.DataFrame(current_world).T
    # Categorizes every state
    count = 0
    dct_map = {}
    for x in range(shape_w[0]):
        for y in range(shape_w[1]):
            cur_val = current_world[x][y]
            for key, lst in dct_lsts.items():
                # for num,lst in zip(lst_vals,lst_lsts):
                if cur_val == key:
                    lst.append((x, y))
                if cur_val < -2:
                    lst_clean.append((x,y))
            dct_map.update({(x,y):count})
            count+=1
    all_states.extend(lst_clean)
    all_states.extend(lst_dirty)
    all_states.extend(lst_goal)
    valid_states.extend(all_states)
    all_states.extend(lst_end)

    return valid_states, lst_taboo, lst_wall, lst_clean, lst_dirty, lst_goal, lst_end

def reward_system(valid_states, moves_actual, moves, current_world, lst_clean, lst_taboo, lst_wall):
    rewards = np.copy(current_world)
    actions = {}
    #Stores all the possible movements for valid states
    #Added logic to give scores based on location to other clean tiles and walls
    for coord in valid_states:
        lst_dir = [];lst_neighbor=[]
        count_walls = 0; count_neighbor = 0
        for i in range(4):
            dir = moves_actual[i]
            new_loc = tuple(map(operator.add, coord, dir))
            if current_world[new_loc] == 3:
                rewards[new_loc]= -50
            if new_loc in valid_states:#((new_loc not in lst_taboo) and (new_loc not in end_states)):# and (new_loc[0] in range(bounds[0][0],bounds[0][1]))and(new_loc[1] in range(bounds[1][0],bounds[1][1]))):
                lst_dir.append(moves[i])
                if new_loc in lst_clean and current_world[coord] == 1: #Reward for nearby clean tiles
                    rewards[coord] += 2
            elif current_world[coord]==1:
                if new_loc in lst_wall: #Rewards for being nearby to boundaries of map
                    rewards[coord] += 2
                if new_loc in lst_taboo: #Rewards for nearby boundaries
                    rewards[coord] += 1
                if new_loc in lst_clean or new_loc in lst_taboo or new_loc in lst_wall:
                    count_neighbor += 1
            #elif current_world[coord]==1:
                if new_loc in lst_taboo: #Logic to find if there is a doorway
                    count_walls+=1
                    lst_neighbor.append(dir)
        if count_walls == 2 and tuple(map(operator.add, lst_neighbor[0], lst_neighbor[1]))==(0,0): #If there are two walls next to eachother than treat as door
            rewards[coord] -= 1
            # print('-'*20)
        #If it has no dirty tiles around, prioritize
        if count_neighbor == 4:
            rewards[coord] +=20
        if len(lst_dir)>0:
            actions.update({coord:lst_dir})
    
    return actions, rewards

def initialize_Q(actions):
    Q ={}
    for coord,lis in actions.items():
        rew_dic = {}
        for i in lis:
            rew_dic[i] = 0
        Q[coord] = rew_dic
    return Q

def select_action(Q, state, policy, epsilon):
    #epsilon = 0.5
    if policy and np.random.uniform(0, 1) < epsilon:
        best_action = np.random.choice(list(Q[state].keys()))
    else:
        max_actions = [key for key, value in Q[state].items() if value == max(Q[state].values())]
        best_action = np.random.choice(max_actions)
    return str(best_action)

def take_action(action, rewards,moves,moves_actual, current_pos, visited, lst_dirty):
    done = False
    next_state = tuple(map(operator.add, moves_actual[moves.index(action)], current_pos))
    reward = rewards[next_state]
    difference = list(set(lst_dirty) - set(visited))
    if len(difference) == 0:
        done = True
    return next_state, reward, done

def robot_epoch(robot, alpha=0.6, gamma=0.5, epsilon=0.5):
    #rewards = [-1,1,-2] #reward for landing on clean, dirty or obstacle, respectively
    moves = ['n','e','s','w','nw', 'ne', 'sw', 'se']
    moves_actual = [(0,-1),(1,0),(0,1),(-1,0),(-1,-1),(1,-1),(-1,1),(1,1)]
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape 

    valid_states, lst_taboo, lst_wall, lst_clean, lst_dirty, lst_goal, lst_end = list_creator(shape_w, current_world)
    actions, rewards = reward_system(valid_states, moves_actual, moves, current_world, lst_clean, lst_taboo, lst_wall)
    
    #alpha = 0.6; gamma= 0.5
    Q = initialize_Q(actions)
    
    # Q-LEARNING MAIN LOOP

    #for each episode
    for i in range(200):
        #start with current position
        state = current_pos
        visited = []
        #for each step of episode
        for t in range(100):
            action = select_action(Q, state, True, epsilon) #follow a policy to select action
            # take action 
            next_state, reward, done = take_action(action, rewards, moves, moves_actual, state, visited, lst_dirty)
            
            if next_state not in visited:
                visited.append(next_state)
            
            # choose next action     
            next_action = select_action(Q, next_state, False, epsilon) #don't follow a policy, rather return argmax
            # update Q
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            
            state = next_state
            # check if episode is over
            if done:
                break
    
    max_actions = [key for key, value in Q[current_pos].items() if value == max(Q[current_pos].values())]
    
    if len(max_actions) == 1:
        max_action = max_actions[0]
    else:
        max_action = np.random.choice(max_actions)
    move = moves_actual[moves.index(max_action)]
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]        
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # print(f'current position: {current_pos}\n move: {move}\nMatrix: {pd.DataFrame(V).T}')
    robot.move()