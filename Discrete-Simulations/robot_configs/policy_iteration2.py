import numpy as np
import operator
import pandas as pd

#calc = 0

def policy_evaluation(policy,all_states,actions,moves,moves_actual,noise,rewards,gamma,V,theta):
    #loop and maximize score
    it = 0
    for it in range(100):
        delta = 0
        for state in all_states:
            if state in policy: #Prunes scores for only moves which are possible
                """v_old = V[state]
                v_new = 0"""
                v = 0
                action_probability = 1/(len(actions[state]))
                for act in actions[state]: #Loops through possible actions of a state
                    dir = moves_actual[moves.index(act)] #Gets the movement converted from letter to numeric
                    loc = tuple(map(operator.add, state, dir)) #Finds the new location
                    
                    #Looks at other options
                    #act_other = [i for i in actions[state] if i != act]
                    """if len(act_other)>=1:
                        random_act=np.random.choice([i for i in actions[state] if i != act])
                        noise_dir = moves_actual[moves.index(random_act)]
                        loc_rand = tuple(map(operator.add, state, noise_dir))

                        V_temp = rewards[state] + (gamma * ((1-noise)* V[loc] + (noise * V[loc_rand])))"""
                    
                    v += action_probability*(1-noise)*(rewards[state] + (gamma * (V[loc])))
                delta = max(delta, np.abs(V[state] - v))
                V[state]= v

                if delta < theta:
                    return(V)
                if it == 100:
                    return(V)


def robot_epoch(robot):
    #rewards = [-1,1,-2] #reward for landing on clean, dirty or obstacle, respectively
    moves = ['n','e','s','w'];moves_actual = [(0,-1),(1,0),(0,1),(-1,0)]
    theta = 0.1; gamma = 0.5; noise = 0.2
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape
    #current_world[current_pos] = 0
    print(current_world)


    #bounds = [[current_pos[0], current_pos[0]+shape_w[0]],[current_pos[1], current_pos[1]+shape_w[1]]]
    valid_states = [];lst_end = [];lst_goal=[];lst_taboo = [];lst_wall = [];all_states=[];lst_clean=[];lst_dirty=[]
    dct_lsts = {-3:lst_clean,-2:lst_taboo,-1:lst_wall,0:lst_clean,1:lst_dirty,2:lst_goal,3:lst_end}
    #current_world = pd.DataFrame(current_world).T
    #Categorizes every state
    for x in range(shape_w[0]):
        for y in range(shape_w[1]):
            cur_val = current_world[x][y]
            for key,lst in dct_lsts.items():
            #for num,lst in zip(lst_vals,lst_lsts):
                if cur_val == key:
                    lst.append((x,y))
    for x in range(shape_w[0]):
        for y in range(shape_w[1]):
            cur_val = current_world[x][y]
            if cur_val < -3:
                lst_clean.append((x,y))
    all_states.extend(lst_clean);all_states.extend(lst_dirty);all_states.extend(lst_goal)
    valid_states.extend(all_states)
    all_states.extend(lst_end)

    #Prune mode
    prune_size = 4
    prune_state = 0
    if prune_state == 1:
        window = [[max(current_pos[0]-prune_size,0), min(current_pos[0]+prune_size,shape_w[0])],[max(current_pos[1]-prune_size,0), min(current_pos[1]+prune_size,shape_w[1])]]

        #valid_states_temp = np.copy(valid_states)
        while True and lst_dirty:
            lst_dirty_temp = []
            #print(f'window range: {window}')
            for item in lst_dirty:
                if window[0][0] < item[0] and window[0][1] > item[0]:
                    if window[1][0] < item[1] and window[1][1] > item[1]:
                        lst_dirty_temp.append(item)
            #lst_dirty = lst_dirty_temp

            if not lst_dirty_temp: #if no value is dirty then increase counter
                prune_size+=1
                #print('here')
            else:
                break
        lst_dirty=lst_dirty_temp


    actions = {}; rewards = np.copy(current_world)

    #Stores all the possible movements for valid states
    #Added logic to give scores based on location to other clean tiles and walls
    for coord in valid_states:
        lst_dir = [];lst_neighbor=[]
        count_walls = 0; count_neighbor = 0
        """if current_world[coord]==0: #if its clean
            rewards[coord] = -0.5"""
        for i in range(4):
        #for dir in moves_actual:
            dir = moves_actual[i]
            new_loc = tuple(map(operator.add, coord, dir))
            if new_loc in valid_states:#((new_loc not in lst_taboo) and (new_loc not in end_states)):# and (new_loc[0] in range(bounds[0][0],bounds[0][1]))and(new_loc[1] in range(bounds[1][0],bounds[1][1]))):
                lst_dir.append(moves[i])
                if new_loc in lst_clean and current_world[coord] == 1 or current_world[coord] == 2: #Reward for nearby clean tiles
                    rewards[coord] += 2
            elif current_world[coord]==1 or current_world[coord]==2:
                if new_loc in lst_wall: #Rewards for being nearby to boundaries of map
                    rewards[coord] += 3
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
            #print('-'*20)
        #If it has no dirty tiles around, prioritize
        if count_neighbor == 4:
            rewards[coord] +=20
        if len(lst_dir)>0:
            actions.update({coord:lst_dir})
    """print(f'valid_states: {valid_states}')
    print(f'end states: {end_states}')
    print(f'taboo list: {lst_taboo}')
    print(f'current_pos: {current_pos}')"""
            

    #Random initial policy for all
    policy = {}
    for act in actions.keys():
        policy[act] = np.random.choice(actions[act])

    V = np.copy(rewards)

    for i in range(100):
        stable_policy = True
        V = policy_evaluation(policy,all_states,actions,moves,moves_actual,noise,rewards,gamma,V,theta)
        #print(V)
        for state in valid_states:
            current_action = policy[state]
            best_action = (0,0)
            best_score = 0
            for act in actions[state]: #Loops through possible actions of a states
                dir = moves_actual[moves.index(act)] #Gets the movement converted from letter to numeric
                loc = tuple(map(operator.add, state, dir)) #Finds the new location
                V_temp = rewards[state] + (gamma * (V[loc] ))
                if V_temp > best_score:
                    best_action = act#moves[moves.index(act)]
                    best_score = V_temp
                    #print(f'act: {act} and score: {best_score}')
                    print(best_action)

            if current_action != best_action:
                stable_policy = False
                policy[state]= best_action#moves_actual[moves.index(best_action)]
                print(best_action)
                print('-'*20)
        if stable_policy:
            pass
            #print(policy, V)
        else:
            print(f'Never stable, iteration: {i}')
        
    
    """
    possible_tiles = robot.possible_tiles_after_move()
    #Just to make sure its 1
    possible_tiles = {k:v for k,v in possible_tiles.items() if abs(k[0])+abs(k[1])==1}

    #Update tiles with calculated score (only keeps relevant)
    possible_tiles_fr = {}
    if all(current_world[value] <= 0 for value in valid_states):
        for key in possible_tiles.keys():
            key_n = tuple(map(operator.add, key, current_pos))
            if key_n in all_states:
                possible_tiles_fr.update({key:V[key_n]})
    else:
        for key in possible_tiles.keys():
            key_n = tuple(map(operator.add, key, current_pos))
            if key_n in valid_states:
                possible_tiles_fr.update({key:V[key_n]})"""
    
    #If in vacuum zone just moving across is the best way to find something of value
    """if all(value == 0 for value in possible_tiles_fr.values()):
        robot.move()
    else:"""
    print(current_pos)
    print(f'V: {V}\nPolicy: {policy[current_pos]}')
    print(policy)
    print(moves_actual)
    move = moves_actual[moves.index(policy[current_pos])]
    #move = max(possible_tiles_fr, key=possible_tiles_fr.get)
    #move = list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)]
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
        # Orient ourselves towards the dirty tile:  
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    #print(f'current position: {current_pos}\n move: {move}\nMatrix: {pd.DataFrame(V).T}')
    robot.move()
    #print(pd.DataFrame(V).T)
    #print('-'*50)