import numpy as np
import operator
import pandas as pd
#calc = 0

def robot_epoch(robot):
    #rewards = [-1,1,-2] #reward for landing on clean, dirty or obstacle, respectively
    moves = ['n','e','s','w'];moves_actual = [(0,-1),(1,0),(0,1),(-1,0)]
    theta = 0.1; gamma = 0.5; noise = 0.2
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape

    #bounds = [[current_pos[0], current_pos[0]+shape_w[0]],[current_pos[1], current_pos[1]+shape_w[1]]]
    valid_states = [];lst_end = [];lst_taboo = [];lst_wall = [];all_states=[];lst_clean=[];lst_dirty=[]
    dct_lsts = {-2:lst_taboo,-1:lst_wall,0:lst_clean,1:lst_dirty,2:lst_clean,3:lst_end}
    #Categorizes every state
    for x in range(shape_w[0]):
        for y in range(shape_w[1]):
            cur_val = current_world[x][y]
            for key,lst in dct_lsts.items():
            #for num,lst in zip(lst_vals,lst_lsts):
                if cur_val == key:
                    lst.append((x,y))
    all_states.extend(lst_clean);all_states.extend(lst_dirty)
    valid_states.extend(all_states)
    all_states.extend(lst_end)

    actions = {}; rewards = np.copy(current_world)
    #Stores all the possible movements for valid states
    #Added logic to give scores based on location to other clean tiles and walls
    for coord in valid_states:
        lst_dir = [];lst_neighbor=[]
        count_walls = 0; count_neighbor = 0
        for i in range(4):
        #for dir in moves_actual:
            dir = moves_actual[i]
            new_loc = tuple(map(operator.add, coord, dir))
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
            print('-'*20)
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

    #loop and maximize score
    it = 0
    while it < 25:
        most_change = 0
        for state in all_states:
            if state in policy: #Prunes scores for only moves which are possible
                v_old = V[state]
                v_new = 0
                for act in actions[state]: #Loops through possible actions of a state
                    dir = moves_actual[moves.index(act)] #Gets the movement converted from letter to numeric
                    loc = tuple(map(operator.add, state, dir)) #Finds the new location
                    
                    #Looks at other options
                    act_other = [i for i in actions[state] if i != act]
                    if len(act_other)>=1:
                        random_act=np.random.choice([i for i in actions[state] if i != act])
                        noise_dir = moves_actual[moves.index(random_act)]
                        loc_rand = tuple(map(operator.add, state, noise_dir))

                        V_temp = rewards[state] + (gamma * ((1-noise)* V[loc] + (noise * V[loc_rand])))
                    else:
                        V_temp = rewards[state] + (gamma * (V[loc] ))
                    if V_temp > v_new:
                        v_new = V_temp
                        policy[state] = dir

                V[state] = v_new
                most_change = max(most_change, np.abs(v_old -V[state]))
        
        
        #print(f'iteration: {it+1}\nChange: {most_change}')
        if most_change < theta:
            break
        it+=1
    
    
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
                possible_tiles_fr.update({key:V[key_n]})
    
    #If in vacuum zone just moving across is the best way to find something of value
    """if all(value == 0 for value in possible_tiles_fr.values()):
        robot.move()
    else:"""
    move = max(possible_tiles_fr, key=possible_tiles_fr.get)
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