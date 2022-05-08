import numpy as np
import operator
import pandas as pd
#calc = 0

def robot_epoch(robot):
    #rewards = [-1,1,-2] #reward for landing on clean, dirty or obstacle, respectively
    moves = ['n','e','s','w'];moves_actual = [(0,-1),(1,0),(0,1),(-1,0)]
    theta = 0.2; gamma = 0.8; noise = 0.2
    current_world = robot.grid.cells; current_pos = robot.pos; shape_w = current_world.shape

    bounds = [[current_pos[0], current_pos[0]+shape_w[0]],[current_pos[1], current_pos[1]+shape_w[1]]]
    valid_states = [];end_states = [];lst_taboo = [];lst_wall = [];all_states=[]
    #Categorizes every state
    for x in range(shape_w[0]):
        for y in range(shape_w[1]):
            if current_world[x][y] == 3:
                end_states.append((x,y))
            elif current_world[x][y] == -1:
                lst_wall.append((x,y))
            elif current_world[x][y] == -2:
                lst_taboo.append((x,y))
            else:
                valid_states.append((x,y))
    all_states.extend(valid_states)
    all_states.extend(end_states)

    actions = {}; rewards = np.copy(current_world)
    #Stores all the possible movements for valid states
    for coord in valid_states:
        lst_dir = []
        for i in range(4):
        #for dir in moves_actual:
            dir = moves_actual[i]
            new_loc = tuple(map(operator.add, coord, dir))
            if new_loc in valid_states:#((new_loc not in lst_taboo) and (new_loc not in end_states)):# and (new_loc[0] in range(bounds[0][0],bounds[0][1]))and(new_loc[1] in range(bounds[1][0],bounds[1][1]))):
                lst_dir.append(moves[i])
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
    while it < 100:
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
                        V_temp = rewards[state] + (gamma * (V[loc]))
                    if V_temp > v_new:
                        v_new = V_temp
                        policy[state] = dir

                V[state] = v_new
                most_change = max(most_change, np.abs(v_old -V[state]))
        
        
        print(f'iteration: {it+1}\nChange: {most_change}')
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
    if all(value == 0 for value in possible_tiles_fr.values()):
        robot.move()
    else:
        move = max(possible_tiles_fr, key=possible_tiles_fr.get)
        #move = list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)]
        new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
            # Orient ourselves towards the dirty tile:  
        while new_orient != robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            # print('Rotating right once.')
            robot.rotate('r')
        #print(f'current position: {current_pos}\n move: {move}')
        robot.move()
    print(pd.DataFrame(V).T)
    #print('-'*50)