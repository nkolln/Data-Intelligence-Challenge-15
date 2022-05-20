import numpy as np
import operator
import pandas as pd


# calc = 0

def get_act(cur_loc,cur_state,actions,noise,Q,moves):
    if np.random.uniform(0, 1) < noise:
        cur_act = np.random.choice(actions[cur_loc])
    else:
        cur_lst = Q[cur_state, :]
        if all(val == 0 for val in cur_lst):
            cur_act = np.random.choice(actions[cur_loc])
        else:
            final_lst = [i if i != 0 else -10000 for i in cur_lst]
            max_score = np.argmax(final_lst)
            cur_act = moves[max_score]
    return(cur_act)

def get_info(cur_loc, dct_map, moves,moves_actual,actions,noise,Q):
    cur_state = dct_map[cur_loc]
    cur_act = get_act(cur_loc,cur_state,actions,noise,Q,moves)
    cur_dir = moves_actual[moves.index(cur_act)]
    cur_num = moves.index(cur_act)
    return(cur_state,cur_dir,cur_num)

def categorize_states(shape_w,current_world):
    # current_world = pd.DataFrame(current_world).T
    # Categorizes every state
    valid_states = [];
    lst_end = [];
    lst_goal = [];
    lst_taboo = [];
    lst_wall = [];
    all_states = [];
    lst_clean = [];
    lst_dirty = []
    dct_lsts = {-2: lst_taboo, -1: lst_wall, 0: lst_clean, 1: lst_dirty, 2: lst_goal, 3: lst_end}
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
    all_states.extend(lst_clean);
    all_states.extend(lst_dirty);
    all_states.extend(lst_goal)
    valid_states.extend(all_states)
    all_states.extend(lst_end)
    return(lst_taboo,lst_wall,lst_clean,lst_dirty,lst_goal,lst_end,all_states,valid_states,dct_map)

def initialize_rew_act(rewards,actions,valid_states,moves_actual,moves,lst_clean,lst_wall, lst_taboo,current_world):
    # Stores all the possible movements for valid states
    # Added logic to give scores based on location to other clean tiles and walls
    for coord in valid_states:
        lst_dir = [];
        lst_neighbor = []
        count_walls = 0;
        count_neighbor = 0
        """if current_world[coord]==0: #if its clean
            rewards[coord] = -0.5"""
        for i in range(4):
            # for dir in moves_actual:
            dir = moves_actual[i]
            new_loc = tuple(map(operator.add, coord, dir))
            if new_loc in valid_states:  # ((new_loc not in lst_taboo) and (new_loc not in end_states)):# and (new_loc[0] in range(bounds[0][0],bounds[0][1]))and(new_loc[1] in range(bounds[1][0],bounds[1][1]))):
                lst_dir.append(moves[i])
                if new_loc in lst_clean and current_world[coord] == 1 or current_world[
                    coord] == 2:  # Reward for nearby clean tiles
                    rewards[coord] += 2
            elif current_world[coord] == 1 or current_world[coord] == 2:
                if new_loc in lst_wall:  # Rewards for being nearby to boundaries of map
                    rewards[coord] += 3
                if new_loc in lst_taboo:  # Rewards for nearby boundaries
                    rewards[coord] += 1
                if new_loc in lst_clean or new_loc in lst_taboo or new_loc in lst_wall:
                    count_neighbor += 1
                # elif current_world[coord]==1:
                if new_loc in lst_taboo:  # Logic to find if there is a doorway
                    count_walls += 1
                    lst_neighbor.append(dir)
            elif current_world[coord] == 0:
                rewards[coord] += 0
        if count_walls == 2 and tuple(map(operator.add, lst_neighbor[0], lst_neighbor[1])) == (
        0, 0):  # If there are two walls next to eachother than treat as door
            rewards[coord] -= 1
            # print('-'*20)
        # If it has no dirty tiles around, prioritize
        if count_neighbor == 4:
            rewards[coord] += 40
        if len(lst_dir) > 0:
            actions.update({coord: lst_dir})
            #Q.update({coord:[0 for i in range(len(lst_dir))]})
    return(rewards,actions)


def move(Q,dct_map,current_pos,moves_actual,robot):
    cur_state = dct_map[current_pos]
    final_lst = [i if i != 0 else -10000 for i in Q[cur_state, :]]
    max_score = np.argmax(final_lst)
    #print(f'Array: {Q}\nCurrent position: {current_pos}\nState: {cur_state}\nQ scores: {Q[cur_state]}\nMax score: {max_score}')
    #index = Q[cur_state,:].index(max_score)
    #dir = actions[current_pos][index]
    move = moves_actual[max_score]

    # move = list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)]
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # print(f'current position: {current_pos}\n move: {move}\nMatrix: {pd.DataFrame(V).T}')
    robot.move()

def calculate_Q_matrix(shape_w,moves,moves_actual,current_pos,dct_map,actions,noise,alpha,rewards,gamma,inner_loop_count):
    #V = np.copy(rewards)
    Q = np.zeros((shape_w[0]*shape_w[1], len(moves)))
    # loop and maximize score
    it = 0
    while it < 500:
        most_change = 0
        #Choose current state and choose new direction
        cur_pos = current_pos
        cur_state,cur_dir,cur_num = get_info(cur_pos, dct_map, moves,moves_actual,actions,noise,Q)
        it_inner = 0
        while it_inner < inner_loop_count:
            #Choose another step at a random direction
            #Add clause to choose actual best at random
            nxt_state_actual = tuple(map(operator.add, cur_pos, cur_dir))
            nxt_state,nxt_dir,nxt_num = get_info(nxt_state_actual, dct_map, moves,moves_actual,actions,noise,Q)
            #Update Q according to the new values
            Q[cur_state, cur_num] = Q[cur_state, cur_num] + alpha * (rewards[nxt_state_actual] + gamma * Q[nxt_state, nxt_num] - Q[cur_state, cur_num])
            #save the new step as the current
            cur_state = nxt_state
            cur_dir = nxt_dir
            cur_num = nxt_num
            cur_pos = nxt_state_actual

            it_inner+=1
        it+=1

        # print(f'iteration: {it+1}\nChange: {most_change}')
        """if most_change < theta:
            break"""
    return(Q)

def robot_epoch(robot):
    # rewards = [-1,1,-2] #reward for landing on clean, dirty or obstacle, respectively
    moves = ['n', 'e', 's', 'w'];
    moves_actual = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    theta = 0.2; # 0.2 1 5
    alpha = .8
    gamma = 0.8; # 0.2 0.5 0.8 1
    noise = 0.5
    current_world = robot.grid.cells;
    current_pos = robot.pos;
    shape_w = current_world.shape

    #initialize all possible states
    
    lst_taboo,lst_wall,lst_clean,lst_dirty,lst_goal,lst_end,all_states,valid_states,dct_map = categorize_states(shape_w,current_world)

    actions = {};Q={}
    rewards = np.copy(current_world)

    #Initialize the rewards and possible actions for moves at the beginning of a step
    rewards,actions = initialize_rew_act(rewards,actions,valid_states,moves_actual,moves,lst_clean,lst_wall, lst_taboo,current_world)

    inner_loop_count = shape_w[0]*shape_w[1]/4
    inner_loop_count = min(inner_loop_count,40)
    Q = calculate_Q_matrix(shape_w,moves,moves_actual,current_pos,dct_map,actions,noise,alpha,rewards,gamma,inner_loop_count)
    
    move(Q,dct_map,current_pos,moves_actual,robot)