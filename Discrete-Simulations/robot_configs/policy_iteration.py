import numpy as np
from random import choices

'''
Update 7 May: Included Nick's heuristics for the reward values. The performance is still not
great and it seems to be caused by certain tiles around the robot's current tile being assigned
a value of 0 when they shouldn't be. I thought this was caused by something going wrong during
the policy optimization step, as it seemed to happen most often after the third round of policy
optimization. However, capping policy optimization to no more than three only helped a little.
Yasemin and I are both stuck, we've only been able to conclude that whatever is happening is weird.
Pls help.
'''

#TO IMPLEMENT: global variable(s) so we remember Values and policies between epochs, for speed
#V = ...
#policy_map = ...
rewards = [-4,50,-10] #reward for landing on clean, dirty or obstacle, respectively
moves = ['n','e','s','w']
previous_move = None

def update_cor(x,y,move):
    if move=='n':
        y -= 1
    elif move=='e':
        x += 1
    elif move=='s':
        y += 1
    elif move=='w':
        x -= 1
    return x,y

def robot_epoch(robot):
    #global V
    #global policy_map
    global rewards
    global moves
    global previous_move
    pm_one_hot = np.array(moves)==previous_move if previous_move else None
    pm_idx = np.where(pm_one_hot==True)[0] if previous_move else None
    
    current_world = robot.grid.cells
    current_pos = robot.pos
    
    #policy iteration:
    #create Value grid
    V = np.zeros((robot.grid.cells.shape))
        
    #keep track of clean and dirty tiles:
    world_state = np.zeros((*V.shape,3)).astype(bool)
    world_state[:,:,0] = (current_world==0)|(current_world==-3) #clean/current position
    world_state[:,:,1] = (current_world==1) #dirty
    world_state[:,:,2] = (current_world==-1)|(current_world==-2) #wall/obstacle, -3 is robot's current location so can't use <0
    
    #print(world_state[:,:,1])
    
    #start with 25% chance for a move in each direction, arbitrary policy
    policy_map = np.full(shape=(*V.shape,4),fill_value=0.25)
    theta = 0.1
    gamma = 1
    
    opt_policy = False
    pol_it = 0
    #let's start iterating:
    while opt_policy==False and pol_it < 2:
        #print("still haven't found optimal policy")
        #fill Value grid with raw rewards as a starting point
        for i in range(3):
            #print("filling V")
            V[world_state[:,:,i]] = rewards[i]
            
        #Alter rewards based on proximity to walls/clean tiles
        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                if world_state[x,y,2] == True: #ignore walls, keep initial rewards
                    continue
                lst_neighbor=[]
                count_walls = 0; count_neighbor = 0
                for i in range(4):
                #for dir in moves_actual:
                    #dir = moves_actual[i]
                    new_pos = update_cor(x,y,moves[i])
                    if world_state[new_pos][2] == False: #if new position isn't a wall
                        #lst_dir.append(moves[i])
                        if world_state[new_pos][0] == True and world_state[x,y,1] == True: #Reward for nearby clean tiles
                            V[x,y] += 0.2*V[x,y]
                    elif world_state[x,y,1] == True: #if current tile is dirty
                        if world_state[new_pos][2] == True: #Rewards for being nearby to boundaries of obstacles
                            V[x,y] += 0.2*V[x,y]
                            count_neighbor += 1
                            count_walls += 1
                            lst_neighbor.append(moves[i])
                        if world_state[new_pos][0] == True:
                            count_neighbor += 1

                if count_walls == 2 and (('w' in lst_neighbor and 'e' in lst_neighbor) or ('n' in lst_neighbor and 's' in lst_neighbor)): #If there are two walls next to eachother than treat as door
                    V[x,y] -= 0.1*V[x,y]
                    #print('-'*20)
                #If it has no dirty tiles around, prioritize
                if count_neighbor == 4:
                    V[x,y] += 0.7*V[x,y]
            
            
        biggest_dif = 1000
        V_it = 0
        while biggest_dif > theta and V_it < 25:
            #print(f"still iterating, biggest dif is {biggest_dif}")
            
            max_dif_so_far = 0
            for x in range(V.shape[0]):
                for y in range(V.shape[1]):
                    if world_state[x,y,2] == True: #if it's a wall
                        V[x,y] = rewards[2]
                        continue
                    V_old = V[x,y]
                    V_new = 0
                    #get new V value
                    for i in range(4): #the first summation in the bellman equation (actions)
                        G = 0
                        new_cor = update_cor(x,y,moves[i])
                        for j in range(2): #the second summation (states)
                            G += world_state[new_cor][j] * (rewards[j] + gamma*V_old)
                        V_new += policy_map[x,y,i] * G
                    V[x,y] = V_new
                    max_dif_so_far = max(max_dif_so_far, abs(V_new-V_old))
            
            #print(V)
            V_it += 1
                
            #update biggest_dif so the loop won't run forever
            biggest_dif = max_dif_so_far
        
        #policy optimization:
        opt_policy = True
        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                #if it's a wall it doesn't need a policy, we'll never end up there anyway
                if world_state[x,y,2] == True:
                    continue
                best_moves = np.zeros(4)
                best_V = -np.inf
                for i in range(4):
                    pos_after_move = update_cor(x,y,moves[i])
                    if V[pos_after_move] == best_V:
                        best_moves[i] = 1 #add current move as alt best move
                    elif V[pos_after_move] > best_V:
                        best_moves = np.zeros(4) #delete previous best move(s)
                        best_moves[i] = 1 #current move is now best move
                        best_V = V[pos_after_move] #new best Value
                old_policy = policy_map[x,y,:]
                #the lines below ensure that if the policy is torn between moves, and one of the
                #moves it is torn between was the move we took previously, it will take that same
                #move again. Forces it to go in straight lines.
                if previous_move and np.sum(best_moves) > 1 and best_moves[pm_idx]==1:
                    best_moves = pm_one_hot.astype(int)
                    
                new_policy = best_moves / np.sum(best_moves) #turn into probabilities
                # if (x,y)==current_pos:
                #     print(f"{x},{y}: old: {old_policy}, new: {new_policy}, it: {pol_it}")
                
                if (old_policy != new_policy).any(): #if the policy needs to be changed
                    opt_policy = False
                    policy_map[x,y,:] = new_policy #update policy
                    
        pol_it += 1
        for i in range(4):
            neigh_pos = update_cor(*current_pos, moves[i])
            print(f"{neigh_pos[0]},{neigh_pos[1]}: {policy_map[neigh_pos]}, iteration: {pol_it}")
                
    
    #choose move randomly based on the probabilities in the policy_map
    best_move = choices(population = moves, weights = policy_map[current_pos])[0]
    for i in range(4):
        pos_after = update_cor(*current_pos, moves[i])
        print(f"{moves[i]}: {V[pos_after]}")
    #print(f"chosen move on {current_pos}: {best_move}, out of: {policy_map[current_pos]}")
    #print(V)
    previous_move = best_move
    while robot.orientation != best_move:
        #print('turning')
        robot.rotate('r')
        
    robot.move()