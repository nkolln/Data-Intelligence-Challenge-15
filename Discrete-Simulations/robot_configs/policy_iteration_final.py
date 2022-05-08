import numpy as np
from random import choices

rewards = [0,1,-2] #reward for landing on clean, dirty or obstacle tile, respectively
moves = ['n','e','s','w']

def update_cor(x,y,move):
    '''A function that, given the robot's current tile and the move it means to make, will return
    the coordinates of the tile it will be on after it has made the given move.
    '''
    if move=='n':
        y -= 1
    elif move=='e':
        x += 1
    elif move=='s':
        y += 1
    elif move=='w':
        x -= 1
    return x,y

def robot_epoch(robot, theta_value, gamma_value):
    theta = theta_value
    gamma = gamma_value
    global rewards
    global moves
    global previous_move
    
    current_world = robot.grid.cells
    current_pos = robot.pos
    
    #policy iteration:
    #create Value grid, which will have the same shape as the worldmap and contain a Value for each tile
    V = np.zeros((robot.grid.cells.shape))
        
    #keep track of the world state, the first two dimensions are the same as the world map, the third dimension indicates the state of the tile:
    world_state = np.zeros((*V.shape,3)).astype(bool)
    world_state[:,:,0] = (current_world==0)|(current_world<=-3) #clean/current position
    world_state[:,:,1] = (current_world==1)|(current_world==2) #dirty (incl goal tiles)
    world_state[:,:,2] = (current_world==-1)|(current_world==-2)|(current_world==3) #wall/obstacle/death
    
    #a policy map with the first two dimension the same as the world map. The third dimension provides the probabilities for taking each of the four moves.
    #start with 25% chance for a move in each direction, arbitrary policy
    policy_map = np.full(shape=(*V.shape,4),fill_value=0.25)
    
    opt_policy = False
    pol_it = 0
    #let's start iterating:
    while opt_policy==False and pol_it < 25: #as long as the optimal policy isn't found and the iteration cap isn't reached:
        #fill Value grid with raw rewards as a starting point
        for i in range(3):
            V[world_state[:,:,i]] = rewards[i]
            
        #Alter rewards based on proximity to walls/clean tiles
        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                #for each tile in the value grid...
                if world_state[x,y,2] == True: #ignore walls/obstacles, keep initial rewards
                    continue
                lst_neighbor = []
                count_walls, count_neighbor = 0, 0
                for i in range(4): #check each of the tile's neighbors
                    new_pos = update_cor(x,y,moves[i])
                    #if a neighboring tile is clean, and the current tile is dirty: higher reward
                    if world_state[new_pos][0] == True and world_state[x,y,1] == True:
                            V[x,y] += 2
                    elif world_state[x,y,1] == True: #if the current tile is dirty...
                        if world_state[new_pos][2] == True: #Higher rewards for being nearby boundaries of obstacles
                            V[x,y] += 1
                            count_neighbor += 1
                            count_walls += 1
                            lst_neighbor.append(moves[i]) #keep track of neighboring walls to detect doorways
                        if world_state[new_pos][0] == True:
                            count_neighbor += 1

                if count_walls == 2 and (('w' in lst_neighbor and 'e' in lst_neighbor) or ('n' in lst_neighbor and 's' in lst_neighbor)): #If there are two walls next to eachother than treat as door
                    V[x,y] -= 1
                #Prioritize dirty tiles surrounded by non-dirty neighbors
                if count_neighbor == 4:
                    V[x,y] += 20
                
                #update rewards for death/goal tiles (rare tiles that aren't worth adding a dimension to world_state)
                if current_world[x,y] == 3: #death tile
                    V[x,y] -= 20
                if current_world[x,y] == 2: #goal tile
                    V[x,y] += 1
            
        #now we will update the Values in V iteratively using the Bellman equation:  
        biggest_dif = 1000
        V_it = 0
        while biggest_dif > theta and V_it < 25: 
            #As long as any Value has an update larger than theta, and the iteration cap isn't reached:
            max_dif_so_far = 0
            for x in range(V.shape[0]):
                for y in range(V.shape[1]):
                    #loop over all tiles in V
                    if world_state[x,y,2] == True: #if it's a wall/obstacle, don't bother updating
                        V[x,y] = rewards[2]
                        continue
                    V_old = V[x,y] #remember the tile's old Value
                    V_new = 0
                    #get new V value
                    for i in range(4): #the first summation in the bellman equation (actions)
                        G = 0
                        new_cor = update_cor(x,y,moves[i])
                        for j in range(2): #the second summation (states)
                            G += world_state[new_cor][j] * (rewards[j] + gamma*V_old)
                        V_new += policy_map[x,y,i] * G
                    V[x,y] = V_new
                    #calculate the difference between the old and new Value
                    max_dif_so_far = max(max_dif_so_far, abs(V_new-V_old))
            
            V_it += 1
            #update biggest_dif so the loop won't run forever
            biggest_dif = max_dif_so_far
        
        #policy optimization:
        opt_policy = True
        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                #loop over all the tiles in the world map.
                if world_state[x,y,2] == True: #if it's a wall it doesn't need a policy, we'll never end up there anyway
                    continue
                #keep track of the moves that end up on the tile with the highest Value
                best_moves = np.zeros(4)
                best_V = -np.inf
                for i in range(4):
                    pos_after_move = update_cor(x,y,moves[i])
                    if V[pos_after_move] == best_V: #if a move's Value is the same as the current highest
                        best_moves[i] = 1 #add current move as alt best move
                    elif V[pos_after_move] > best_V: #if a move's Value is larger than the current higherst
                        best_moves = np.zeros(4) #delete previous best move(s)
                        best_moves[i] = 1 #current move is now best move
                        best_V = V[pos_after_move] #new best Value
                        
                old_policy = policy_map[x,y,:]
                new_policy = best_moves / np.sum(best_moves) #turn best moves into probabilities
                
                if (old_policy != new_policy).any(): #if the policy needs to be changed
                    opt_policy = False
                    policy_map[x,y,:] = new_policy #update policy
                    
        pol_it += 1              
    
    #After policy iteration has completed...
    #choose move randomly based on the probabilities in the policy_map
    best_move = choices(population = moves, weights = policy_map[current_pos])[0]
    while robot.orientation != best_move:
        robot.rotate('r')
        
    robot.move()