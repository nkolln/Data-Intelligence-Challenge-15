import numpy as np
from random import choices

'''
NOTE: This implementation does not work AT ALL. I've yet to figure out why though.
'''


#TO IMPLEMENT: global variable(s) so we remember Values and policies between epochs
#world = np.array(...)

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
    #global world
    rewards = [-1,1,-2] #reward for landing on clean, dirty or obstacle, respectively
    moves = ['n','e','s','w']
    
    current_world = robot.grid.cells
    current_pos = robot.pos
    
    #policy iteration:
    #start with 0 Value for all cells
    V = np.zeros((robot.grid.cells.shape))
    
    #keep track of (expected) clean and dirty tiles:
    initial_world = np.zeros((*V.shape,3))
    initial_world[:,:,0] = (current_world==0).astype(float) #probability of clean tiles
    initial_world[:,:,1] = (current_world==1).astype(float) #probability of dirty tiles
    initial_world[:,:,2] = (current_world<0).astype(float) #probability of obstacle/wall
    
    #start with 25% chance for a move in each direction, arbitrary policy
    policy_map = np.full(shape=(*V.shape,4),fill_value=0.25)
    theta = 0.1
    gamma = 0.5
    
    opt_policy = False
    #let's start iterating:
    while opt_policy==False:
        print("still haven't found optimal policy")
        biggest_dif = 10
        world_state = initial_world.copy() #keep track of probability of clean vs dirty tiles
        while biggest_dif > theta:
            print(f"still iterating, biggest dif is {biggest_dif}")
            max_dif_so_far = 0
            for x in range(V.shape[0]):
                for y in range(V.shape[1]):
                    if world_state[x,y,2] == 1: #if it's a wall
                        V[x,y] = rewards[2]
                        continue
                    V_old = V[x,y]
                    V_new = 0
                    #get new V value
                    for i in range(4):
                        G = 0
                        new_cor = update_cor(x,y,moves[i])
                        for j in range(3):
                            G += world_state[new_cor][j] * (rewards[j] + gamma*V_old)
                        V_new += policy_map[x,y,i] * G
                    V[x,y] = V_new
                    max_dif_so_far = max(max_dif_so_far, abs(V_new-V_old))
                    
            #update probabilities for each tile state (idk if I'm doing prob theory right...):
            for i in range(4):
                new_pos = update_cor(*current_pos, moves[i])
                if world_state[new_pos][2] == 0: #if bordering tile is Not a wall
                    world_state[new_pos][1] *= 1 - policy_map[current_pos][i]
                    world_state[new_pos][0] = 1 - world_state[new_pos][1]
                
            #update biggest_dif so the loop won't run forever
            biggest_dif = max_dif_so_far
        
        #policy optimization:
        opt_policy = True
        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                #if it's a wall it doesn't need a policy, we'll never end up there anyway
                if world_state[x,y,2] == 1:
                    continue
                best_moves = np.zeros(4)
                best_V = 0
                for i in range(4):
                    pos_after_move = update_cor(x,y,moves[i])
                    if V[pos_after_move] == best_V:
                        best_moves[i] = 1 #add current move as alt best move
                    elif V[pos_after_move] > best_V:
                        best_moves = np.zeros(4) #delete previous best move(s)
                        best_moves[i] = 1 #current move is now best move
                        best_V = V[pos_after_move] #new best Value
                old_policy = policy_map[x,y,:]
                new_policy = best_moves / np.sum(best_moves) #turn into probabilities
                if (old_policy != new_policy).any(): #if the policy needs to be changed
                    opt_policy = False
                policy_map[x,y,:] = new_policy #update policy
                
    
    # policy = 0
    # best_moves = []
    # for i in range(4):
    #     new_pos = update_cor(*current_pos, moves[i])
    #     if policy_map[new_pos][i] == policy:
    #         best_moves.append(moves[i])
    #     elif policy_map[new_pos][i] > policy:
    #         policy = policy_map[new_pos][i]
    #         best_moves = [moves[i]] 
    
    #choose move randomly based on the probabilities in the policy_map
    best_move = choices(population = moves, weights = policy_map[current_pos])[0]
    print(f"chosen move: {best_move}")
    #print(f'moving the bot now to {best_move}!')
    while robot.orientation != best_move:
        #print('turning')
        robot.rotate('r')
        
    robot.move()
        
    
    
    
    
    