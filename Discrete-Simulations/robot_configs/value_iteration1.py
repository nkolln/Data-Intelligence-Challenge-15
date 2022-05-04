
import operator
import random
# Very boring robot that does nothing:
def robot_epoch(robot):

    learn_rate = 0.5
    possible_tiles = robot.possible_tiles_after_move()#{(1,1):1,(2,0):1,(1,0):2,(0,1):3,(0,2):1,(-1,1):0,(-1,0):2,(0,-1):1}

    #Shuffle list to add a random element to the program
    l = list(possible_tiles.keys())
    random.shuffle(l)
    possible_tiles = {key:possible_tiles[key] for key in l}
    #print(f'possible_tiles: {possible_tiles}')
    #possible_tiles_good = {k:v for k,v in possible_tiles.items() if float(v) >= 1}
    
    #Select all the tiles of distance 1. This is what is needed
    possible_tiles_one = {k:v for k,v in possible_tiles.items() if abs(k[0])+abs(k[1])==1}
    #print(f'possible_tiles_one: {possible_tiles_one}')

    #Finds the current farthest vision
    farthest_step_vision = max([abs(k[0])+abs(k[1]) for k in possible_tiles.keys()])
    
    for i in range(1,farthest_step_vision):
        for key, val in possible_tiles_one.items():
            lst_val = []
            #finds all moves at distance i + 1 from origin
            possible_tiles_iter = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) + abs(move[1]) == i + 1}
            for key_it,val_it in possible_tiles_iter.items():
                #Subtracts keys to calulate distance from a certain key
                key_new = tuple(map(operator.sub, key_it, key))
                #Adds to list if it is in distance i
                if(abs(key_new[0])+ abs(key_new[1]) == i):
                    lst_val.append(val_it)

            #Logic to get max val from list
            if lst_val:
                if len(lst_val)!=1:
                    max_val = max(lst_val)
                else:
                    max_val = lst_val[0]
            else:
                max_val = 0
            """if max_val <1:
                max_val = 0"""
            
            possible_tiles_one.update({key:(learn_rate**i)*max_val+val})
    #print(f'possible_tiles_one: {possible_tiles_one}')

    if all(value == 0 for value in possible_tiles_one.values()):
        robot.move()
    else:
        move = max(possible_tiles_one, key=possible_tiles_one.get)
        #move = list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)]
        new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
            # Orient ourselves towards the dirty tile:  
        while new_orient != robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            # print('Rotating right once.')
            robot.rotate('r')
        robot.move()