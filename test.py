
"""import operator

possible_tiles = {(1,1):1,(2,0):1,(1,0):1,(0,1):3,(0,2):1,(-1,1):0,(-1,0):2,(0,-1):1} #robot.possible_tiles_after_move()
#possible_tiles_good = {k:v for k,v in possible_tiles.items() if float(v) >= 1}
possible_tiles_one = {k:v for k,v in possible_tiles.items() if abs(k[0])+abs(k[1])==1}
print(possible_tiles_one)
#[sum(abs(k[0])+sum(abs(k[1]))) for k in possible_tiles.keys()]
farthest_step_vision = max([abs(k[0])+abs(k[1]) for k in possible_tiles.keys()])
print(farthest_step_vision)
#cap at 2 for now
if farthest_step_vision > 2:
    farthest_step_vision = 2
#farthest_step_vision = max([abs(k[0])+abs(k[1]) for k in possible_tiles.keys()])
for i in range(1,farthest_step_vision):
    for key, val in possible_tiles_one.items():
        #dct_val = {}
        lst_val = []
        possible_tiles_iter = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) + abs(move[1]) == i + 1}
        for key_it,val_it in possible_tiles_iter.items():
            key_new = tuple(map(operator.sub, key_it, key))
            if(abs(key_new[0])+ abs(key_new[1]) == i):
                lst_val.append(val_it)
                print(key_new, key_it, key)
        print(lst_val)
        if lst_val:
            if len(lst_val)!=1:
                max_val = max(lst_val)
            else:
                max_val = lst_val[0]
        else:
            max_val = 0
        
        if max_val <1:
            max_val = 0
        possible_tiles_one.update({key:max_val+val})
        print(possible_tiles_one)

print(max(possible_tiles_one, key=possible_tiles_one.get))
print(list(possible_tiles_one.keys())[list(possible_tiles_one.values()).index(1.0)])

#-----------------------------------
possible_tiles = {(1, -1): -2, (1, 0): 1, (0, 1): 1, (-1, 0): -1}
print({k:v for k,v in possible_tiles.items() if float(v) < 1})
print(list(possible_tiles.keys()))
print(list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)])

possible_tiles = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) < 2 and abs(move[1]) < 2}
print(possible_tiles)"""