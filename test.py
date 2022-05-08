
import gym
import numpy as np

def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
        # Number of evaluation iterations
        evaluation_iterations = 1
        # Initialize a value function for each state as zero
        V = np.zeros(environment.nS)
        # Repeat until change in value is below the threshold
        for i in range(int(max_iterations)):
                # Initialize a change of value function as zero
                delta = 0
                # Iterate though each state
                for state in range(environment.nS):
                       # Initial a new value of current state
                       v = 0
                       # Try all possible actions which can be taken from this state
                       for action, action_probability in enumerate(policy[state]):
                             # Check how good next state will be
                             for state_probability, next_state, reward, terminated in environment.P[state][action]:
                                  # Calculate the expected value
                                  v += action_probability * state_probability * (reward + discount_factor * V[next_state])
                       
                       # Calculate the absolute change of value function
                       delta = max(delta, np.abs(V[state] - v))
                       # Update value function
                       V[state] = v
                evaluation_iterations += 1
                
                # Terminate if value change is insignificant
                if delta < theta:
                        print(f'Policy evaluated in {evaluation_iterations} iterations.')
                        return V

def one_step_lookahead(environment, state, V, discount_factor):
        action_values = np.zeros(environment.nA)
        for action in range(environment.nA):
                for probability, next_state, reward, terminated in environment.P[state][action]:
                        action_values[action] += probability * (reward + discount_factor * V[next_state])
        return action_values


def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
        # Start with a random policy
        #num states x num actions / num actions
        policy = np.ones([environment.nS, environment.nA]) / environment.nA
        # Initialize counter of evaluated policies
        evaluated_policies = 1
        # Repeat until convergence or critical number of iterations reached
        for i in range(int(max_iterations)):
                stable_policy = True
                # Evaluate current policy
                V = policy_evaluation(policy, environment, discount_factor=discount_factor)
                # Go through each state and try to improve actions that were taken (policy Improvement)
                for state in range(environment.nS):
                        # Choose the best action in a current state under current policy
                        current_action = np.argmax(policy[state])
                        # Look one step ahead and evaluate if current action is optimal
                        # We will try every possible action in a current state
                        action_value = one_step_lookahead(environment, state, V, discount_factor)
                        # Select a better action
                        best_action = np.argmax(action_value)
                        # If action didn't change
                        if current_action != best_action:
                                stable_policy = True
                                # Greedy policy update
                                policy[state] = np.eye(environment.nA)[best_action]
                evaluated_policies += 1
                # If the algorithm converged and policy is not changing anymore, then return final policy and value function
                if stable_policy:
                        print(f'Evaluated {evaluated_policies} policies.')
                        return policy, V


environment = gym.make('FrozenLake-v1')
# Search for an optimal policy using policy iteration
policy, V = policy_iteration(environment.env)





"""import operator
import pandas as pd
if  tuple(map(operator.add, (1,0), (-1,0)))==(0,0):
    print('here')

arr = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
print(pd.DataFrame(arr))
print(pd.DataFrame(arr).shape)"""

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