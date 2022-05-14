print('hi')
"""
import numpy as np
import gym

# FrozenLake-v0 gym environment
env = gym.make('FrozenLake-v0')

# Parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.05
gamma = 0.95

print('Made it')
  
#Initializing the Q-vaue
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Function to choose the next action with episolon greedy
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action
    
#Initializing the reward
reward=0
  
# Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)
  
    while t < max_steps:
        # Visualizing the training
#         env.render()
          
        # Getting the next state
        state2, reward, done, info = env.step(action1)
  
        #Choosing the next action
        action2 = choose_action(state2)
          
        #Learning the Q-value
        Q[state1, action1] = Q[state1, action1] + alpha * (reward + gamma * Q[state2, action2] - Q[state1, action1])
  
        state1 = state2
        action1 = action2
          
        #Updating the respective vaLues
        t += 1
        reward += 1
          
        #If at the end of learning process
        if done:
            break
            
#Evaluating the performance
print ("Performace : ", reward/total_episodes)
  
#Visualizing the Q-matrix
print(Q)

    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    possible_tiles = robot.possible_tiles_after_move()
    # Get rid of any tiles outside a 1 step range (we don't care about our vision for this algorithm):
    possible_tiles = {move:possible_tiles[move] for move in possible_tiles if abs(move[0]) < 2 and abs(move[1]) < 2}
    if 1.0 in list(possible_tiles.values()) or 2.0 in list(possible_tiles.values()):
        # If we can reach a goal tile this move:
        if 2.0 in list(possible_tiles.values()):
            move = list(possible_tiles.keys())[list(possible_tiles.values()).index(2.0)]
        # If we can reach a dirty tile this move:
        elif 1.0 in list(possible_tiles.values()):
            # Find the move that makes us reach the dirty tile:
            move = list(possible_tiles.keys())[list(possible_tiles.values()).index(1.0)]
        else:
            assert False
        # Find out how we should orient ourselves:
        new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
        # Orient ourselves towards the dirty tile:  
        while new_orient != robot.orientation:
            # If we don't have the wanted orientation, rotate clockwise until we do:
            # print('Rotating right once.')
            robot.rotate('r')
        # Move:
        robot.move()
    # If we cannot reach a dirty tile:
    else:
        # If we can no longer move:
        while not robot.move():
            # Check if we died to avoid endless looping:
            if not robot.alive:
                break
            # Decide randomly how often we want to rotate:
            times = random.randrange(1, 4)
            # Decide randomly in which direction we rotate:
            if random.randrange(0, 2) == 0:
                # print(f'Rotating right, {times} times.')
                for k in range(times):
                    robot.rotate('r')
            else:
                # print(f'Rotating left, {times} times.')
                for k in range(times):
                    robot.rotate('l')
    #print('Historic coordinates:', [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])])
"""