import copy
import operator
import random

import numpy as np
from statistics import mean

def convert_cell_to_state(cell: tuple, grid: np.ndarray):
    shape = grid.shape
    column_count = shape[1]
    state = (cell[1]) + (cell[0]*column_count)
    return state


# grid = np.zeros((5, 5))
# print(grid)
# for row in range(5):
#     for column in range(5):
#         grid[row, column] = convert_cell_to_state((row, column), grid)
# print(grid)


# returns a random policy dictionary with each action having equal probabilities
def init_random_pi(robot):
    grid = robot.grid.cells
    # possible actions
    moves = ['n', 'e', 's', 'w']
    moves_actual = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # initialize policy dict
    pi = {}
    # initialize each state in the grid with equal probability actions
    for state in range(grid.size):
        p = {}

        for action in moves_actual:
            # assign equal probability to all actions
            p[action] = 1 / len(moves_actual)

        pi[state] = p
    return pi


# returns an empty state-action dictionary Q(s,a)
def init_Q(policy: dict):
    # possible actions
    moves = ['n', 'e', 's', 'w']
    moves_actual = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    Q = {}
    # initialize every state action pair
    for state in policy.keys():
        Q[state] = {action: 0.0 for action in moves_actual}

    return Q


def turn_and_move(robot, move: tuple):
    moves = ['n', 'e', 's', 'w']
    moves_actual = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    current_world = robot.grid.cells
    current_pos = robot.pos
    shape_w = current_world.shape
    rewards = np.copy(current_world)
    # current_world[current_pos] = 0

    # bounds = [[current_pos[0], current_pos[0]+shape_w[0]],[current_pos[1], current_pos[1]+shape_w[1]]]
    valid_states = []
    lst_end = []
    lst_goal = []
    lst_taboo = []
    lst_wall = []
    all_states = []
    lst_clean = []
    lst_dirty = []
    dct_lsts = {-2: lst_taboo, -1: lst_wall, 0: lst_clean, 1: lst_dirty, 2: lst_goal, 3: lst_end}

    # Categorizes every state
    for x in range(shape_w[0]):
        for y in range(shape_w[1]):
            cur_val = current_world[x][y]
            for key, lst in dct_lsts.items():

                if cur_val == key:
                    lst.append((x, y))
    all_states.extend(lst_clean)
    all_states.extend(lst_dirty)
    all_states.extend(lst_goal)
    valid_states.extend(all_states)
    all_states.extend(lst_end)

    for coord in valid_states:
        lst_dir = []
        lst_neighbor = []
        count_walls = 0
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
        if count_walls == 2 and tuple(map(operator.add, lst_neighbor[0], lst_neighbor[1])) == (
        0, 0):  # If there are two walls next to eachother than treat as door
            rewards[coord] -= 1
            # print('-'*20)
        # If it has no dirty tiles around, prioritize
        if count_neighbor == 4:
            rewards[coord] += 20
        # if len(lst_dir) > 0:
        #     actions.update({coord: lst_dir})

    statistics = {"clean": 0,
                  "dirty": 0,
                  "goal": 0}
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()
    reward = rewards[robot.pos]
    if not robot.alive:
        return robot, reward, statistics, True
    # Calculate some statistics:
    clean = (current_world == 0).sum()
    dirty = (current_world >= 1).sum()
    goal = (current_world == 2).sum()
    statistics.update({"clean": clean,
                       "dirty": dirty,
                       "goal": goal})
    # Calculate the cleaned percentage:
    clean_percent = (clean / (dirty + clean)) * 100
    # See if the room can be considered clean, if so, stop the simulaiton instance:
    if clean_percent >= 100 and goal == 0:
        return robot, reward, statistics, True

    return robot, reward, statistics, False


def calculate_episode_efficiency(robot):
    n_total_tiles = ((robot.grid.cells >= 0) & (robot.grid.cells < 3)).sum()
    moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
    u_moves = set(moves)
    n_revisted_tiles = len(moves) - len(u_moves)
    efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
    return efficiency

def run_episode(robot, policy: dict):
    grid = robot.grid.cells
    episode = []
    done = False

    while not done:
        # state-action-reward for a step
        sar = []
        state = convert_cell_to_state(robot.pos, grid)
        sar.append(state)

        n = random.uniform(0, sum(policy[state].values()))
        top_range = 0
        for probability in policy[state].items():
            top_range += probability[1]
            if n < top_range:
                action = probability[0]
                break

        robot, reward, statistics, done = turn_and_move(robot, action)

        sar.append(action)
        sar.append(reward)
        episode.append(sar)
    efficiency = calculate_episode_efficiency(robot)
    return episode, efficiency


def test_policy(robot, policy, iter_count):
    wins = 0,
    efficiencies = []
    for i in range(iter_count):
        print("iter ", i)
        episode, efficiency = run_episode(robot, policy)
        efficiencies.append(efficiency)

    return mean(efficiencies)


def train_monte_carlo_e_soft(robot, episodes=100, policy=None, epsilon=0.01):
    if not policy:
        policy = init_random_pi(robot)  # Create an empty dictionary to store state action values
    Q = init_Q(policy)  # Empty dictionary for storing rewards for each state-action pair
    returns = {}  # 3.

    for _ in range(episodes):  # Looping through episodes
        test_robot = copy.deepcopy(robot)
        print("episode ", _)
        G = 0  # Store cumulative reward in G (initialized at 0)
        episode, efficiency = run_episode(robot=test_robot, policy=policy)  # Store state, action and value respectively

        # for loop through reversed indices of episode array.
        # The logic behind it being reversed is that the eventual reward would be at the end.
        # So we have to go back from the last timestep to the first one propagating result from the future.

        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t  # Increment total reward by reward on current timestep

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:  #
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action])  # Average reward across episodes

                Q_list = list(map(lambda x: x[1], Q[s_t].items()))  # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)

                A_star = max_Q  # 14.

                for a in policy[s_t].items():  # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))
    print("training done.")
    return policy




# def robot_epoch(robot):
#     pass
