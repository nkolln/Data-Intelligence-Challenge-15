import copy
import operator
import random
import numpy as np
from statistics import mean


# converts the indices of a cell to its order number. e.g. in a 5x5 grid, (0,0) is 0th state and (1,0) is the 5th state
def convert_cell_to_state(cell: tuple, grid: np.ndarray):
    shape = grid.shape
    column_count = shape[1]
    state = (cell[1]) + (cell[0]*column_count)
    return state


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

# calculates the current rewards for all the states and makes the given move
def turn_and_move(robot, move: tuple):
    moves = ['n', 'e', 's', 'w']
    moves_actual = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    current_world = robot.grid.cells
    current_pos = robot.pos
    shape_w = current_world.shape
    rewards = np.copy(current_world)

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

            dir = moves_actual[i]
            new_loc = tuple(map(operator.add, coord, dir))
            if new_loc in valid_states:
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
                if new_loc in lst_taboo:  # Logic to find if there is a doorway
                    count_walls += 1
                    lst_neighbor.append(dir)
        if count_walls == 2 and tuple(map(operator.add, lst_neighbor[0], lst_neighbor[1])) == (
        0, 0):  # If there are two walls next to each other then treat as door
            rewards[coord] -= 1
        # If it has no dirty tiles around, prioritize
        if count_neighbor == 4:
            rewards[coord] += 20

    statistics = {"clean": 0,
                  "dirty": 0,
                  "goal": 0}
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        robot.rotate('r')
    # Move:
    robot.move()
    reward = rewards[robot.pos]

    # Calculate some statistics:
    clean = (current_world == 0).sum()
    dirty = (current_world >= 1).sum()
    goal = (current_world == 2).sum()
    statistics.update({"clean": clean,
                       "dirty": dirty,
                       "goal": goal})
    # Calculate the cleaned percentage:
    clean_percent = (clean / (dirty + clean)) * 100
    # return if robot died
    if not robot.alive:
        return robot, reward, clean_percent, True
    # See if the room can be considered clean, if so, end the episode:
    if clean_percent >= 100 and goal == 0:
        return robot, reward, clean_percent, True

    return robot, reward, clean_percent, False


# calculates the efficiency of an episode by looking at the revisited tiles
def calculate_episode_efficiency(robot):
    n_total_tiles = ((robot.grid.cells >= 0) & (robot.grid.cells < 3)).sum()
    moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
    u_moves = set(moves)  # unique moves
    n_revisited_tiles = len(moves) - len(u_moves)
    efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisited_tiles)
    return efficiency


#  runs a single episode with the given policy
def run_episode(robot, policy: dict):
    grid = robot.grid.cells
    episode = []
    done = False
    clean_percent = 0
    while not done:
        # state-action-reward for a step
        sar = []
        state = convert_cell_to_state(robot.pos, grid)
        sar.append(state)

        n = random.uniform(0, sum(policy[state].values()))
        top = 0
        # select an action to make
        for probability in policy[state].items():
            top += probability[1]
            if n < top:
                action = probability[0]
                break
        # make a step and get the reward for it
        robot, reward, clean_percent, done = turn_and_move(robot, action)

        sar.append(action)
        sar.append(reward)
        episode.append(sar)
    efficiency = calculate_episode_efficiency(robot)
    return episode, efficiency, clean_percent


#  runs the given policy iter_count times to get the average efficiency
def test_pi(robot, policy, iter_count):
    efficiencies = []
    for i in range(iter_count):
        print("iter ", i)
        episode, efficiency, clean = run_episode(robot, policy)
        efficiencies.append(efficiency)

    return mean(efficiencies)

# returns the indices of all the dirty cells of the grid as tuples
def get_dirty_cells(robot):
    grid = robot.grid.cells
    dirty_tiles = np.argwhere(grid >= 0)
    dirty_tiles_tuples = []
    for i in range(len(dirty_tiles)):
        dirty_tiles_tuples.append(tuple(dirty_tiles[i]))
    # print(type(dirty_tiles_tuples[0]))
    return dirty_tiles_tuples


def train(robot, episodes=100, policy=None, epsilon=0.9):
    if not policy:
        policy = init_random_pi(robot)  # Create an empty dictionary to store state action values
    Q = init_Q(policy)  # Empty dictionary for storing rewards for each state-action pair
    returns = {}

    for _ in range(episodes):  # Looping through episodes
        test_robot = copy.deepcopy(robot) # work on a copy of the robot and the grid

        dirty_tiles = get_dirty_cells(test_robot)  # get the dirty cells in the grid
        test_robot.pos = random.choice(dirty_tiles)  # choose a random starting position to increase exploration

        G = 0  # cumulative reward
        episode, efficiency, clean = run_episode(robot=test_robot, policy=policy)   # run 1 episode
        print("episode ", _, "efficiency: ", efficiency, " cleaned: ", clean, "%")

        #loop through the episode in reverse order
        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t  # Increment total reward by reward on current step

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]

                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action])  # Average reward across episodes

                Q_list = list(map(lambda x: x[1], Q[s_t].items()))  # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)

                A_star = max_Q

                for a in policy[s_t].items():  # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))
    print("training done.")
    return policy




# def robot_epoch(robot):
#     pass
