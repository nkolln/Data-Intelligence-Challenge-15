import numpy as np


# returns a random policy dictionary with each action having equal probabilities
def init_random_pi(robot, grid: np.ndarray):
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
def init_Q(robot, grid: np.ndarray, policy: dict):
    # possible actions
    moves = ['n', 'e', 's', 'w']
    moves_actual = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    Q = {}
    # initialize every state action pair
    for state in policy.keys():
        Q[state] = {action: 0.0 for action in range(len(moves))}

    return Q


def robot_epoch(robot):
    pass
