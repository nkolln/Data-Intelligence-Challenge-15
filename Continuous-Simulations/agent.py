from copy import deepcopy
from pickletools import optimize

import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import pygame
import copy
from collections import deque
from pygame_env import Environment, StaticObstacle, Robot, MovingHorizontalObstacle, MovingVerticalObstacle, \
    ChargingDock, random_obstacles, room_types, generate_room
from model import LinearQNet, QTrainer
from plotter import plot

MAX_MEMORY = 5_000_000
BATCH_SIZE = 500
LR = 0.001


# np.set_printoptions(threshold=sys.maxsize)

class Agent:

    def __init__(self, model, lr, optimizer, criterion):
        self.simulation_count = 0  # total number of simulation ran
        self.epsilon = 0  # randomness
        self.gamma = 0.6  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # deque allows easy popping from the start if memory get too large
        self.model = model
        self.trainer = QTrainer(model, lr, self.gamma, optimizer, criterion)

    def get_state(self, simulation: Environment):
        robot = simulation.robot

        # points to check if there is an obstacle or a wall. Currently, it sees as far as its size (33,33)
        # 66 is because x and y points are the center of the robot, so it has to use double its size
        left_vision = (robot.rect.x - 66, robot.rect.y)
        right_vision = (robot.rect.x + 66, robot.rect.y)
        up_vision = (robot.rect.x, robot.rect.y - 66)
        down_vision = (robot.rect.x, robot.rect.y + 66)

        # current direction of the robot
        dir_left = robot.move_left
        dir_right = robot.move_right
        dir_up = robot.move_up
        dir_down = robot.move_down

        # create the state representation from the vision, direction, and location cleanliness
        state = [
            # check obstacle or wall straight ahead
            (dir_right and simulation.is_obstacle(right_vision)) or
            (dir_left and simulation.is_obstacle(left_vision)) or
            (dir_up and simulation.is_obstacle(up_vision)) or
            (dir_down and simulation.is_obstacle(down_vision)),

            # check obstacle or wall on right
            (dir_up and simulation.is_obstacle(right_vision)) or
            (dir_down and simulation.is_obstacle(left_vision)) or
            (dir_left and simulation.is_obstacle(up_vision)) or
            (dir_right and simulation.is_obstacle(down_vision)),

            # check obstacle or wall on left
            (dir_down and simulation.is_obstacle(right_vision)) or
            (dir_up and simulation.is_obstacle(left_vision)) or
            (dir_right and simulation.is_obstacle(up_vision)) or
            (dir_left and simulation.is_obstacle(down_vision)),

            # current Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # check if current location is dirty
            simulation.is_robot_location_dirty(),
            # check if robot vicinity is dirty
            simulation.is_robot_vicinity_dirty("up"),
            simulation.is_robot_vicinity_dirty("down"),
            simulation.is_robot_vicinity_dirty("right"),
            simulation.is_robot_vicinity_dirty("left"),
            # check if robot battery is low
            simulation.is_robot_battery_low()
        ]
        # return the state as an array that only contains 0s and 1s
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done))  # automatically pops left if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 20 - self.simulation_count

        final_move = [0, 0, 0, 0, 0, 0, 0, 0]

        if random.randint(0, 130) < self.epsilon:
            move = random.randint(0, 7)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        # print(final_move)
        return final_move


# End of Agent class


def train(model, lr, optimizer, criterion, room_name, plot_count = None, simulationnr_stop = None):
    plot_scores = []
    plot_mean_scores = [] 
    config_list = [str(lr), optimizer.__class__.__name__, criterion.__class__.__name__ , room_name] #learning rate, optimizer, criterion, roomtype
    total_score = 0
    record = 0
    eff_record = 0
    
    agent = Agent(model, lr, optimizer, criterion)

    screen = pygame.display.set_mode((800, 600))

    all_sprites = pygame.sprite.Group()
    collision_sprites = pygame.sprite.Group()

    room = generate_room(all_sprites, collision_sprites, screen, room_name)
    charging_dock = ChargingDock((25, 554), (50, 50), [all_sprites])
    robot = Robot(all_sprites, collision_sprites, screen, 0.1, 5, 20)

    game = Environment(robot, room, charging_dock, all_sprites, collision_sprites, screen)

    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)

        previous_matrix = copy.deepcopy(game.matrix)
        # print(previous_matrix.shape)
        # perform move and get new state
        reward, done, score, efficiency = game.discrete_step(final_move)
        state_new = agent.get_state(game)
        # print(game.robot_location())

        # diff_matrix = np.subtract(current_matrix, previous_matrix)
        # print("previous: " , previous_matrix)
        # print("current: " , current_matrix)
        # print("diff: " , np.count_nonzero(diff_matrix == 1))
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # prev = curr
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result

            # room = generate_room(all_sprites, collision_sprites, screen, random.choice(room_types))
            # robot = Robot(all_sprites, collision_sprites, screen, 0.1, 5, 20)
            # charging_dock = ChargingDock((25, 554), (50, 50), [all_sprites])

            # game.obstacles = room
            # game.robot = robot
            # game.charging_dock = charging_dock
            game.reset()
            agent.simulation_count += 1
            agent.train_long_memory()

            if efficiency > eff_record:
                eff_record = efficiency

            if score > record:
                record = score
                model.save()

            print('Game', agent.simulation_count, 'Score', score, 'Record:', record, "Eff: ", efficiency, "Eff record:",
                  eff_record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.simulation_count
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, config_list, plot_count, simulationnr_stop)

        if simulationnr_stop != None and agent.simulation_count == simulationnr_stop:
            break

# if __name__ == '__main__':
#     lr = LR
#     model = LinearQNet(13, 512, 256, 8)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     train(model, lr, optimizer, criterion)
