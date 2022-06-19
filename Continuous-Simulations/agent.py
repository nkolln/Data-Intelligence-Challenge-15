from copy import deepcopy

import torch
import random
import numpy as np
import pygame
import copy
from collections import deque
from pygame_env import Environment, StaticObstacle, Robot, MovingHorizontalObstacle, MovingVerticalObstacle, \
    ChargingDock, random_obstacles
from model import LinearQNet, QTrainer
from plotter import plot

MAX_MEMORY = 5_000_000
BATCH_SIZE = 500
LR = 0.001


# np.set_printoptions(threshold=sys.maxsize)

class Agent:

    def __init__(self):
        self.simulation_count = 0  # total number of simulation ran
        self.epsilon = 0  # randomness
        self.gamma = 0.6  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # deque allows easy popping from the start if memory get too large

        self.model = LinearQNet(13, 512, 256, 8)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    eff_record = 0
    agent = Agent()

    screen = pygame.display.set_mode((800, 600))

    all_sprites = pygame.sprite.Group()
    collision_sprites = pygame.sprite.Group()

    # obstacles = random_obstacles(10, screen, [all_sprites, collision_sprites], 20, 300)

    # triangle room
    # obs1 = StaticObstacle(pos=(0, 0), size=(screen.get_width(), 50), groups=[all_sprites, collision_sprites])
    #
    # obs2 = StaticObstacle(pos=(0, 50), size=(screen.get_width() // 2 - 33, 50), groups=[all_sprites, collision_sprites])
    # obs3 = StaticObstacle(pos=(screen.get_width() // 2 + 33, 50), size=(screen.get_width() // 2 - 33, 50),
    #                       groups=[all_sprites, collision_sprites])
    #
    # obs4 = StaticObstacle(pos=(0, 100), size=(screen.get_width() // 2 - 66, 50),
    #                       groups=[all_sprites, collision_sprites])
    # obs5 = StaticObstacle(pos=(screen.get_width() // 2 + 66, 100), size=(screen.get_width() // 2 - 77 / 2, 50),
    #                       groups=[all_sprites, collision_sprites])
    #
    # obs6 = StaticObstacle(pos=(0, 150), size=(screen.get_width() // 2 - 100, 50),
    #                       groups=[all_sprites, collision_sprites])
    # obs7 = StaticObstacle(pos=(screen.get_width() // 2 + 100, 150), size=(screen.get_width() // 2 - 100, 50),
    #                       groups=[all_sprites, collision_sprites])
    #
    # obs8 = StaticObstacle(pos=(0, 200), size=(screen.get_width() // 2 - 133, 50),
    #                       groups=[all_sprites, collision_sprites])
    # obs9 = StaticObstacle(pos=(screen.get_width() // 2 + 133, 200), size=(screen.get_width() // 2 - 133, 50),
    #                       groups=[all_sprites, collision_sprites])
    #
    # obs10 = StaticObstacle(pos=(0, 250), size=(screen.get_width() // 2 - 166, 50),
    #                        groups=[all_sprites, collision_sprites])
    # obs11 = StaticObstacle(pos=(screen.get_width() // 2 + 166, 250), size=(screen.get_width() // 2 - 166, 50),
    #                        groups=[all_sprites, collision_sprites])
    #
    # obs12 = StaticObstacle(pos=(0, 300), size=(screen.get_width() // 2 - 200, 50),
    #                        groups=[all_sprites, collision_sprites])
    # obs13 = StaticObstacle(pos=(screen.get_width() // 2 + 200, 300), size=(screen.get_width() // 2 - 200, 50),
    #                        groups=[all_sprites, collision_sprites])
    #
    # obs14 = StaticObstacle(pos=(0, 350), size=(screen.get_width() // 2 - 233, 50),
    #                        groups=[all_sprites, collision_sprites])
    # obs15 = StaticObstacle(pos=(screen.get_width() // 2 + 233, 350), size=(screen.get_width() // 2 - 233, 50),
    #                        groups=[all_sprites, collision_sprites])
    #
    # obs16 = StaticObstacle(pos=(0, 400), size=(screen.get_width() // 2 - 266, 50),
    #                        groups=[all_sprites, collision_sprites])
    # obs17 = StaticObstacle(pos=(screen.get_width() // 2 + 266, 400), size=(screen.get_width() // 2 - 266, 50),
    #                        groups=[all_sprites, collision_sprites])
    #
    # obs18 = StaticObstacle(pos=(0, 450), size=(screen.get_width() // 2 - 300, 50),
    #                        groups=[all_sprites, collision_sprites])
    # obs19 = StaticObstacle(pos=(screen.get_width() // 2 + 300, 450), size=(screen.get_width() // 2 - 300, 50),
    #                        groups=[all_sprites, collision_sprites])
    #
    # triangle_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14, obs15, obs16,
    #              obs17, obs18, obs19]

    # # moving obstacle room
    # obs1 = StaticObstacle(pos=(100, 500), size=(100, 50), groups=[all_sprites, collision_sprites])
    # obs2 = StaticObstacle((400, 400), (100, 200), [all_sprites, collision_sprites])
    # obs3 = StaticObstacle((200, 200), (200, 100), [all_sprites, collision_sprites])
    # obs4 = StaticObstacle((300, 100), (200, 300), [all_sprites, collision_sprites])
    # obs5 = StaticObstacle((1, 1), (200, 100), [all_sprites, collision_sprites])
    # obs6 = StaticObstacle((700, 1), (50, 400), [all_sprites, collision_sprites])
    # obs7 = MovingHorizontalObstacle((0, 300), (50, 50), [all_sprites, collision_sprites], max_left=0, max_right=300, speed=5)
    # obs8 = MovingVerticalObstacle((500, 0), (25, 25), [all_sprites, collision_sprites], max_up=0, max_down=500, speed=5)
    #
    # moving_room_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8]

    # # corridor room
    # obs1 = StaticObstacle(pos=(0, 0), size=(25, screen.get_height()-100), groups=[all_sprites, collision_sprites])
    # obs2 = StaticObstacle(pos=(100, 100), size=(25, screen.get_height()), groups=[all_sprites, collision_sprites])
    # obs3 = StaticObstacle(pos=(200, 0), size=(50, 100), groups=[all_sprites, collision_sprites])
    # obs4 = StaticObstacle(pos=(200, 200), size=(50, screen.get_height()-300), groups=[all_sprites, collision_sprites])
    # obs5 = StaticObstacle(pos=(300, 100), size=(50, screen.get_height()- 133), groups=[all_sprites, collision_sprites])
    # obs6 = StaticObstacle(pos=(400, 0), size=(400, 50), groups=[all_sprites, collision_sprites])
    # obs7 = StaticObstacle(pos=(350, 100), size=(400, 50), groups=[all_sprites, collision_sprites])
    # obs8 = StaticObstacle(pos=(450, 200), size=(400, 50), groups=[all_sprites, collision_sprites])
    # obs9 = StaticObstacle(pos=(500, 300), size=(200, 200), groups=[all_sprites, collision_sprites])
    #
    # corridor_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9]

    # full house
    obs1 = StaticObstacle(pos=(0, screen.get_height()//2), size=(screen.get_width()//2-100, 25), groups=[all_sprites, collision_sprites])
    obs2 = StaticObstacle(pos=(screen.get_width()//2-125, screen.get_height()//2+80), size=(25, screen.get_height()), groups=[all_sprites, collision_sprites])

    obs3 = StaticObstacle(pos=(screen.get_width()//2+125, screen.get_height()//2+80), size=(25, screen.get_height()), groups=[all_sprites, collision_sprites])
    obs4 = StaticObstacle(pos=(screen.get_width()//2 + 125, screen.get_height()//2), size=(screen.get_width(), 25), groups=[all_sprites, collision_sprites])

    obs5 = StaticObstacle(pos=(screen.get_width()//2 - 125, 0), size=(25, screen.get_height()//2-125), groups=[all_sprites, collision_sprites])
    obs6 = StaticObstacle(pos=(screen.get_width()//2 - 125, screen.get_height()//2-125), size=(screen.get_width()//3, 25), groups=[all_sprites, collision_sprites])

    obs7 = StaticObstacle(pos=(screen.get_width()//1.5, 0), size=(25, screen.get_height()//2-175), groups=[all_sprites, collision_sprites])
    obs8 = StaticObstacle(pos=(0, screen.get_height()//2-125), size=(200,25), groups=[all_sprites, collision_sprites])

    obs9 = StaticObstacle(pos=(600, 500), size=(100,100), groups=[all_sprites, collision_sprites])

    house_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9]

    charging_dock = ChargingDock((25, 554), (50, 50), [all_sprites])
    robot = Robot(all_sprites, collision_sprites, screen, 0.1, 5, 20)
    game = Environment(robot, house_obstacles, charging_dock, all_sprites, collision_sprites, screen)

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
            game.reset()
            agent.simulation_count += 1
            agent.train_long_memory()

            if efficiency > eff_record:
                eff_record = efficiency

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.simulation_count, 'Score', score, 'Record:', record, "Eff: ", efficiency, "Eff record:",
                  eff_record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.simulation_count
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()