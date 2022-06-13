import torch
import random
import numpy as np
import pygame
from collections import deque
from pygame_env import Environment, StaticObstacle, Robot, ChargingDock
from model import LinearQNet, QTrainer
from plotter import plot

MAX_MEMORY = 1_000_000
BATCH_SIZE = 500
LR = 0.003


class Agent:

    def __init__(self):
        self.simulation_count = 0  # total number of simulation ran
        self.epsilon = 0  # randomness
        self.gamma = 0.6  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # deque allows easy popping from the start if memory get too large

        self.model = LinearQNet(12, 1024, 8)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, simulation: Environment):
        robot = simulation.robot

        # points to check if there is an obstacle or a wall. Currently, it sees as far as its size (33,33)
        # 66 is because x and y points are the center of the robot, so it has to use double its size
        left_vision = (robot.rect.x-66, robot.rect.y)
        right_vision = (robot.rect.x+66, robot.rect.y)
        up_vision = (robot.rect.x, robot.rect.y-66)
        down_vision = (robot.rect.x, robot.rect.y+66)

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
            simulation.is_robot_vicinity_dirty("up"),
            simulation.is_robot_vicinity_dirty("down"),
            simulation.is_robot_vicinity_dirty("right"),
            simulation.is_robot_vicinity_dirty("left"),
        ]
        # return the state as an array that only contains 0s and 1s
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # automatically pops left if MAX_MEMORY is reached

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
        self.epsilon = 80 - self.simulation_count
        final_move = [0, 0, 0, 0,0,0,0,0]

        if random.randint(0, 200) < self.epsilon:
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

    # obstacle setup, random generation will be implemented
    obs1 = StaticObstacle(pos=(100, 500), size=(100, 50), groups=[all_sprites, collision_sprites])
    obs2 = StaticObstacle((400, 400), (100, 200), [all_sprites, collision_sprites])
    obs3 = StaticObstacle((200, 200), (200, 100), [all_sprites, collision_sprites])
    obs4 = StaticObstacle((300, 100), (200, 300), [all_sprites, collision_sprites])
    obs5 = StaticObstacle((1, 1), (200, 100), [all_sprites, collision_sprites])
    obs6 = StaticObstacle((700, 1), (50, 400), [all_sprites, collision_sprites])
    charging_dock = ChargingDock((25, 554), (50,50), [all_sprites])
    
    robot = Robot(all_sprites, collision_sprites, charging_dock, screen, 0.09, 0.5, 50)
    
    game = Environment(robot, [obs1, obs2, obs3, obs4, obs5, obs6], all_sprites, collision_sprites, screen)

    while True:
        #print("dock pos: " ,charging_dock.pos)
        #print("dock size: " , charging_dock.size)
        print("robot pos: " , robot.pos)
        print("battery percentage", robot.battery_percentage)
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score, efficiency = game.discrete_step(final_move)
        # print(reward)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        #print("Battery: " , robot.battery_percentage)
        #print("Clean: " , game.clean_percentage)
        
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

            print('Game', agent.simulation_count, 'Score', score, 'Record:', record, "Eff: ", efficiency)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.simulation_count
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
