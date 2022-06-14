# import torch
import random

import numpy
import numpy as np
import pygame
# from collections import deque
from pygame_env import Environment, StaticObstacle, Robot
# from model import LinearQNet, QTrainer
from plotter import plot

screen = pygame.display.set_mode((800, 600))

all_sprites = pygame.sprite.Group()
copy_sprites = pygame.sprite.Group()

collision_sprites = pygame.sprite.Group()

# obstacle setup, random generation will be implemented
obs1 = StaticObstacle(pos=(100, 500), size=(100, 50), groups=[all_sprites, copy_sprites, collision_sprites])
obs2 = StaticObstacle((400, 400), (100, 200), [all_sprites, copy_sprites, collision_sprites])
obs3 = StaticObstacle((200, 200), (200, 100), [all_sprites, copy_sprites, collision_sprites])
obs4 = StaticObstacle((300, 100), (200, 300), [all_sprites, copy_sprites, collision_sprites])
obs5 = StaticObstacle((1, 1), (200, 100), [all_sprites, copy_sprites, collision_sprites])
obs6 = StaticObstacle((700, 1), (50, 400), [all_sprites, copy_sprites, collision_sprites])

# init robot object. First 3 inputs are pygame stuff
robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=0.1, battery_drain_l=2, speed=40)

env = Environment(robot, [obs1, obs2, obs3, obs4, obs5, obs6], all_sprites, collision_sprites, screen, copy_sprites)

simulation_count = 0
total_score = 0
record = 0
eff_record = 0
move_range = numpy.arange(-1, 1, 0.1)

plot_scores = []
plot_mean_scores = []

copy = False

iter = 0
while True:

    # move_x = random.choice([-1, 0, 1])
    # move_y = random.choice([-1, 0, 1])
    move_x = random.choice(move_range)
    move_y = random.choice(move_range)

    # tests the copy robot. switches between copy robot and original robot
    # copy robot will be seen moving, but when copy is set to false, original robot will keep moving from where it left off.
    if iter % 50 == 0:
        copy = not copy

    reward, done, score, efficiency = env.cont_step(move_x, move_y, copy)
    iter += 1
    if done:
        env.reset()
        print("Eff: ", efficiency)
        if efficiency > eff_record:
            eff_record = efficiency

        if score > record:
            record = score

        simulation_count += 1

        print('Game', simulation_count, 'Score', score, 'Record:', record, "Eff: ", efficiency, "Eff Record: ",
              eff_record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / simulation_count
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
