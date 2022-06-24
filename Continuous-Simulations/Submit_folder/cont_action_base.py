# import torch
import random

import numpy as np
import pygame
# from collections import deque
from pygame_env import Environment, StaticObstacle, Robot, ChargingDock
# from model import LinearQNet, QTrainer
from cont_act_control_v4 import direction_control

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

charging_dock = ChargingDock((25, 554), (50, 50), [all_sprites])
# init robot object. First 3 inputs are pygame stuff
robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=0.1, battery_drain_l=2, speed=35)

env = Environment(robot, [obs1, obs2, obs3, obs4, obs5, obs6], charging_dock,all_sprites, collision_sprites, screen, copy_sprites)

simulation_count = 0
total_score = 0
record = 0
eff_record = 0
move_range = np.arange(-1, 1, 0.1)

plot_scores = []
plot_mean_scores = []

copy = True

iter = 0
tot_reward = 0
loops = 0
while True:
    loops+=1

    # tests the copy robot. switches between copy robot and original robot
    # copy robot will be seen moving, but when copy is set to true, original robot will keep moving from where it left off.

    #initializes the control class
    dc = direction_control(env,alpha=1,gamma=0.8,neighbors=4,size_rand=15,step_size=1,further_step=2,mode=-1)
    #runs the code to generate a vector
    move_x,move_y=dc.generate_vector()
    #gets the reward
    reward, done, score, efficiency,_ = env.cont_step(move_x, move_y, copy)
    tot_reward+=reward
    print(f'Reward: {reward}\tReward_Step: {tot_reward/loops}\tScore: {score}\tEfficiency: {efficiency}')

    if not copy:
        env.revert_copy()

    iter += 1
    if done:
        #resets the environment
        env.reset()
        

        