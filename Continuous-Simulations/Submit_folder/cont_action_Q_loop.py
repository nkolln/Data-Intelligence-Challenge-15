
# import torch
import random

import pandas as pd
import numpy
import numpy as np
import pygame
# from collections import deque
from pygame_env import Environment, StaticObstacle, Robot, ChargingDock,generate_room
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

room = generate_room(all_sprites,collision_sprites,screen,'house')
robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=1, battery_drain_l=2, speed=35)
env = Environment(robot, room, charging_dock,all_sprites, collision_sprites, screen, copy_sprites)
#env = Environment(robot, [obs1, obs2, obs3, obs4, obs5, obs6], charging_dock,all_sprites, collision_sprites, screen, copy_sprites)

simulation_count = 0
total_score = 0
record = 0
eff_record = 0

plot_scores = []
plot_mean_scores = []

df_final = pd.DataFrame(columns = ['Room_Name','Tot_reward','Iterations','Rps','Score','Efficiency','Final_battery','Size_rand','Neighbors','Alpha','Gamma','Mode','Repeated'])

copy = True
count_its=0
for rep in range(0,1):
    for alpha in [1]:
        g_count = 0
        for gamma in [1]:
        #for gamma in [0.2,0.5,0.8,1]:
            room_types = ["house", "triangle", "corridor", "moving"]
            room_name = room_types[0]

            room = generate_room(all_sprites,collision_sprites,screen,room_name)
            robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=1, battery_drain_l=2, speed=35)
            env = Environment(robot, room, charging_dock,all_sprites, collision_sprites, screen, copy_sprites)

            tot_reward = 0
            loops = 0
            it = 0
            while True:
                loops+=1
                if room_name == 'triangle':
                    gamma = 0.8

                # tests the copy robot. switches between copy robot and original robot
                # copy robot will be seen moving, but when copy is set to true, original robot will keep moving from where it left off.


                neighbors=4;size_rand=15;step_size=1;further_step=2;mode=-1
                #initializes control class
                dc = direction_control(env,alpha=alpha,gamma=gamma,neighbors=neighbors,size_rand=size_rand,step_size=step_size,further_step=further_step,mode=mode)
                #Generates vector
                move_x,move_y=dc.generate_vector()
                reward, done, score, efficiency,_ = env.cont_step(move_x, move_y, copy)
                tot_reward+=reward
                print(f'Reward: {reward}\tReward_Step: {tot_reward/loops}\tScore: {score}\tEfficiency: {efficiency}\tBattery: {robot.battery_percentage}')
                bp = robot.battery_percentage
                if not copy:
                    env.revert_copy()
                # print("Eff: ", efficiency)
                if done:
                    it+=1
                    count_its+=1
                    if it < 4:
                        #saves data
                        df_final.loc[len(df_final.index)] = [room_name,tot_reward,loops,tot_reward/loops,score,efficiency,bp,size_rand,neighbors,alpha,gamma,mode,rep]
                        #Regenerates the new room
                        room_name = room_types[it]
                        room = generate_room(all_sprites,collision_sprites,screen,room_name)
                        robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=1, battery_drain_l=2, speed=35)
                        env = Environment(robot, room, charging_dock,all_sprites, collision_sprites, screen, copy_sprites)
                        tot_reward=0
                        loops = 0
                        #count_its+=1
                    else:
                        df_final.loc[len(df_final.index)] = [room_name,tot_reward,loops,tot_reward/loops,score,efficiency,bp,size_rand,neighbors,alpha,gamma,mode,rep]
                        break
                    
                        
                    env.reset()

    df_final.to_csv(f'Data/df_final_q_cont{str(rep)}.csv')