
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

for rep in range(0,1):
    #Generates the room, robot and environment
    room_types = ["house", "triangle", "corridor", "moving"]
    room_name = room_types[0]
    room = generate_room(all_sprites,collision_sprites,screen,'house')
    robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=1, battery_drain_l=2, speed=35)
    env = Environment(robot, room, charging_dock,all_sprites, collision_sprites, screen, copy_sprites)

    tot_reward = 0
    loops = 0
    it = 0
    while True:
        loops+=1
        

        # tests the copy robot. switches between copy robot and original robot
        # copy robot will be seen moving, but when copy is set to true, original robot will keep moving from where it left off.


        #dc = cont_act_control_v2.direction_control(env,alpha=1,gamma=0.8,neighbors=5,size_rand=50,step_size=1,further_step=2,mode=-1)
        alpha=1;gamma=1;neighbors=1;size_rand=15;step_size=1;further_step=2;mode=-2
        #initializes control class
        dc = direction_control(env,alpha=alpha,gamma=gamma,neighbors=neighbors,size_rand=size_rand,step_size=step_size,further_step=further_step,mode=mode)
        #generates vector
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
            if it < 4:
                #appends information to dataframe
                df_final.loc[it-1 + rep*4] = [room_name,tot_reward,loops,tot_reward/loops,score,efficiency,bp,size_rand,neighbors,alpha,gamma,mode,rep]
                #reinitializes the environment
                room_name = room_types[it]
                room = generate_room(all_sprites,collision_sprites,screen,room_name)
                robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=1, battery_drain_l=2, speed=35)
                env = Environment(robot, room, charging_dock,all_sprites, collision_sprites, screen, copy_sprites)
                tot_reward = 0
                loops = 0
            else:
                df_final.loc[len(df_final.index)] = [room_name,tot_reward,loops,tot_reward/loops,score,efficiency,bp,size_rand,neighbors,alpha,gamma,mode,rep]
                tot_reward = 0
                loops = 0
                break
            env.reset()
            
"""print("Eff: ", efficiency)
                if efficiency > eff_record:
                    eff_record = efficiency

                room = generate_room(all_sprites,collision_sprites,screen,'triangle')
                robot = Robot(all_sprites, collision_sprites, screen, battery_drain_p=0.1, battery_drain_l=2, speed=40)
                env = Environment(robot, room, charging_dock,all_sprites, collision_sprites, screen, copy_sprites)

                if score > record:
                    record = score

                simulation_count += 1

                print('Game', simulation_count, 'Score', score, 'Record:', record, "Eff: ", efficiency, "Eff Record: ",
                    eff_record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / simulation_count
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)"""

df_final.to_csv('Data/df_final_greedy.csv')