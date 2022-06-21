from agent import train
from model import LinearQNet
from qlearning_discr import robot_epoch
import torch.optim as optim
import torch.nn as nn
from pygame_env import room_types
import pandas as pd

alphas = [0.3, 0.7]
epsilons = [0.3, 0.7]
gammas = [0.3, 0.7]

df = pd.DataFrame(data=np.zeros((24,6)), columns=["alpha", "epsilon", "gamma","room","cleaned","efficiency"])
i = 0

for room in room_types[0:]:
    for a in alphas:
        for g in gammas:
            for e in epsilons:
                screen = pygame.display.set_mode((800, 600))

                all_sprites = pygame.sprite.Group()
                collision_sprites = pygame.sprite.Group()

                room = generate_room(all_sprites, collision_sprites, screen, room_name)
                charging_dock = ChargingDock((25, 554), (50, 50), [all_sprites])
                robot = Robot(all_sprites, collision_sprites, screen, 0.1, 5, 20)

                game = Environment(robot, room, charging_dock, all_sprites, collision_sprites, screen)

                clean_percent = 0
                battery_percent = 100
                while clean_percent < 100 and battery_percent > 0:
                    robot_epoch(game, a, g, e)
                    clean_percent = game.calc_clean_percentage()
                    battery_percent = game.robot.battery_percentage

                df['cleaned'][i] = clean_percent
                efficiency = game.calculate_efficiency()
                df['efficiency'][i] = efficiency
                df['room'][i] = room
                df['epsilon'][i] = e
                df['alpha'][i] = a
                df['gamma'][i] = g

                i += 1
                df.to_csv(PATH, index=False)

                #print("Alpha is ", lr, "Gamma is", g, "Epsilon is", e, "room type is ", room)
