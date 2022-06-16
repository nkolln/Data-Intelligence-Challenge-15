import numpy as np
import pygame, sys, time, random
from typing import List
from copy import deepcopy
from PIL import Image as im


class StaticObstacle(pygame.sprite.Sprite):
    def __init__(self, pos, size, groups):
        super().__init__(groups)
        self.pos = pos
        self.size = size
        self.image = pygame.Surface(size)
        self.image.fill('black')

        self.rect = self.image.get_rect(center=pos)
        self.old_rect = self.rect.copy()


# class VisionLine(pygame.sprite.Sprite):
#
#     def __init__(self, size: tuple, group):
#         super().__init__(group)
#         self.size = size
#         self.image = pygame.Surface(size)
#         self.rect = self.image.get_rect()
#
#         self.pos = pygame.math.Vector2(self.rect.center)


class Robot(pygame.sprite.Sprite):
    def __init__(self, groups, obstacles, screen, battery_drain_p, battery_drain_l, speed):
        super().__init__(groups)
        self.screen = screen
        # image
        self.image = pygame.image.load(r"static/robot_n.png").convert_alpha()
        self.image = pygame.transform.rotozoom(self.image, 0, 0.05)

        self.og_image = self.image.copy()  # to handle image rotation

        # position
        self.rect = self.image.get_rect(topleft=(0, self.screen.get_size()[1] - self.image.get_height()))

        self.old_rect = self.rect

        # movement
        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2((0, -1))  # upwards
        self.speed = speed
        self.obstacles = obstacles

        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False

        self.vision_range = 300
        self.robot_collided = False

        self.battery_percentage = 100
        self.battery_drain_p = battery_drain_p
        self.battery_drain_l = battery_drain_l

    def reset_robot(self):
        self.rect = self.og_image.get_rect(topleft=(0, self.screen.get_size()[1] - self.image.get_height()))
        self.old_rect = self.rect
        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2((0, -1))

        # self.init_vision_lines()
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False
        self.battery_percentage = 100

    # 4 directional action space
    def set_action_4(self, action: List[int]):
        if np.array_equal(action, [1, 0, 0, 0]):
            self.move_left = False
            self.move_right = True
            self.move_up = False
            self.move_down = False
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.move_left = False
            self.move_right = False
            self.move_up = False
            self.move_down = True
        elif np.array_equal(action, [0, 0, 1, 0]):
            self.move_left = True
            self.move_right = False
            self.move_up = False
            self.move_down = False
        else:
            self.move_left = False
            self.move_right = False
            self.move_up = True
            self.move_down = False

    # 8 directional action space
    def set_action_8(self, action: List[int]):
        if np.array_equal(action, [1, 0, 0, 0, 0, 0, 0, 0]):
            self.move_left = False
            self.move_right = True
            self.move_up = False
            self.move_down = False
        elif np.array_equal(action, [0, 1, 0, 0, 0, 0, 0, 0]):
            self.move_left = False
            self.move_right = True
            self.move_up = False
            self.move_down = True
        elif np.array_equal(action, [0, 0, 1, 0, 0, 0, 0, 0]):
            self.move_left = False
            self.move_right = False
            self.move_up = False
            self.move_down = True
        elif np.array_equal(action, [0, 0, 0, 1, 0, 0, 0, 0]):
            self.move_left = True
            self.move_right = False
            self.move_up = False
            self.move_down = True
        elif np.array_equal(action, [0, 0, 0, 0, 1, 0, 0, 0]):
            self.move_left = True
            self.move_right = False
            self.move_up = False
            self.move_down = False
        elif np.array_equal(action, [0, 0, 0, 0, 0, 1, 0, 0]):
            self.move_left = True
            self.move_right = False
            self.move_up = True
            self.move_down = False
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 1, 0]):
            self.move_left = False
            self.move_right = False
            self.move_up = False
            self.move_down = False
        elif np.array_equal(action, [0, 0, 0, 0, 0, 0, 0, 1]):
            self.move_left = False
            self.move_right = True
            self.move_up = True
            self.move_down = False

    def move_rotate(self):

        if self.move_up:
            # print("moving up")
            self.direction.y = -1
            # self.image = pygame.transform.rotate(self.og_image, 0)
            # self.rect = self.image.get_rect(center=self.rect.center)
        elif self.move_down:
            self.direction.y = 1
            # self.image = pygame.transform.rotate(self.og_image, 180)
            # self.rect = self.image.get_rect(center=self.rect.center)
        else:
            self.direction.y = 0

        if self.move_left:
            self.direction.x = -1
            # self.image = pygame.transform.rotate(self.og_image, 90)
            # self.rect = self.image.get_rect(center=self.rect.center)
        elif self.move_right:
            self.direction.x = 1
            # self.image = pygame.transform.rotate(self.og_image, -90)
            # self.rect = self.image.get_rect(center=self.rect.center)
        else:
            self.direction.x = 0

        self.rotate_image()

    # drains the battery by lambda given probability
    def drain_battery(self, dt, is_cont):
        movement_vector = pygame.Vector2(self.direction.x * self.speed * dt, self.direction.y * self.speed * dt).length_squared()

        do_battery_drain = np.random.binomial(1, self.battery_drain_p)
        if do_battery_drain == 1 and self.battery_percentage > 0:
            if movement_vector != 0 and is_cont:
                # movement_vector = movement_vector.normalize()
                self.battery_percentage -= np.random.exponential(self.battery_drain_l) / movement_vector
            else:
                self.battery_percentage -= np.random.exponential(self.battery_drain_l)

    # detects collisions and does not let the robot go into walls
    def collision(self, direction):

        collision_sprites = pygame.sprite.spritecollide(self, self.obstacles, False)
        if collision_sprites:
            self.robot_collided = True
            # print("robot hit something")
            if direction == 'horizontal':
                for sprite in collision_sprites:
                    # collision on the right
                    if self.rect.right >= sprite.rect.left and self.old_rect.right <= sprite.old_rect.left:
                        self.rect.right = sprite.rect.left
                        self.pos.x = self.rect.x

                        # rand_move = random.choice([0, 1])
                        # if rand_move:
                        #     action = [0,0,0,1]
                        # else:
                        #     action = [0,1,0,0]
                        # self.set_action(action)

                    # collision on the left
                    if self.rect.left <= sprite.rect.right and self.old_rect.left >= sprite.old_rect.right:
                        self.rect.left = sprite.rect.right
                        self.pos.x = self.rect.x

                        # rand_move = random.choice([0, 1])
                        # if rand_move:
                        #     action = [0, 0, 0, 1]
                        # else:
                        #     action = [0, 1, 0, 0]
                        # self.set_action(action)

            if direction == 'vertical':
                for sprite in collision_sprites:
                    # collision on the bottom
                    if self.rect.bottom >= sprite.rect.top and self.old_rect.bottom <= sprite.old_rect.top:
                        self.rect.bottom = sprite.rect.top
                        self.pos.y = self.rect.y

                        # rand_move = random.choice([0, 1])
                        # if rand_move:
                        #     action = [1, 0, 0, 0]
                        # else:
                        #     action = [0, 0, 1, 0]
                        # self.set_action(action)

                    # collision on the top
                    if self.rect.top <= sprite.rect.bottom and self.old_rect.top >= sprite.old_rect.bottom:
                        self.rect.top = sprite.rect.bottom
                        self.pos.y = self.rect.y

                        # rand_move = random.choice([0, 1])
                        # if rand_move:
                        #     action = [1, 0, 0, 0]
                        # else:
                        #     action = [0, 0, 1, 0]
                        #
                        # self.set_action(action)
        else:
            self.robot_collided = False

    def window_collision(self, direction):
        if direction == 'horizontal':
            if self.rect.left < 0:
                self.robot_collided = True
                # print("robot hit something")
                self.rect.left = 0
                self.pos.x = self.rect.x
                # self.direction.x *= -1
                # self.move_left = False
                # rand_move = random.choice([0, 1])
                # if rand_move:
                #     self.move_up = True
                # else:
                #     self.move_down = True
                return

            if self.rect.right > self.screen.get_width():
                self.robot_collided = True
                # print("robot hit something")
                self.rect.right = self.screen.get_width()
                self.pos.x = self.rect.x
                # self.direction.x *= -1
                # self.move_right = False
                # rand_move = random.choice([0, 1])
                # if rand_move:
                #     self.move_up = True
                # else:
                #     self.move_down = True
                return

        if direction == 'vertical':
            if self.rect.top < 0:
                self.robot_collided = True
                # print("robot hit something")
                self.rect.top = 0
                self.pos.y = self.rect.y
                # self.direction.y *= -1
                # self.move_up = False
                # rand_move = random.choice([0, 1])
                # if rand_move:
                #     self.move_right = True
                # else:
                #     self.move_left = True
                return

            if self.rect.bottom > self.screen.get_height():
                self.robot_collided = True
                # print("robot hit something")
                self.rect.bottom = self.screen.get_height()
                self.pos.y = self.rect.y
                # self.direction.y *= -1
                # self.move_down = False
                # rand_move = random.choice([0, 1])
                # if rand_move:
                #     self.move_right = True
                # else:
                #     self.move_left = True
                return
            self.robot_collided = False

    def update(self, dt, action=None, x=None, y=None, is_cont=False):
        # if using continuous action space
        if is_cont:
            self.cont_update(dt, x, y)
            return
        self.old_rect = self.rect.copy()
        # self.input()
        self.set_action_8(action)
        # self.set_action_4(action)
        self.move_rotate()
        self.drain_battery(dt, False)
        # print("battery perc: ", self.battery_percentage)

        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()
        # discrete action
        # if x is None:
        self.pos.x += self.direction.x * self.speed * dt
        self.rect.x = round(self.pos.x)
        self.collision('horizontal')
        self.window_collision('horizontal')

        self.pos.y += self.direction.y * self.speed * dt
        self.rect.y = round(self.pos.y)
        self.collision('vertical')
        self.window_collision('vertical')

    # used for continuous action space
    def cont_update(self, dt, x, y):
        # y goes to x and x goes to y because pygame uses y-x matrix smh.
        self.old_rect = self.rect.copy()
        self.direction.x = y
        self.direction.y = x

        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()

        self.drain_battery(dt, True)
        # print("battery perc: ", self.battery_percentage)

        # move and check for collisions on x axis
        self.pos.x += self.direction.x * self.speed * dt
        self.rect.x = round(self.pos.x)
        self.collision('horizontal')
        self.window_collision('horizontal')

        # move and check for collisions on y axis
        self.pos.y += self.direction.y * self.speed * dt
        self.rect.y = round(self.pos.y)
        self.collision('vertical')
        self.window_collision('vertical')

    # rotates the image of the robot its current direction. purely visual, does not affect anything
    def rotate_image(self):
        if self.direction.y == 0:
            if self.direction.x >= 0:  # rotate right
                self.image = pygame.transform.rotate(self.og_image, -90)
            else:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 90)
        elif self.direction.y > 0:
            if self.direction.x == 0:  # rotate right
                self.image = pygame.transform.rotate(self.og_image, 180)
            elif self.direction.x > 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 180+45)

            elif self.direction.x < 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 180 - 45)
        elif self.direction.y < 0:
            if self.direction.x == 0:  # rotate right
                self.image = pygame.transform.rotate(self.og_image, 0)

            elif self.direction.x > 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 0 - 45)

            elif self.direction.x < 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 0 + 45)

class Environment:
    def __init__(self, robot: Robot, obstacles: List[StaticObstacle], all_sprites, collision_sprites, screen):
        self.display_width = screen.get_width()
        self.display_height = screen.get_height()
        pygame.init()
        self.screen = screen

        # group setup
        self.all_sprites = all_sprites
        self.collision_sprites = collision_sprites
        self.robot = robot
        self.trail_lines = []
        self.obstacles = obstacles

        # matrix representation of the display. Used for clean percentage calculation
        self.matrix = np.array([])
        self.init_matrix()
        self.clean_percentage = self.calc_clean_percentage()

        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("Environment")

        self.step_count = 0
        self.repeated_step_count = 0

        self.last_time = time.time()
        self.dt = time.time() - self.last_time

        self.reset()


    # initializes the matrix representation
    def init_matrix(self):
        self.matrix = np.zeros((self.display_height + 1, self.display_width + 1), dtype=int)
        # set the cells corresponding to each obstacle in the matrix to 2
        for obstacle in self.obstacles:
            self.matrix[obstacle.pos[1]:obstacle.pos[1] + obstacle.size[1],
            obstacle.pos[0]:obstacle.pos[0] + obstacle.size[0]] = 2
        # set the cells corresponding to the current robot location to 1 (clean)
        self.matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
        self.robot.rect.topleft[0]:self.robot.rect.topright[0]] = 1

    # sets current robot location to 1 (clean)
    def update_matrix(self):

        self.matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
        self.robot.rect.topleft[0]:self.robot.rect.topright[0]] = 1

        # set obstacle locations to 2 in case robot went through them
        for obstacle in self.obstacles:
            self.matrix[obstacle.pos[1]:obstacle.pos[1] + obstacle.size[1],
            obstacle.pos[0]:obstacle.pos[0] + obstacle.size[0]] = 2

        self.clean_percentage = self.calc_clean_percentage()

    def calc_clean_percentage(self):
        clean_count = np.count_nonzero(self.matrix == 1)
        dirty_count = np.count_nonzero(self.matrix == 0)
        clean_percentage = (clean_count / (clean_count + dirty_count)) * 100
        # print("clean percentage: ", clean_percentage)
        return clean_percentage

    # resets environment so that it can be run again
    def reset(self):
        self.init_matrix()
        self.clean_percentage = self.calc_clean_percentage()
        self.last_time = time.time()
        self.robot.reset_robot()
        self.battery_percentage = 100

        self.trail_lines = []
        self.dt = time.time() - self.last_time
        self.step_count = 0
        self.repeated_step_count = 0

        self.cont_step(0, 0)
        self.all_sprites.draw(self.screen)
        pygame.display.flip()

        # checks if robots current location is clean

    def is_robot_location_dirty(self):
        robot_location = self.matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                         self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
        dirty = np.count_nonzero(robot_location == 0)
        # print("dirty count: ", dirty)
        return dirty > 0

    # checks if up down left right of the robot is dirty
    def is_robot_vicinity_dirty(self, location):
        robot_location = self.matrix[
                         self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                         self.robot.rect.topleft[0]:self.robot.rect.topright[0]]

        next_up_down = self.robot.direction.y * self.robot.speed * self.dt
        next_right_left = self.robot.direction.x * self.robot.speed * self.dt

        if location == "up" and round(self.robot.rect.topleft[1] - next_up_down) >= 0:
            next_location = self.matrix[
                            round(self.robot.rect.topleft[1] - next_up_down):round(self.robot.rect.bottomleft[1] - next_up_down),
                            self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        elif location == "down " and round(self.robot.rect.topleft[1] + next_up_down) < self.display_height:
            next_location = self.matrix[
                            round(self.robot.rect.topleft[1] + next_up_down):round(self.robot.rect.bottomleft[1] + next_up_down),
                            self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        elif location == "right" and round(self.robot.rect.topleft[0] - next_right_left) >= 0:
            next_location = self.matrix[
                            self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                            round(self.robot.rect.topleft[0]-next_right_left):round(self.robot.rect.topright[0]-next_right_left)]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        elif location == "left" and round(self.robot.rect.topleft[0] + next_right_left) < self.display_width:
            next_location = self.matrix[
                            self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                            round(self.robot.rect.topleft[0]+next_right_left):round(self.robot.rect.topright[0]+next_right_left)]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        # if out of bounds
        else:
            return False

    # returns robots center as tuple and whole location as a matrix
    def robot_location(self):
        robot_location = self.matrix[
                         self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                         self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
        robot_center = self.robot.rect.center

        return robot_center, robot_location


    def cont_step(self,x,y,update_matrix= True):

        self.step_count +=1
        # set time passed since last step. Used for smooth movement
        self.dt = time.time() - self.last_time
        self.last_time = time.time()
        step_reward = 0  # reward for the current step
        done = False  # check for simulation end

        # event loop. Basically closes the program when the window is closed. Can also be used to print final results, write data etc.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # data export can be written here
                pygame.quit()
                sys.exit()

        # drawing and updating the screen
        self.screen.fill('lightblue')   # fills background with specified color. Everytihng including this has to be re-drawn at each step smh.
        self.all_sprites.update(self.dt, None, x, y, True)  # calls the update function of the robot and the obstacles with the given parameters. last input says to call const_update

        # reward system
        # if current location is more clean than dirty give low reward, else give high reward
        if self.is_robot_location_dirty():
            step_reward = 20
            # print("location dirty")
        else:
            step_reward = -15
            self.repeated_step_count += 1
            # print("location clean")
        # low reward if robot hit something
        if self.robot.robot_collided:
            step_reward = -5

        # matrix representation of the screen updated or not w.r.t given input
        if update_matrix:
            self.update_matrix()

        # drawing robot and the obstacles to the screen
        self.all_sprites.draw(self.screen)

        # drawing the trail line
        if len(self.trail_lines) <= 0:
            self.trail_lines.append(self.robot.rect.center)
            self.trail_lines.append(self.robot.old_rect.center)
        else:
            self.trail_lines.append(self.robot.rect.center)

        pygame.draw.lines(surface=self.screen, color="red", closed=False, points=self.trail_lines, width=5)

        # update the environment so the changes can be seen on the screen
        pygame.display.update()

        efficiency = self.calculate_efficiency()
        # print("eff: ",efficiency)
        # return done = True if battery is dead or run completed
        if self.clean_percentage >= 100:# or self.robot.battery_percentage <= 1:
            done = True
            return step_reward, done, self.clean_percentage, efficiency
        # return done = False id the simulation is not done
        return step_reward, done, self.clean_percentage, efficiency

    def discrete_step(self, action, x=None, y=None):
        self.step_count += 1
        self.dt = time.time() - self.last_time
        self.last_time = time.time()
        step_reward = 0
        done = False
        # event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # drawing and updating the screen
        self.screen.fill('lightblue')
        self.all_sprites.update(self.dt, action, x, y, False)
        # if current location is more clean than dirty give low reward, else give high reward
        if self.is_robot_location_dirty():
            step_reward = 20
            # print("location dirty")
        else:
            step_reward = -15
            self.repeated_step_count += 1
            # print("location clean")
        # low reward if robot hit something
        if self.robot.robot_collided:
            step_reward += -5

        self.update_matrix()

        self.all_sprites.draw(self.screen)

        # if len(self.robot.vision_lines) <= 0:
        #     self.robot.init_vision_lines()
        # else:
        #     self.robot.update_vision_lines(dt)
        # draw the trail line
        if len(self.trail_lines) <= 0:
            self.trail_lines.append(self.robot.rect.center)
            self.trail_lines.append(self.robot.old_rect.center)
        else:
            self.trail_lines.append(self.robot.rect.center)

        pygame.draw.lines(surface=self.screen, color="red", closed=False, points=self.trail_lines, width=5)

        pygame.display.update()

        efficiency = self.calculate_efficiency()
        # print("eff: ", efficiency)

        if self.clean_percentage >= 100 or self.robot.battery_percentage <= 1:
            done = True
            return step_reward, done, self.clean_percentage, efficiency

        return step_reward, done, self.clean_percentage, efficiency

    # simulates robot vision by checking if there are objects within range
    def is_obstacle(self, point):
        # check screen borders
        if point[0] >= self.display_width or point[0] <= 0 or point[1] >= self.display_height or point[1] <= 0:
            # print("wall nearby")
            return True
        # check obstacles
        for obstacle in self.obstacles:
            if obstacle.rect.collidepoint(point[0], point[1]):
                # print("obstacle nearby")
                return True

        return False

    def calculate_efficiency(self):
        efficiency = (self.step_count / (self.step_count+self.repeated_step_count))*100
        return  efficiency

# general setup
# pygame.init()
# screen = pygame.display.set_mode((1280, 720))
#
# # group setup
# all_sprites = pygame.sprite.Group()
# collision_sprites = pygame.sprite.Group()
#
# # sprite setup
# a = StaticObstacle(pos=(100, 300), size=(100, 50), groups=[all_sprites, collision_sprites])
# b = StaticObstacle((800, 600), (100, 200), [all_sprites, collision_sprites])
# c = StaticObstacle((900, 200), (200, 10), [all_sprites, collision_sprites])
#
# robot = Robot(all_sprites, collision_sprites, screen)
#
# # loop
# last_time = time.time()
#
# env = Environment(robot, [a, b, c])
# print(robot.rect.x, " ", robot.rect.y)
#
#
# # print(env.matrix[720,0])
# # np.savetxt("test.csv",X=env.matrix,fmt="%d",delimiter=",")
# while True:
#     reward, done, clean, battery = env.step()
#
#     if done:
#         print("done")
#         env.reset()
# # delta time
# dt = time.time() - last_time
# last_time = time.time()
#
# # event loop
# for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#         pygame.quit()
#         sys.exit()
#
# # drawing and updating the screen
# screen.fill('lightblue')
# all_sprites.update(dt)
# all_sprites.draw(screen)
#
# # draw the trail line
# if len(trail_lines) <= 0:
#     trail_lines.append(robot.rect.center)
#     trail_lines.append(robot.old_rect.center)
# else:
#     trail_lines.append(robot.rect.center)
#
# pygame.draw.lines(surface=screen, color="red", closed=False, points=trail_lines, width=5)
#
# # display output
# pygame.display.update()
