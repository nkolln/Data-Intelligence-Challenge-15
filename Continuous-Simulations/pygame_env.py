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

        self.rect = self.image.get_rect(topleft=pos)
        self.old_rect = self.rect.copy()


class MovingVerticalObstacle(StaticObstacle):
    def __init__(self, pos, size, groups, max_up, max_down, speed):
        super().__init__(pos, size, groups)
        self.image.fill('green')
        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2((0, 1))
        self.speed = speed
        self.max_up = max_up
        self.max_down = max_down
        self.old_rect = self.rect.copy()

    def update(self, dt):
        self.old_rect = self.rect.copy()  # previous frame
        if self.rect.bottom > self.max_down:
            self.rect.bottom = self.max_down
            self.pos.y = self.rect.y
            self.direction.y *= -1
        if self.rect.top < self.max_up:
            self.rect.top = self.max_up
            self.pos.y = self.rect.y
            self.direction.y *= -1

        self.pos.y += self.direction.y * self.speed * dt
        self.rect.y = round(self.pos.y)  # current frame


class MovingHorizontalObstacle(StaticObstacle):
    def __init__(self, pos, size, groups, max_left, max_right, speed):
        super().__init__(pos, size, groups)
        self.image.fill('purple')
        self.pos = pygame.math.Vector2(self.rect.topleft)
        self.direction = pygame.math.Vector2((1, 0))
        self.max_left = max_left
        self.max_right = max_right
        self.speed = speed
        self.old_rect = self.rect.copy()

    def update(self, dt):
        self.old_rect = self.rect.copy()
        if self.rect.right > self.max_right:
            self.rect.right = self.max_right
            self.pos.x = self.rect.x
            self.direction.x *= -1
        if self.rect.left < self.max_left:
            self.rect.left = self.max_left
            self.pos.x = self.rect.x
            self.direction.x *= -1

        self.pos.x += self.direction.x * self.speed * dt
        self.rect.x = round(self.pos.x)


class RobotCopy(pygame.sprite.Sprite):
    def __init__(self, groups, og_robot):
        super().__init__(groups)
        self.screen = og_robot.screen
        self.og_image = og_robot.og_image
        self.image = og_robot.image.copy()
        self.rect = deepcopy(og_robot.rect)
        self.old_rect = deepcopy(og_robot.old_rect)
        self.pos = deepcopy(og_robot.pos)
        self.direction = deepcopy(og_robot.direction)
        self.speed = deepcopy(og_robot.speed)
        self.obstacles = og_robot.obstacles

        self.battery_percentage = og_robot.battery_percentage
        self.battery_drain_p = og_robot.battery_drain_p
        self.battery_drain_l = og_robot.battery_drain_l
        self.robot_collided = og_robot.robot_collided

    # drains the battery by lambda given probability
    def drain_battery(self, is_cont):
        movement_vector = pygame.Vector2(self.direction.x * self.speed, self.direction.y * self.speed)

        do_battery_drain = np.random.binomial(1, self.battery_drain_p)
        if do_battery_drain == 1 and self.battery_percentage > 0:

            if movement_vector.length() != 0 and is_cont:
                movement_vector = movement_vector.normalize()
                self.battery_percentage -= np.random.exponential(
                    self.battery_drain_l) * movement_vector.magnitude_squared()
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

                    # collision on the left
                    if self.rect.left <= sprite.rect.right and self.old_rect.left >= sprite.old_rect.right:
                        self.rect.left = sprite.rect.right
                        self.pos.x = self.rect.x

            if direction == 'vertical':
                for sprite in collision_sprites:
                    # collision on the bottom
                    if self.rect.bottom >= sprite.rect.top and self.old_rect.bottom <= sprite.old_rect.top:
                        self.rect.bottom = sprite.rect.top
                        self.pos.y = self.rect.y

                    # collision on the top
                    if self.rect.top <= sprite.rect.bottom and self.old_rect.top >= sprite.old_rect.bottom:
                        self.rect.top = sprite.rect.bottom
                        self.pos.y = self.rect.y

        else:
            self.robot_collided = False

    def window_collision(self, direction):

        if direction == 'horizontal':
            if self.rect.left < 0:
                self.robot_collided = True
                # print("robot hit something")
                self.rect.left = 0
                self.pos.x = self.rect.x

                return

            if self.rect.right > self.screen.get_width():
                self.robot_collided = True
                # print("robot hit something")
                self.rect.right = self.screen.get_width()
                self.pos.x = self.rect.x

                return
            self.robot_collided = False

        if direction == 'vertical':
            if self.rect.top < 0:
                self.robot_collided = True
                # print("robot hit something")
                self.rect.top = 0
                self.pos.y = self.rect.y

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
                self.image = pygame.transform.rotate(self.og_image, 180 + 45)

            elif self.direction.x < 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 180 - 45)
        elif self.direction.y < 0:
            if self.direction.x == 0:  # rotate right
                self.image = pygame.transform.rotate(self.og_image, 0)

            elif self.direction.x > 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 0 - 45)

            elif self.direction.x < 0:  # rotate left
                self.image = pygame.transform.rotate(self.og_image, 0 + 45)

    # used for continuous action space
    def update(self, x, y):
        # y goes to x and x goes to y because pygame uses y-x matrix smh.
        self.old_rect = self.rect.copy()
        self.direction.x = y
        self.direction.y = x
        self.rotate_image()

        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()

        self.drain_battery(True)
        # print("battery perc: ", self.battery_percentage)

        # move and check for collisions on x axis
        self.pos.x += self.direction.x * self.speed
        self.rect.x = round(self.pos.x)
        self.collision('horizontal')
        self.window_collision('horizontal')

        # move and check for collisions on y axis
        self.pos.y += self.direction.y * self.speed
        self.rect.y = round(self.pos.y)
        self.collision('vertical')
        self.window_collision('vertical')


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
        self.direction = pygame.math.Vector2((0, 0))  # upwards
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
    def drain_battery(self, is_cont):
        movement_vector = pygame.Vector2(self.direction.x * self.speed, self.direction.y * self.speed)

        do_battery_drain = np.random.binomial(1, self.battery_drain_p)
        if do_battery_drain == 1 and self.battery_percentage > 0:

            if movement_vector.length() != 0 and is_cont:
                movement_vector = movement_vector.normalize()
                self.battery_percentage -= np.random.exponential(
                    self.battery_drain_l) * movement_vector.magnitude_squared()
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

                    # collision on the left
                    if self.rect.left <= sprite.rect.right and self.old_rect.left >= sprite.old_rect.right:
                        self.rect.left = sprite.rect.right
                        self.pos.x = self.rect.x

            if direction == 'vertical':
                for sprite in collision_sprites:
                    # collision on the bottom
                    if self.rect.bottom >= sprite.rect.top and self.old_rect.bottom <= sprite.old_rect.top:
                        self.rect.bottom = sprite.rect.top
                        self.pos.y = self.rect.y

                    # collision on the top
                    if self.rect.top <= sprite.rect.bottom and self.old_rect.top >= sprite.old_rect.bottom:
                        self.rect.top = sprite.rect.bottom
                        self.pos.y = self.rect.y

        else:
            self.robot_collided = False

    def window_collision(self, direction):

        if direction == 'horizontal':
            if self.rect.left < 0:
                self.robot_collided = True
                # print("robot hit something")
                self.rect.left = 0
                self.pos.x = self.rect.x

                return

            if self.rect.right > self.screen.get_width():
                self.robot_collided = True
                # print("robot hit something")
                self.rect.right = self.screen.get_width()
                self.pos.x = self.rect.x

                return
            self.robot_collided = False

        if direction == 'vertical':
            if self.rect.top < 0:
                self.robot_collided = True
                # print("robot hit something")
                self.rect.top = 0
                self.pos.y = self.rect.y

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
        self.drain_battery(False)
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
        self.rotate_image()

        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()

        self.drain_battery(True)
        # print("battery perc: ", self.battery_percentage)

        # move and check for collisions on x axis
        self.pos.x += self.direction.x * self.speed
        self.rect.x = round(self.pos.x)
        self.collision('horizontal')
        self.window_collision('horizontal')

        # move and check for collisions on y axis
        self.pos.y += self.direction.y * self.speed
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
                self.image = pygame.transform.rotate(self.og_image, 180 + 45)

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
    def __init__(self, robot: Robot, obstacles: List[StaticObstacle], all_sprites, collision_sprites, screen,
                 copy_sprites=None):
        self.display_width = screen.get_width()
        self.display_height = screen.get_height()
        pygame.init()
        self.screen = screen

        # group setup
        self.all_sprites = all_sprites
        self.collision_sprites = collision_sprites
        self.copy_sprites = copy_sprites
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

        # used in continuous action space to simulate test movements
        self.temp_matrix = deepcopy(self.matrix)
        self.copy_robot = None
        self.temp_step_count = deepcopy(self.step_count)
        self.temp_rep_step_count = deepcopy(self.repeated_step_count)

        self.last_time = time.time()
        self.dt = time.time() - self.last_time

        self.clock = pygame.time.Clock()
        self.reset()

    # resets environment so that it can be run again
    def reset(self):
        self.init_matrix()
        self.clean_percentage = self.calc_clean_percentage()
        self.last_time = time.time()
        self.robot.reset_robot()

        self.trail_lines = []
        self.dt = time.time() - self.last_time
        self.step_count = 0
        self.repeated_step_count = 0

        # self.cont_step(0, 0, True)

        self.temp_matrix = deepcopy(self.matrix)
        self.copy_robot = None
        self.copy_sprites.empty()
        self.temp_step_count = deepcopy(self.step_count)
        self.temp_rep_step_count = deepcopy(self.repeated_step_count)

        self.all_sprites.draw(self.screen)
        pygame.display.flip()

    # initializes the matrix representation
    def init_matrix(self):

        self.matrix = np.zeros((self.display_height + 1, self.display_width + 1), dtype=int)

        self.init_screen_borders()

        # set the cells corresponding to each obstacle border in the matrix to 4
        for obstacle in self.obstacles:
            self.init_obstacle_borders(obstacle)
        # set the cells corresponding to each obstacle in the matrix to 2
        for obstacle in self.obstacles:
            self.matrix[obstacle.pos[1]:obstacle.pos[1] + obstacle.size[1]+1,
            obstacle.pos[0]:obstacle.pos[0] + obstacle.size[0]+1] = 2
        # set the cells corresponding to the current robot location to 1 (clean)
        self.set_robot_location()

    # set the cells corresponding to the current robot location to 1 (clean). if it is a wall border, it is set to 5
    def set_robot_location(self, is_copy=False):
        if not is_copy:
            robot_location = self.matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1]+1,
            self.robot.rect.topleft[0]:self.robot.rect.topright[0]+1]

            for i, value in enumerate(robot_location):
                if value == 4:
                    robot_location[i] = 5
                else:
                    robot_location[i] = 1

            self.matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1] + 1,
            self.robot.rect.topleft[0]:self.robot.rect.topright[0] + 1] = robot_location
            return
        robot_location = self.temp_matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1] + 1,
                         self.robot.rect.topleft[0]:self.robot.rect.topright[0] + 1]

        for i, value in enumerate(robot_location):
            if value == 4:
                robot_location[i] = 5
            else:
                robot_location[i] = 1

        self.temp_matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1] + 1,
        self.robot.rect.topleft[0]:self.robot.rect.topright[0] + 1] = robot_location
    # sets the borders of the given obstacle in the matrix representation to 4. border size is the size of the robot (33)
    def init_obstacle_borders(self, obstacle:StaticObstacle):
        # top border
        if not obstacle.pos[1]-33 < 0:
            self.matrix[obstacle.pos[1]-33:obstacle.pos[1], obstacle.pos[0]:obstacle.pos[0]+obstacle.size[0]+1] = 4
        # bottom border
        if not obstacle.rect.bottomleft[1]+34 > self.screen.get_height:
            self.matrix[obstacle.rect.bottomleft[1]+1:obstacle.rect.bottomleft[1]+34, obstacle.pos[0]:obstacle.pos[0] + obstacle.size[0]] = 4

        if not obstacle.pos[0]-33 < 0:
            self.matrix[obstacle.pos[1]:obstacle.pos[1]+obstacle.size[1]+1,obstacle.pos[0]-33:obstacle.pos[0]] = 4

        if not obstacle.rect.topright[0]+34 > self.screen.get_width:
            self.matrix[obstacle.pos[1]:obstacle.pos[1] + obstacle.size[1]+1, obstacle.rect.topright[0]:obstacle.rect.topright[0]+34] = 4

    # sets screen borders to 4. border size is size of the robot (33)
    def init_screen_borders(self):
        # left border
        self.matrix[0:-1, 0:33] = 4
        # right border
        self.matrix[0:-1, -33:-1] = 4

        # top border
        self.matrix[0:33, 0:-1] = 4

        # bottom border
        self.matrix[-33:-1, 0:-1] = 4

    # sets current robot location to 1 (clean)
    def update_matrix(self, is_copy=False):

        if not is_copy:
            self.set_robot_location()
            # set obstacle locations to 2 in case robot went through them

            # set the cells corresponding to each obstacle in the matrix to 2
            for obstacle in self.obstacles:
                self.matrix[obstacle.pos[1]:obstacle.pos[1] + obstacle.size[1] + 1,
                obstacle.pos[0]:obstacle.pos[0] + obstacle.size[0] + 1] = 2

            self.clean_percentage = self.calc_clean_percentage()
            return

        self.set_robot_location(is_copy=is_copy)

        # set obstacle locations to 2 in case robot went through them
        for obstacle in self.obstacles:
            self.temp_matrix[obstacle.pos[1]:obstacle.pos[1] + obstacle.size[1],
            obstacle.pos[0]:obstacle.pos[0] + obstacle.size[0]] = 2

        self.clean_percentage = self.calc_clean_percentage(True)

    def calc_clean_percentage(self, is_copy=False):
        if not is_copy:
            clean_count = np.count_nonzero(self.matrix == 1)
            clean_count += np.count_nonzero(self.matrix == 5)
            dirty_count = np.count_nonzero(self.matrix == 0)
            clean_percentage = (clean_count / (clean_count + dirty_count)) * 100
            # print("clean percentage: ", clean_percentage)
            return clean_percentage

        clean_count = np.count_nonzero(self.temp_matrix == 1)
        clean_count += np.count_nonzero(self.temp_matrix == 5)
        dirty_count = np.count_nonzero(self.temp_matrix == 0)
        clean_percentage = (clean_count / (clean_count + dirty_count)) * 100
        # print("clean percentage: ", clean_percentage)
        return clean_percentage

    # checks if robots current location is clean
    def is_robot_location_dirty(self, is_copy=False):
        if not is_copy:
            robot_location = self.matrix[self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                             self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
            dirty = np.count_nonzero(robot_location == 0)
            # print("dirty count: ", dirty)
            return dirty > 0
        robot_location = self.temp_matrix[self.copy_robot.rect.topleft[1]:self.copy_robot.rect.bottomleft[1],
                         self.copy_robot.rect.topleft[0]:self.copy_robot.rect.topright[0]]
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
                            round(self.robot.rect.topleft[1] - next_up_down):round(
                                self.robot.rect.bottomleft[1] - next_up_down),
                            self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        elif location == "down " and round(self.robot.rect.topleft[1] + next_up_down) < self.display_height:
            next_location = self.matrix[
                            round(self.robot.rect.topleft[1] + next_up_down):round(
                                self.robot.rect.bottomleft[1] + next_up_down),
                            self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        elif location == "right" and round(self.robot.rect.topleft[0] - next_right_left) >= 0:
            next_location = self.matrix[
                            self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                            round(self.robot.rect.topleft[0] - next_right_left):round(
                                self.robot.rect.topright[0] - next_right_left)]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        elif location == "left" and round(self.robot.rect.topleft[0] + next_right_left) < self.display_width:
            next_location = self.matrix[
                            self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                            round(self.robot.rect.topleft[0] + next_right_left):round(
                                self.robot.rect.topright[0] + next_right_left)]
            dirty = np.count_nonzero(next_location == 0)
            return dirty > 0

        # if out of bounds
        else:
            return False

    # returns robots center as tuple and whole location as a matrix
    def robot_location(self, is_copy=False):
        if not is_copy:
            robot_location = self.matrix[
                             self.robot.rect.topleft[1]:self.robot.rect.bottomleft[1],
                             self.robot.rect.topleft[0]:self.robot.rect.topright[0]]
            robot_center = self.robot.rect.center
            robot_center = (robot_center[1], robot_center[0])  # yx to xy
            return robot_center, robot_location
        robot_location = self.temp_matrix[self.copy_robot.rect.topleft[1]:self.copy_robot.rect.bottomleft[1],
                         self.copy_robot.rect.topleft[0]:self.copy_robot.rect.topright[0]]
        robot_center = self.copy_robot.rect.center
        robot_center = (robot_center[1], robot_center[0])  # yx to xy
        return robot_center, robot_location

    def cont_step(self, x, y, update=True):

        # set time passed since last step. Used for smooth movement
        self.dt = time.time() - self.last_time
        self.last_time = time.time()
        step_reward = 0  # reward for the current step
        done = False  # flag for simulation end

        # event loop. Basically closes the program when the window is closed. Can also be used to print final results, write data etc.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # data export can be written here
                pygame.quit()
                sys.exit()

        # create a copy of the robot and the matrx and start updating those while update is set to False
        if not update:
            # if it is the fisrt time switching to a copy, create a new instance of robotCopy
            if self.copy_robot is None:
                self.copy_robot = RobotCopy(self.copy_sprites, self.robot)
                print("Switched to copy robot")
            # increment step count for test moves, will return to original value when switched backed to original robot
            self.temp_step_count += 1

            self.screen.fill('lightblue')
            # move the copy robot
            self.copy_sprites.update(x, y)

            # reward system
            # if current location is more clean than dirty give low reward, else give high reward
            if self.is_robot_location_dirty(True):
                step_reward = 20
                # print("location dirty")
            else:
                step_reward = -15
                self.temp_rep_step_count += 1
                # print("location clean")
            # low reward if robot hit something
            if self.copy_robot.robot_collided:
                step_reward += -5
            # update the temp matrix
            self.update_matrix(True)
            # drawing robot and the obstacles to the screen
            self.copy_sprites.draw(self.screen)
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
            self.clock.tick(60)

            efficiency = self.calculate_efficiency(True)

            if self.clean_percentage >= 100 or self.copy_robot.battery_percentage <= 1:
                done = True
                return step_reward, done, self.clean_percentage, efficiency
            # return done = False id the simulation is not done
            return step_reward, done, self.clean_percentage, efficiency

        print("switched to original robot")
        self.step_count += 1
        self.copy_robot = None
        self.copy_sprites.empty()

        # drawing and updating the screen
        self.screen.fill(
            'lightblue')  # fills background with specified color. Everytihng including this has to be re-drawn at each step smh.
        self.all_sprites.update(self.dt, None, x, y,
                                True)  # calls the update function of the robot and the obstacles with the given parameters. last input says to call const_update

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
            step_reward += -5

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
        self.clock.tick(60)

        self.temp_matrix = deepcopy(self.matrix)
        self.temp_step_count = deepcopy(self.step_count)
        self.temp_rep_step_count = deepcopy(self.repeated_step_count)

        efficiency = self.calculate_efficiency()
        # print("eff: ",efficiency)
        # return done = True if battery is dead or run completed
        if self.clean_percentage >= 100 or self.robot.battery_percentage <= 1:
            # if self.clean_percentage >= 100:
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

                return True

        return False

    def calculate_efficiency(self, is_copy=False):
        if not is_copy:
            efficiency = (self.step_count / (self.step_count + self.repeated_step_count)) * 100
            return efficiency

        efficiency = (self.temp_step_count / (self.temp_step_count + self.temp_rep_step_count)) * 100
        return efficiency
