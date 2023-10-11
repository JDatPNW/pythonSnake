'''             
███████╗███╗   ██╗ █████╗ ██╗  ██╗███████╗
██╔════╝████╗  ██║██╔══██╗██║ ██╔╝██╔════╝
███████╗██╔██╗ ██║███████║█████╔╝ █████╗  
╚════██║██║╚██╗██║██╔══██║██╔═██╗ ██╔══╝  
███████║██║ ╚████║██║  ██║██║  ██╗███████╗
╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
                                          
'''

import random
import numpy as np
import random
# import keyboard # if MANUAL

# The snakeGame class represents a game of snake, with methods for initializing the game, updating the
# field and snake, handling eating, and controlling the snake's direction.
class snakeGame():
        
    def __init__(self, width, height, length, fruit_num, can_teleport, reward_step, reward_fruit, reward_into_self, reward_wall):
        """
        The above function is a Python class constructor that initializes the attributes of a snake game.
        
        :param width: The width parameter represents the width of the game board or grid. It determines the
        number of columns in the grid
        :param height: The `height` parameter represents the number of rows in the game grid
        :param length: The "length" parameter represents the initial length of the snake. It determines how
        many segments the snake starts with
        :param fruit_num: The parameter `fruit_num` represents the number of fruits in the game
        :param can_teleport: The `can_teleport` parameter determines whether the snake can teleport from one
        side of the game board to the other. If `can_teleport` is set to `True`, the snake will be able to
        move from the right edge of the board to the left edge, and vice versa. If
        """
        self.WIDTH = width
        self.HEIGHT = height

        self.LENGTH = length
        self.startingLENGTH = length
        self.SNAKE = [[int(self.HEIGHT/2), int(self.WIDTH/2)]] # TODO: check if they are in the right order or not
        self.direction = [1,0]  # [-N/+S , E+/-W]

        self.num_fruits = fruit_num
        self.can_teleport = can_teleport

        self.reward_into_self = reward_into_self
        self.reward_wall = reward_wall
        self.reward_fruit = reward_fruit
        self.reward_step = reward_step

    def initGame(self):
        """
        The `initGame` function initializes the game by setting the initial direction, length, and position
        of the snake, creating a field with boundaries, and randomly placing food on the field.
        :return: The function `initGame` returns the `self.field` variable.
        """
        
        # Fixed starting Direction
        # self.direction = [1,0]  # [-N/+S , E+/-W] <- use this one if want no randomness
        
        # Random starting Direction
        self.direction = []
        start_dir = random.randint(0,3)
        if(start_dir==(0)):
            self.direction = [-1,0]
        elif(start_dir==(1)):
            self.direction = [1,0]
        elif(start_dir==(2)):
            self.direction = [0,1]
        elif(start_dir==(3)):
            self.direction = [0,-1]

        self.LENGTH = self.startingLENGTH
        # If fixed starting point
        # self.SNAKE = [[int(self.HEIGHT/2), int(self.WIDTH/2)]] # TODO: check if they are in the right order or not <- use this if no random starting
        self.SNAKE = [[random.randint(0+3, self.HEIGHT-3), random.randint(0+3, self.WIDTH-3)]]

        row = list([0] * (self.WIDTH+2))
        self.field = []
        for _ in range(self.WIDTH+2):
            self.field.append(list(row))

        counter = 0
        while(counter < (int(self.num_fruits))):
            x = random.randint(1, self.WIDTH-2)
            y = random.randint(1, self.HEIGHT-2)
            if(self.field[x][y]== 0):
                self.field[x][y] = 7
                counter += 1
        del row, _, start_dir, counter, x, y
        return self.field

    def update_field(self):
        """
        The function updates the game field by setting all non-7 values to 0, setting the snake body to 1,
        and setting the snake head to 4.
        :return: the updated field.
        """
        for row in range(len(self.field)):
            for column in range(len(self.field[0])):
                if(self.field[row][column]!=7):
                    self.field[row][column] = 0

        for body in self.SNAKE:
            self.field[body[0]][body[1]] = 1
        self.field[self.SNAKE[0][0]][self.SNAKE[0][1]] = 4

        return self.field

    def update_snake(self):
        """
        The `update_snake` function updates the position of the snake and checks for collisions with walls
        or itself, returning a boolean value indicating if the snake is dead.
        :return: a boolean value indicating whether the snake is dead or not.
        """
        dead = False
        cause = "Alive"
        head = list(self.SNAKE[0])
        head[0] += self.direction[0]
        head[1] += self.direction[1]

        # If wrapping teleopring
        # if(head[0] == 0):
        #     head[0] = height
        # if(head[0] == height+1):
        #     head[0] = 1
        # if(head[1] == 0):
        #     head[1] = width
        # if(head[1] == width+1):
        #     head[1] = 1

        # if wall = dead
        if(head[0] == 0):
            dead = True
            cause = "Into Wall"
        if(head[0] == self.HEIGHT+1):
            dead = True
            cause = "Into Wall"
        if(head[1] == 0):
            dead = True
            cause = "Into Wall"
        if(head[1] == self.WIDTH+1):
            dead = True
            cause = "Into Wall"

        if head in self.SNAKE:
            dead = True
            cause = "Eat Self"
        self.SNAKE.insert(0, list(head))

        if(len(self.SNAKE) > self.LENGTH):
            self.SNAKE.pop()
        del head
        return dead, cause

    def eat(self):
        """
        The function "eat" checks if the snake's head is on a food item, and if so, increases the snake's
        length, updates the field, and returns the updated field and a reward of 2
        ; otherwise, it returns
        the original field and a reward of 0.
        :return: the updated field and the reward.
        """
        if(self.field[self.SNAKE[0][0]][self.SNAKE[0][1]] == 7):
            self.LENGTH +=1
            counter = 0
            reward = self.reward_fruit
            while(counter < 1):
                x = random.randint(1,self.WIDTH-2)
                y = random.randint(1,self.HEIGHT-2)
                if(self.field[x][y] == 0):
                    self.field[x][y] = 7
                    counter += 1
        else:
            reward = 0
        return self.field, reward

    def control(self, action):
        """
        The `control` function takes an action as input and updates the direction of the snake based on the
        action, returning a boolean indicating whether the snake will run into itself.
        
        :param action: The `action` parameter is an integer that represents the action to be taken. The
        possible values for `action` are: depending on the action space
        :return: the value of the variable "run_into_self".
        """

        run_into_self = False

        #MANUAL controls using import keyboard
        '''
        direction = list(direction)
        if(keyboard.is_pressed("w")):
            direction = [-1,0]
        elif(keyboard.is_pressed("s")):
            direction = [1,0]
        if(keyboard.is_pressed("d")):
            direction = [0,1]
        if(keyboard.is_pressed("a")):
            direction = [0,-1]
        return direction
        '''

        if(action==(0)):
            if self.direction == [1,0]:
                self.direction = [1,0]
                run_into_self = True
            else:
                self.direction = [-1,0]
        elif(action==(1)):
            if self.direction == [-1,0]:
                self.direction = [-1,0]
                run_into_self = True
            else:
                self.direction = [1,0]
        elif(action==(2)):
            if self.direction == [0,-1]:
                self.direction = [0,-1]
                run_into_self = True
            else:
                self.direction = [0,1]
        elif(action==(3)):
            if self.direction == [0,1]:
                self.direction = [0,1]
                run_into_self = True
            else:
                self.direction = [0,-1]
        elif(action==(4)):
            pass
        return run_into_self
