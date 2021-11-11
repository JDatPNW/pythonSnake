import time
import random
# import pygame
import keyboard

width = 25
height = 25

length = 5
snake = [[int(height/2), int(width/2)]]
direction = [1,0]  # [-N/+S , E+/-W]

def init(width, height):
    row = list([0] * (width+2))
    field = []
    for i in range(height+2):
        field.append(list(row))

    counter = 0
    while(counter < (int(width+height/2))):
        x = random.randint(1,width-2)
        y = random.randint(1,height-2)
        if(field[x][y]== 0):
            field[x][y] = "X"
            counter += 1
    return field

def render(field, snake):
    field = list(field)
    snake = list(snake)
    for i in range(31):
        print("\n")

    for row in range(len(field)):
        for column in range(len(field[0])):
            if(field[row][column]!="X"):
                field[row][column] = " "

    for body in snake:
        field[body[0]][body[1]] = "O"

    for i in range(len(field)):
        print(field[i])

def step(direction, length, snake, width, height):
    snake = list(snake)
    dead = False
    direction = list(direction)
    head = list(snake[0])
    head[0] += direction[0]
    head[1] += direction[1]
    if(head[0] == 0):
        head[0] = height
    if(head[0] == height+1):
        head[0] = 1
    if(head[1] == 0):
        head[1] = width
    if(head[1] == width+1):
        head[1] = 1

    if head in snake:
        dead = True

    snake.insert(0, list(head))

    if(len(snake) > length):
        snake.pop()

    return snake, dead

def eat(field, snake, length):
    if(field[snake[0][0]][snake[0][1]] == "X"):
        length +=1
        counter = 0
        while(counter < 1):
            x = random.randint(1,width-2)
            y = random.randint(1,height-2)
            if(field[x][y] == " "):
                field[x][y] = "X"
                counter += 1
    else:
        length = length



    return length, field

def control(direction):
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

field = list(init(width, height))
dead = False
while not dead:
    direction = list(control(direction))
    snake, dead = list(step(direction, length, snake, width, height))
    length, field = eat(field, snake, length)
    render(field, snake)
    time.sleep(0.25)


import pygame

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 20
HEIGHT = 20

# This sets the margin between each cell
MARGIN = 5

# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
for row in range(10):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(10):
        grid[row].append(0)  # Append a cell

# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
grid[1][5] = 1

# Initialize pygame
pygame.init()

# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [255, 255]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set title of screen
pygame.display.set_caption("Snake")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # User clicks the mouse. Get the position
            pos = pygame.mouse.get_pos()
            # Change the x/y screen coordinates to grid coordinates
            column = pos[0] // (WIDTH + MARGIN)
            row = pos[1] // (HEIGHT + MARGIN)
            # Set that location to one
            grid[row][column] = 1
            print("Click ", pos, "Grid coordinates: ", row, column)

    # Set the screen background
    screen.fill(BLACK)

    # Draw the grid
    for row in range(10):
        for column in range(10):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])

    # Limit to 60 frames per second
    clock.tick(60)

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()
