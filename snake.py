import time
import keyboard
import random

width = 25
height = 25

length = 5
snake = [[int(height/2), int(width/2)]]
direction = [1,0]  # [-N/+S , E+/-W]

def init(width, height):
    row = list([0] * width)
    field = []
    for i in range(height):
        field.append(list(row))

    counter = 0
    while(counter < (int(width+height/2))):
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)
        if(field[x][y]==0):
            field[x][y] = "X"
            counter += 1
            print(counter)
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

def step(direction, length, snake):
    snake = list(snake)
    direction = list(direction)
    head = snake[0]
    head[0] += direction[0]
    head[1] += direction[1]
    snake.insert(0, list(head))
    if(len(snake) > length):
        snake.pop()
    return snake

def eat(field, snake, length):
    if(field[snake[0][0]][snake[0][1]] == "X"):
        length +=1
    else:
        length = length
    return length

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
for i in range(1000):
    direction = list(control(direction))
    snake = list(step(direction, length, snake))
    length = eat(field, snake, length)
    render(field, snake)
    time.sleep(0.25)
