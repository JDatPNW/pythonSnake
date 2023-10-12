import pygame
import time
import copy

# The `Renderer` class is responsible for rendering the game field either as text or as a visual
# display using the Pygame library.
class Renderer:
  
    def __init__(self, text, visual, gameLogicWidth, gameLogicHeight, numMode, convMode):
        """
        The above function is the initialization function for a Snake game, setting up various variables and
        initializing the Pygame library.
        
        :param text: The `text` parameter is the text renderer object that will be used to display text on
        the screen
        :param visual: The "visual" parameter is a boolean value that determines whether or not to use a
        visual renderer for the game. If it is set to True, the game will use a visual renderer to display
        the game on the screen. If it is set to False, the game will only use a text renderer
        :param gameLogicWidth: The `gameLogicWidth` parameter represents the number of cells in the width of
        the game logic grid
        :param gameLogicHeight: The `gameLogicHeight` parameter represents the number of rows in the game
        logic grid. It determines the height of the game screen
        :param numMode: The `numMode` parameter is used to determine the mode of the game. It is a boolean
        value that indicates whether the game is in number mode or not
        :param convMode: The `convMode` parameter is used to determine the mode of the game. It is likely an
        integer value that represents different game modes. The specific values and their meanings would
        depend on the implementation of the game
        """
        self.textRenderer = text
        self.visualRenderer = visual
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.GREY = (111, 111, 111)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.MARGIN = 2
        self.gameLogicWidth = gameLogicWidth 
        self.gameLogicHeight = gameLogicHeight 
        self.numMode = numMode
        self.convertedMode = convMode
        self.WIDTH = 12
        self.HEIGHT = 12
        # Set the HEIGHT and WIDTH of the screen
        self.WINDOW_SIZE = [(self.gameLogicWidth + 2)*(self.WIDTH + self.MARGIN), (self.gameLogicHeight + 2)*(self.HEIGHT + self.MARGIN)]
        if self.visualRenderer:
            pygame.init()
            self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
            # Set title of screen
            pygame.display.set_caption("Snake")
            # Used to manage how fast the screen updates
            self.clock = pygame.time.Clock()


    def textRender(self, field, sleep):
        """
        The `textRender` function takes a field and a sleep time as input, creates a deep copy of the field,
        converts the values in the copied field to corresponding characters, and then prints the field
        either in number mode or converted mode.
        
        :param field: The `field` parameter is a 2-dimensional list representing a grid. Each element in the
        grid represents a cell and can have one of the following values:
        :param sleep: The `sleep` parameter is the amount of time (in seconds) that the program will pause
        before printing the field. This can be useful if you want to create an animation effect or slow down
        the rendering process
        """
        printField = copy.deepcopy(field)
        time.sleep(sleep)
        for i, row in enumerate(printField):
            for j, cell in enumerate(row):
                if cell == 0:
                    printField[i][j] = " "
                elif cell == 1:
                    printField[i][j] = "o"
                elif cell == 4:
                    printField[i][j] = "0"
                elif cell == 7:
                    printField[i][j] = "X"

                    
        # for i in range(31):
        #     print("\n")
        if self.numMode:
            print("\n")
            for i in range(len(field)):
                print(field[i])
                
        if self.convertedMode:
            print("\n")
            for i in range(len(printField)):
                print(printField[i])

        del printField

    def visualRender(self, field, sleep):
        """
        The `visualRender` function in Python uses the Pygame library to render a visual representation of a
        game field on the screen.
        
        :param field: The `field` parameter is a 2D list that represents the game field. Each element in the
        list represents a cell in the field and contains a value that determines the color of the cell. The
        values 7, 1, and 4 correspond to different colors (green, red,
        :param sleep: The sleep parameter is the amount of time (in seconds) that the program will pause
        before updating the screen. This can be used to control the speed at which the visual rendering is
        displayed
        """
        time.sleep(sleep)
        self.screen.fill(self.BLACK)

        # Draw the grid
        for row in range(self.gameLogicHeight + 2):
            for column in range(self.gameLogicWidth + 2):
                color = self.WHITE
                if field[row][column] == 7:
                    color = self.GREEN
                elif field[row][column] == 1:
                    color = self.RED
                elif field[row][column] == 4:
                    color = self.YELLOW
                elif row == 0 or row == (self.gameLogicHeight + 1) or column == 0 or column == (self.gameLogicWidth + 1):
                    color = self.GREY

                pygame.draw.rect(self.screen,
                                color,
                                [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                self.WIDTH,
                                self.HEIGHT])
        # Limit to 60 frames per second
        self.clock.tick(60)
        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()