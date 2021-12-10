class Board(object):
    """docstring for Board."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board  = buildBoard(width, height)

    def buildBoard(width, height):
        row = list([0] * (width+2))
        board = []
        for i in range(height+2):
            board.append(list(row))

        counter = 0
        while(counter < (int(width+height/2))):
            x = random.randint(1,width-2)
            y = random.randint(1,height-2)
            if(board[x][y]== 0):
                board[x][y] = "X"
                counter += 1

        return board

    def getBoard():
        return self.board

    def setField(x, y, value):
        self.board[x][y] = value
