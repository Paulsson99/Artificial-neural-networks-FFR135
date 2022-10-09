import numpy as np


class InvalidMove(Exception):
    pass


class TicTacToe:
    '''
    A game of tic-tac-toe.
    The game board is represented by a 2D numpy array.
    0 is an empty square.
    1 is player one
    -1 is player two
    '''
    
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = np.zeros((3, 3))

    def play(self, move: tuple[int, int], player: int) -> int:
        """
        Play a move on the board
        If the square is not empty rasie an exception
        otherwise set the square equal to the player
        """
        if self.board[move]:
            raise InvalidMove(f"The move {move} is invalid. Square is not empty")
        
        self.board[move] = player

    def winner(self) -> int:
        """
        See if someone has won the game yet
        If player1 has won, return 1.
        If player2 has won, return -1
        Otherwise return 0
        """
        # Check win condition in columns, rows and both diagonals (sum to +-3)
        sum_cols = np.sum(self.board, axis=0)
        sum_rows = np.sum(self.board, axis=1)
        sum_diag = np.sum(self.board.diagonal())
        sum_reverse_diag = np.sum(self.board[:, ::-1].diagonal())

        sums = np.concatenate((sum_cols, sum_rows, np.array([sum_diag, sum_reverse_diag])))

        if 3 in sums:
            return 1
        elif -3 in sums:
            return -1
        else:
            return 0

    
    def __str__(self) -> str:
        """
        Make a nice string representation of the game
        """
        game_char = { 0: ' ', 1: 'X', -1: 'O' }
        rows = [' | '.join(game_char[x] for x in row) for row in self.board]
        return ' ' + f' \n{"-" * 11}\n '.join(rows) + ' '


if __name__ == '__main__':
    ttt = TicTacToe()

    ttt.play((0, 0), -1)
    ttt.play((0, 1), -1)
    ttt.play((0, 2), -1)
    ttt.play((1, 1), 1)
    ttt.play((2, 2), 1)

    print(ttt)