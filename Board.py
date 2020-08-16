import chess.variant
import numpy as np
from utils import *

class Board:

    def __init__(self):
        self.board = chess.variant.HordeBoard()

        # array board of pawns on white side.
        self.wBoard = np.zeros((1, 8, 8))

        # array board of pieces on black side. ORDER IS:
        # PAWNS, KNIGHTS, BISHOPS, ROOKS, QUEEN(S), KING
        self.bBoard = np.zeros((6, 8, 8))

        self.half_moves = 0

        self.update_board()

    def board_to_FEN(self):
        self.stateFEN = self.board.fen()
        return self.stateFEN

    def white_has_promoted(self):  # determines if white has a promoted piece.
        for i in range(64):
            piece = str(self.board.piece_at(i))
            pieces = ['N', 'B', 'R', 'Q']
            if piece in pieces:
                return True
        return False

    def update_board(self):
        for i in range(8):
            for j in range(8):
                if self.board.piece_at(8 * i + j) is not None:  # if the square has a piece, encode it as 1.
                    piece = str(self.board.piece_at(8*i+j))
                    if piece == 'P':
                        self.wBoard[0][7-i][j] = 1
                    else:
                        pieces = ['p', 'n', 'b', 'r', 'q', 'k']
                        for k in range(len(pieces)):
                            if piece == pieces[k]:
                                self.bBoard[k][7-i][j] = 1

                else:  # if the square does not have a piece, encode as 0.
                    self.wBoard[0][7-i][j] = 0

                    for k in range(6):
                        self.bBoard[k][7-i][j] = 0

    def move(self, move):
        if chess.Move.from_uci(move) in self.board.legal_moves:
            self.board.push(chess.Move.from_uci(move))

        self.update_board()
        self.half_moves += 1

    def legal_moves(self):
        return [str(x) for x in self.board.legal_moves]

    def print_np_boards(self):
        print("White:")
        print(self.wBoard)
        print("\nBlack:")
        print(self.bBoard)

    def game_result(self):  # return None if unfinished, return 1 if win for pawns, -1 for pieces, 0 for draw
        if self.board.is_game_over():
            if self.board.is_stalemate():
                return 0
            elif self.board.is_variant_end():
                if self.half_moves % 2 == 1:  # white moved last, ergo win for white.
                    return 1
                else:
                    return -1
        return None

    def current_state(self):  # create a state that can be run onto a neural network.

        # CASTLING + TURN
        k_castle = np.zeros((1, 8, 8))
        q_castle = np.zeros((1, 8, 8))
        if self.board.has_kingside_castling_rights(chess.BLACK):
            k_castle = np.ones((1, 8, 8))
        if self.board.has_queenside_castling_rights(chess.BLACK):
            q_castle = np.ones((1, 8, 8))
        # this is the board.
        return np.concatenate((self.wBoard, self.bBoard, k_castle, q_castle), axis=0)


if __name__ == "__main__":
    board = Board()

    print(board.board)

    for move in board.legal_moves():
        print(move_to_array(move))

    print(state_to_database(board.current_state()))


