import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ResNet import *
from dataset import *
from utils import *
from Board import *
import time
import chess.engine

import chess.pgn


# POTENTIAL NPS: 70

if __name__ == "__main__":

    board = Board()
    game = chess.pgn.Game()
    game.headers["Variant"] = "Horde"

    net = ResNetDoubleHeadSmall().double()
    net.load_state_dict(torch.load("models/trial_nn.pt"))
    net.eval()  # set it to evaluation mode

    while board.game_result() == None:

        if board.half_moves % 2 == 0 and not board.white_has_promoted():
            current_state = torch.from_numpy(board.current_state().reshape(1, 9, 8, 8))
            policy, value = net.forward(current_state)

            # convert policy and value predictions to numpy
            policy = np.exp(policy.detach().numpy().flatten())
            value = value.detach().numpy().flatten()

            win_prob = win_rate(value)
            move_evals = move_evaluations(board.legal_moves(), policy)
            top_move = best_move(board.legal_moves(), policy)

            print("Win Rate:", win_prob[0], "%")
            print("Best Move:", top_move)

            print(board.legal_moves())
            print(move_evals)

            board.move(top_move)
        else:
            stockfish = chess.engine.SimpleEngine.popen_uci("models/stockfish")
            info = stockfish.analyse(board.board, chess.engine.Limit(time=0.002))
            top_move = str(info["pv"][0])

            print("stockfish moves:", top_move)
            board.move(top_move)

        if board.half_moves == 1:
            node = game.add_variation(chess.Move.from_uci(top_move))
        else:
            node = node.add_variation(chess.Move.from_uci(top_move))

        print(game)

    if board.game_result() == 1:
        print("White wins.")
    else:
        print("Black wins.")
