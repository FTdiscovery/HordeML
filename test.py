import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ResNet import *
from dataset import *
from utils import *
from Board import *

if __name__ == "__main__":

    board = Board()
    board.move("a4a5")
    board.move("a7a6")

    net = ResNetDoubleHeadSmall().double()
    net.load_state_dict(torch.load("models/trial_nn.pt"))
    net.eval()  # set it to evaluation mode

    current_state = torch.from_numpy(board.current_state().reshape(1, 9, 8, 8))
    policy, value = net.forward(current_state)

    # convert policy and value predictions to numpy
    policy = np.exp(policy.detach().numpy().flatten())
    value = value.detach().numpy().flatten()

    win_prob = win_rate(value)
    move_evals = move_evaluations(board.legal_moves(), policy)

    print("Win Rate:", win_prob[0], "%")
    print("Best Move:", best_move(board.legal_moves(), policy))

    print(board.legal_moves())
    print(move_evals)