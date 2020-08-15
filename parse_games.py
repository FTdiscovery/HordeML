import chess.variant
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
import pathlib
import h5py
import json
from utils import *
import chess.pgn

if __name__ == "__main__":

    games_used = 0

    positions = []
    moves_chosen = []
    results = []

    files = list(pathlib.Path('hordedatabase').glob('*.pgn'))
    for file in files:
        print(file)
        pgn = open(file)

        for i in range(75000):
            try:
                game = chess.pgn.read_game(pgn)
                whiteElo = int(game.headers["WhiteElo"])
                blackElo = int(game.headers["BlackElo"])
                time_control = str(game.headers["TimeControl"]).split('+')
                result = str(game.headers["Result"])
                if result == "1-0":
                    result = 1
                elif result == "0-1":
                    result = -1
                else:
                    result = 0

                if (whiteElo > 2100 and int(time_control[0]) > 90) or \
                    (whiteElo > 1900 and (int(time_control[0])/60 + int(time_control[1])) >= 3) or \
                        (whiteElo > 2300):

                    games_used += 1

                    if games_used % 100 == 0:
                        print(games_used)
            except:
                continue

