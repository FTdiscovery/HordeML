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
from Board import Board

if __name__ == "__main__":

    moves_chosen = []
    results = []

    files = list(pathlib.Path('hordedatabase').glob('*.pgn'))
    for file in files:
        print(file)
        pgn = open(file)

        for i in range(5000):
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
                        (whiteElo > 1900 and (int(time_control[0]) / 60 + int(time_control[1])) >= 3) or \
                        (whiteElo > 2300):

                    single_game = []

                    board = game.board()
                    for move in game.mainline_moves():
                        board.push_uci(move.uci())
                        single_game.append(move.uci())
                    moves_chosen.append(single_game)
                    results.append(result)

            except:
                continue

    print("Reading through", len(results), "games...")

    states = []
    policy = []
    value = []

    # now that all the moves and results are parsed out, start analyzing.
    for j in range(len(moves_chosen)):

        if j % 100 == 99:
            print("Parsing", int(j+1), "th game...")

        temp = Board()
        outcome = results[j]

        for k in range(len(moves_chosen[j])):
            if temp.half_moves % 2 == 0 and not temp.white_has_promoted():  # we only consider pawn moves by white ...
                state = state_to_database(temp.current_state())
                action = move_representation_dict()[moves_chosen[j][k]]

                if np.sum(temp.current_state().flatten() - database_to_state(state).flatten()) != 0:
                    print("bitboard corrupted...")

                # then save information onto database
                states.append(state)
                policy.append(action)
                value.append(outcome)

            else:  # we do not save information regarding the board.
                print("", end="")

            temp.move(moves_chosen[j][k])

    states = np.asarray(states, dtype=np.uint64)
    policy = np.asarray(policy)
    value = np.asarray(value)

    print(database_to_state(states[0]))
    print("Total number of positions:", len(states))

    # save outputs!
    saveName = 'training_data/lichess_database.h5'

    with h5py.File(saveName, 'w') as hf:
        hf.create_dataset("States", data=states, compression='gzip', compression_opts=5)
        hf.create_dataset("Policy", data=policy, compression='gzip', compression_opts=5)
        hf.create_dataset("Value", data=value, compression='gzip', compression_opts=5)
