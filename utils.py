import numpy as np

def numpy_to_bitboard(arr):  # converts a numpy 8 x 8 matrix into an integer
    bitboard = '0' * 64  # start off as a string
    for i in range(8):
        for j in range(8):
            if arr[i][j] == 1:
                bitboard = bitboard[0:8*i+j]+'1'+bitboard[8*i+j+1:]
    return int(bitboard, 2)

def bitboard_to_numpy(num):  # converts integer into a numpy 8 x 8 matrix.
    arr = np.zeros((8, 8))
    bitboard = str(format(num, 'b').zfill(64))
    print(bitboard)
    for i in range(8):
        for j in range(8):
            if bitboard[8*i+j] == '1':
                arr[i][j] = 1

    return arr

def state_to_database(state):  # converts a 9 x 8 x 8 state representation into an array of 9 integers.
    data = []
    for board in state:
        data.append(numpy_to_bitboard(board))
    return np.asarray(data).reshape(1, -1)

def square_name(num):  # converts square number into name
    files = 'abcdefgh'
    return files[num % 8]+str(8 - (num // 8))

def move_representation_dict():  # constructs a dictionary that converts all possible moves from uci to an index.

    moves = {}

    # first, cover all promotions. there should be 8x4+7x4+7x4 = 32+56=88 in total

    promotions = 'nbrq'
    for i in range(8, 16):
        for piece in promotions:

            if i % 8 != 0: # left captures
                moves[square_name(i) + square_name(i - 9) + piece] = len(moves)

            moves[square_name(i)+square_name(i-8)+piece] = len(moves)  # straight promotions

            if i % 8 != 7:  # right captures
                moves[square_name(i) + square_name(i - 7) + piece] = len(moves)

    # next, we cover all other possible moves. There should be 8x2 + 8x6 + 7x6 + 7x6 = 16+48+84 = 64+84 = 148 in total

    for i in range(16, 64):

        if i % 8 != 0:  # left captures
            moves[square_name(i) + square_name(i - 9)] = len(moves)

        moves[square_name(i) + square_name(i - 8)] = len(moves)  # one move forwards

        if i >= 48:  # move two squares forward
            moves[square_name(i) + square_name(i - 16)] = len(moves)

        if i % 8 != 7:  # right captures
            moves[square_name(i) + square_name(i - 7)] = len(moves)

    return moves

def move_to_array(move):

    dict = move_representation_dict()
    arr = np.zeros(236)

    arr[dict[move]] = 1

    return arr

def array_to_move(arr):

    dict = move_representation_dict()
    return list(dict.keys())[list(dict.values()).index(np.argmax(arr))]


if __name__ == "__main__":
    print(array_to_move(move_to_array('a7a8b')))




