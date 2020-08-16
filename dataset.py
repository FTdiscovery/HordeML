import torch.utils.data as data_utils
import torch.nn as nn
import torch
import numpy as np
from utils import *
import h5py

class TrainingSet(torch.utils.data.Dataset):

    def __init__(self, inputs, policy_out, value_out):
        self.features = inputs
        self.targets = policy_out
        self.targets2 = value_out

    def __getitem__(self, index):

        state = database_to_state(self.features[index])

        # policy output vector created
        array = np.zeros(236)
        if self.targets2[index] == -1:  # policy value lower for lost move
            array[self.targets[index]] = 0.2
        elif self.targets2[index] == 0:  # policy value lower for drawn move
            array[self.targets[index]] = 0.5
        else:
            array[self.targets[index]] = 1

        return torch.from_numpy(state), torch.from_numpy(array), torch.from_numpy(np.expand_dims(self.targets2[index], axis=0)) # may have to wrap

    def __len__(self):
        return len(self.features)

if __name__ == "__main__":

    with h5py.File("training_data/lichess_database.h5", 'r') as hf:
        states = hf["States"][:]
        policy = hf["Policy"][:]
        value = hf["Value"][:]


    processor = TrainingSet(states, policy, value)
    print(processor.__len__())
    print(processor.__getitem__(1500))

