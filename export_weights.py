"""
When we train networks via train_networkk.py, the saved .pt file contains
information on the state dict, the training loss, and the optimizer gradients.
This, as expected, takes a lot more memory. So here, we're taking a training checkpoint
and converting it to a pytorch file with weights
"""
import torch
from ResNet import *

from torch import nn


if __name__ == "__main__":
    # Specify the model
    model = ResNetDoubleHeadSmall().double()

    # Load training checkpoint
    savedFile = torch.load("models/nn.pt", map_location='cpu')
    model.load_state_dict(savedFile['model_state_dict'])

    # Save Weights
    torch.save(model.state_dict(), "models/trial_nn.pt")