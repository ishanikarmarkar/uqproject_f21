import torch
import math
from math import cos, sin, atan
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

class FNN(nn.Module):
    '''Fully-connected neural network implementation (FNN):
       - layer_sizes: the sizes of the layers from input to output layer
       - activation (optional; default False): whether or not to use activation
       functions in this FNN. '''

    def __init__(self, layer_sizes, activation=True):
        super(FNN, self).__init__()
        # Create a list of linear layer modules of the specified sizes.
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        for i in range(1, len(layer_sizes)):
            self.linear_layers.append(
                nn.Linear(layer_sizes[i - 1],
                layer_sizes[i])
            )
            # Override default Lecun initialization to use Xavier's
            # initialization method instead
            nn.init.xavier_normal_(self.linear_layers[-1].weight)

    def forward(self, x):
        # Iterate through linear layers.
        for linear_layer in self.linear_layers[:-1]:
            # Apply activation to hidden layers if needed.
            if self.activation:
                x = torch.tanh(linear_layer(x))
            else:
                x = linear_layer(x)
        # Don't use activation for last layer.
        x = self.linear_layers[-1](x)
        return x
