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

class MFNN(nn.Module):
    ''' Multi-fidelity neural network (Meng and Karniadakis 2019):
        - nn_l: a neural network which generates predictions of the
        low-fidelity data
        - nn_h1: a neural network which computes linear correlation between
        the low- and high-fidelity data
        - nn_h2: a neural network which computes nonlinear correlation
        between the low- and high-fidelity data '''

    def __init__(self, nn_l, nn_h1, nn_h2, dim_out=1, use_yl2=False, tau=None):
        super(MFNN, self).__init__()
        self.nn_l = nn_l
        self.nn_h1 = nn_h1
        self.nn_h2 = nn_h2
        self.use_yl2 = use_yl2
        self.tau=tau
        self.dim_out=dim_out

    def forward(self, x):
        # Compute low-fidelity prediction (y_l) via nn_l.
        y_l = self.nn_l(x)
        if self.use_yl2 and self.tau:
            y_l2 = self.nn_l(x-self.tau)
        # Compute linear and nonlinear correlations (F_l, F_nl) via
        # nn_h1 and nn_h2.
        if self.use_yl2 and self.tau:
            F_l = self.nn_h1(torch.cat((x, y_l), 1))
            F_nl = self.nn_h2(torch.cat((x, y_l, y_l2), 1))
        else:
            F_l = self.nn_h1(torch.cat((x, y_l), 1))
            F_nl = self.nn_h2(torch.cat((x, y_l), 1))
        # Compute multi-fidelity prediction (h_1) from F_l and F_nl.
        y_h = F_l + F_nl
        return torch.cat((y_l, y_h), 1)

    def draw(self,
             number_of_neurons_in_widest_layer,
             vertical_distance_between_layers = 6,
             horizontal_distance_between_neurons = 2,
             neuron_radius = 0.5,
            ):
        network = NeuralNetwork()
        for (name, param) in self.named_parameters():
            network.add_layer(4, weights1)
