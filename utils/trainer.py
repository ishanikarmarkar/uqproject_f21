import torch
import math
from math import cos, sin, atan
from torch import nn
import torch.nn.functional as F
import sys
import torch.optim as optim
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import sys
sys.path.append('../')
from modules.fnn import FNN
from modules.mfnn import MFNN

def train_MFNN(
    module,
    num_epochs,
    lr,
    l2_lambda_h2,
    train_data,
    num_low,
    num_high,
    grad_low_fi=None,
    l2_lambda_l=0,
    verbose=False,
    p=2
):
    ''' Runs end-to-end training for multi-fidelity neural network.
        - module: the torch module (i.e., neural network) to train
        - num_epochs: number of epochs to train
        - lr: learning rate
        - l2_lambda_h2: regularization rate for NNH2
        - num_low: number of low-fidelity training data
        - num_high: number of high-fidelity training data
        - grad_low_fi (optional; default None): exact gradient of low-fidelity
        function, if available
        - l2_lambda_l (optional; default false): regularization rate for NNL
        - verbose (optiona; default False): whether to print information such as
        the current loss periodically during training. '''

    # Set up optimizer.
    optimizer = optim.LBFGS(module.parameters(), lr=lr)

    # Store inputs and targets, and gradients of low-fidelity function if
    # available.
    (inputs, targets) = (
        torch.narrow(train_data, 1, 0, 1),
        torch.narrow(train_data, 1, 1, 1)
    )
    inputs.requires_grad = True
    if grad_low_fi:
        GT_l = grad_low_fi(inputs[:][:num_low])
    losses = []

    # Define the loss criterion.
    def loss_MFNN(preds, targets, inputs, print_grad=False):
        '''MSE loss calculation as specified in Meng and Karniadakis 2019'''

        # For low-fidelity train data, use low-fidelity prediction from nn_l.
        # For high-fidelity train data, use high-fidelity prediction from nn_h1
        # and nn_h2.
        Y_l = torch.narrow(torch.narrow(preds, 0, 0, num_low), 1, 0, 1)
        Y_h = torch.narrow(torch.narrow(preds, 0, num_low, num_high), 1, 1, 1)
        T_l, T_h = targets[:num_low], targets[num_low:]

        if grad_low_fi:
            # Compute gradient penalty
            GY_l = torch.empty(size=(num_low, 1))
            for i in range(num_low):
                x = inputs[:][i]
                y = module.nn_l(x)
                gyl = torch.autograd.grad(
                    y,
                    x,
                    grad_outputs=torch.ones_like(y),
                    retain_graph=True
                )[0]
                GY_l[i] = gyl
            if print_grad:
                print(GY_l, flush=True)

        # Compute MSE terms from low_fidelity and high-fidelity contribution.
        if grad_low_fi:
            MSE_l = torch.mean((Y_l - T_l)**2 + (GY_l[:][:num_low] - GT_l)**2)
        else:
            MSE_l = torch.mean((Y_l - T_l)**2)
        MSE_h = torch.mean((Y_h - T_h)**2)

        # Compute L2 regularization term from nn_l and nn_h2 parameters.
        # Only penalize for size of weights
        # (not biases).
        l2_reg_l = torch.tensor(0., requires_grad=True)
        l2_reg_h2 = torch.tensor(0., requires_grad=True)
        for (name, param) in module.nn_l.named_parameters():
            if 'weight' in name:
                l2_reg_l = l2_reg_l + torch.norm(param, p=p)**p
        for (name, param) in module.nn_h2.named_parameters():
            if 'weight' in name:
                l2_reg_h2 = l2_reg_h2 + torch.norm(param, p=p)**p

        # Compute and return loss.
        loss = MSE_l + MSE_h + l2_lambda_l*l2_reg_l + l2_lambda_h2*l2_reg_h2
        return loss

    # Define the closure to update weights in each epoch.
    def closure():
        optimizer.zero_grad()
        preds = module(inputs)
        loss = loss_MFNN(preds, targets, inputs)
        loss.backward(retain_graph = True)
        return loss

    # Loop for specified number of epochs.
    for epoch in tqdm(range(num_epochs)):
        optimizer.step(closure)          # step optimizer
        preds = module(inputs)           # compute new predictions
        losses.append(loss_MFNN(preds, targets, inputs).item()) # record loss
        if verbose and epoch % 100 == 0:
            loss_MFNN(preds, targets, inputs, print_grad=True).item()

    return losses

def setup_training(
    low_fi,
    high_fi,
    nn_l_layers,
    nn_h1_layers,
    nn_h2_layers,
    low_pts,
    high_pts,
    nn_h_layers = None,
    use_yl2 = False,
    tau = None,
):
    '''Sets up model for training by formatting training data appropriately.
       - low_fi: low-fidelity function handle
       - high_fi: high-fidelity function handle
       - nn_l_layers: list containing number of neurons per layer for NNL
       - nn_h2_layers: list containing number of neurons per layer for NNH2
       - low_pts: low-fidelity training data input points
       - high_pts: high-fidelity training data input points
       - nn_h_layers (default None): list containing number of neurons per
       layer for a FNN trained on high-fidelity
         data only (with no low-fidelity input for comparison. '''

    # Format train data.
    inputs = torch.transpose(torch.tensor([low_pts+high_pts]), 0, 1)
    num_low = len(low_pts)
    num_high = len(high_pts)
    low = low_fi(inputs[:][:num_low])
    high = high_fi(inputs[:][num_low:])
    train_data = torch.cat((inputs, torch.cat((low, high))), 1)

    # Initialize model.
    if nn_l_layers:
        nn_l = FNN(nn_l_layers)
    if nn_h1_layers:
        nn_h1 = FNN(nn_h1_layers, activation=False)  # no activations for NN_H1
    if nn_h2_layers:
        nn_h2 = FNN(nn_h2_layers)
    nn_mfl = MFNN(nn_l, nn_h1, nn_h2, use_yl2=use_yl2, tau=tau)

    # Initialize and return all information for training including the
    # initialized model. Also include a newly initialized high-fidelity model,
    # if specified in the inputs.
    if nn_h_layers:
        nn_hfl = FNN(nn_h_layers)
        return (nn_mfl, nn_hfl, hfl, train_data, num_low, num_high)
    return (nn_mfl, train_data, num_low, num_high)
