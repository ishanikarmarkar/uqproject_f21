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
from .neural_network import (
    Network,
    Layer,
    Neuron,
)

def visualize_network(
    nn_mfl,
    vertical_distance_between_layers = 6,
    horizontal_distance_between_neurons = 2,
    neuron_radius = 1,
    number_of_neurons_in_widest_layer = 4,
    width_scale=.5,
    input_dim=1,
    output_dim=1,
):
    # Initialize network.
    network = Network(
        vertical_distance_between_layers,
        horizontal_distance_between_neurons,
        neuron_radius,
        number_of_neurons_in_widest_layer,
        width_scale=width_scale,
    )

    # Add all layers in nn_l.
    weights_to_prev = None
    prev_layer_name = None
    iters = 0
    for name, param in nn_mfl.named_parameters():
        # Skip the biases.
        if "weight" not in name:
            continue
       # When we get to nn_h1, add the output layer and break.
        if "nn_l" not in name:
            name = 'y_l'
            network.add_layer(name, weights_to_prev.shape[0])
            network.connect_layers(
                prev_layer_name,
                name,
                weights_to_prev,
                np.zeros(weights_to_prev.shape)
            )
            prev_layer_name = name
            break
        # Don't add any weights until we have both layers in the network.
        if prev_layer_name:
            # Add the layer by name and size.
            network.add_layer(name, param.shape[1])
            # Add edges.
            network.connect_layers(
                prev_layer_name,
                name,
                weights_to_prev,
                np.zeros(weights_to_prev.shape)
            )
            # Updated previous weights and previous layer.
            weights_to_prev = param.data.numpy()
            prev_layer_name = name
        else:
            # Add the layer by name and size.
            network.add_layer(name, param.shape[1])
            # Update weights and previous layer.
            weights_to_prev = param.data.numpy()
            prev_layer_name = name

    # Add all layers in nn_h1 and nn_h2
    for name, param in nn_mfl.named_parameters():
        # Skip the biases.
        if "weight" not in name:
            continue
        if "nn_h1" in name and "0.weight" in name:
            nn_h1_layer0_num_neurons = param.shape[0]
            nn_h1_layer0_weights = param.data.numpy()[:, -output_dim].reshape(
                (nn_h1_layer0_num_neurons, output_dim)
            )
            nn_h1_x_weights = param.data.numpy()[:, 0:input_dim].reshape(
                (nn_h1_layer0_num_neurons, input_dim)
            )
        elif "nn_h2" in name and "0.weight" in name:
            nn_h2_layer0_num_neurons = param.shape[0]
            nn_h2_layer0_weights = param.data.numpy()[:, -output_dim].reshape(
                (nn_h2_layer0_num_neurons, output_dim)
            )
            nn_h2_x_weights = param.data.numpy()[:, 0:input_dim].reshape(
                (nn_h2_layer0_num_neurons, input_dim)
            )
            # Optionally nn_h2 may take additional inputs (such as a time-delayed
            # low-fidelity prediction), so we want to save these
            nn_h2_extra_weights = param.data.numpy()[:, input_dim:-output_dim].reshape(
                (nn_h2_layer0_num_neurons, param.shape[1]-input_dim-output_dim)
            )
            # Add the layer with name, size, and weights
            nn_h_layer0_weights = np.concatenate(
                (nn_h1_layer0_weights, nn_h2_layer0_weights), axis=0
            )
            network.add_layer(
                'nn_h_layer0',
                nn_h1_layer0_num_neurons + nn_h2_layer0_num_neurons
            )
            network.connect_layers(
                'y_l',
                'nn_h_layer0',
                nn_h_layer0_weights,
                np.zeros(nn_h_layer0_weights.shape)
            )
            # Update previous layer name for constructing the rest of nn_h2
            prev_layer_name = "nn_h_layer0"
        # After this point, only nn_h2 layers should remain to be added.
        elif "nn_h2" in name and "1.weight" in name:
            network.add_layer(name, param.shape[0])
            weights = param.data.numpy()
            weights = np.concatenate(
                (np.zeros((weights.shape[0], output_dim)), weights),
                axis=1
            )
            network.connect_layers(
                prev_layer_name,
                name,
                weights,
                np.zeros(weights.shape)
            )
            prev_layer_name = name
        elif "nn_h2" in name:
            network.add_layer(name, param.shape[0])
            network.connect_layers(
                prev_layer_name,
                name,
                param.data.numpy(),
                np.zeros(param.data.numpy().shape)
            )
            prev_layer_name = name
    # Draw connections between the first layer and the first high fidelity layer
    first_layer_nn_h_weights = np.concatenate(
        (nn_h1_x_weights, nn_h2_x_weights),
        axis=0
    )
    network.connect_layers(
        "nn_l.linear_layers.0.weight",
        "nn_h_layer0",
        first_layer_nn_h_weights,
        np.ones(first_layer_nn_h_weights.shape)
    )
    # Draw connections between F_l, F_nl and the final output
    network.add_layer('y_h', output_dim)
    weights = np.zeros((output_dim, len(network.layers['nn_h_layer0'].neurons)))
    weights[0:output_dim, 0:output_dim] = 1
    network.connect_layers(
        'nn_h_layer0',
        'y_h',
        weights*3,
        np.ones(weights.shape)
    )
    weights = np.ones((output_dim, output_dim))
    network.connect_layers(
        prev_layer_name,
        'y_h',
        weights*3,
        np.ones(weights.shape)
    )
    # Draw the neurons
    network.draw_neurons()

'''Generates plots for a given experiment. Provide:
   - nn_mfl: the (trained) model
   - train_data: the training data used
   - num_low: number of low_fidelity data points used
   - high_fi: a function to compute exact high-fidelity function value, given x
   - low_fi: a function to compute exact low-fidelity function value, given x
   - mesh_size (optional): how fine a mesh to use for plotting the exact and approximate functions.
   - range_loss_plot (optional): lower bound of range for plotting loss vs epochs
'''
def generate_plots(
    nn_mfl,
    train_data,
    num_low,
    high_fi,
    low_fi,
    losses,
    mesh_size=1000,
    range_loss_plot=None):

    plt.subplots(figsize=(10,8))

    '''Plot the data (a) in paper. Plots training data as well as exact low-
    and high-fidelity functions.'''
    ax1 = plt.subplot(2, 2, 1)
    x_test = torch.linspace(0, 1, mesh_size).view(mesh_size, 1)
    y_l = low_fi(x_test)
    y_h = high_fi(x_test)
    ax1.plot(x_test, y_h, 'k', linewidth=2)
    ax1.plot(x_test, y_l, 'grey', linewidth=2)
    inputs = torch.narrow(train_data, 1, 0, 1)
    ax1.plot(
        inputs[:][:num_low],
        low_fi(inputs[:][:num_low]),
        'bo',
        linewidth=5,
        markersize=8
    )
    ax1.plot(
        inputs[:][num_low:],
        high_fi(inputs[:][num_low:]),
        'rx',
        linewidth=5,
        markersize=8,
        mew=2
    )
    ax1.set_xlabel("x")

    '''Plot the approximation (c) in paper. Plots exact and approximate low-
    and high-fidelity functions.'''
    ax2 = plt.subplot(2, 2, 2)
    preds = nn_mfl(x_test)
    x_test = torch.linspace(0, 1, mesh_size).view(mesh_size, 1)
    ax2.plot(x_test, y_h, 'k', linewidth=2)
    ax2.plot(x_test, y_l, 'grey', linewidth=2)
    ax2.plot(
        x_test,
        torch.narrow(preds, 1, 0, 1).detach().numpy(),
        'b--',
        linewidth=2
    )
    ax2.plot(
        x_test,
        torch.narrow(preds, 1, 1, 1).detach().numpy(),
        'r--',
        linewidth=2
    )
    ax2.set_xlabel("x")
#     ax2.set_xlim(0, 1)
#     ax2.set_ylim(-1.5, 0)

    '''Plot the correlations (d) in paper'''
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3.grid(False)
    ax3.plot(
        y_h.view(mesh_size).numpy(),
        y_l.view(mesh_size).numpy(),
        x_test.view(mesh_size).numpy(),
        'k',
        linewidth=2
    )
    ax3.plot(
        torch.narrow(preds, 1, 1, 1).detach().view(mesh_size).numpy(),
        torch.narrow(preds, 1, 0, 1).detach().view(mesh_size).numpy(),
        x_test.view(mesh_size).numpy(),
        'r--',
        linewidth=2
    )
    ax3.plot(
        y_h.view(mesh_size).numpy(),
        y_l.view(mesh_size).numpy(),
        np.zeros(mesh_size),
        'grey',
        linewidth=2
    )
    ax3.plot(
        torch.narrow(preds, 1, 1, 1).detach().view(mesh_size).numpy(),
        torch.narrow(preds, 1, 0, 1).detach().view(mesh_size).numpy(),
        np.zeros(mesh_size),
        'b--',
        linewidth=2
    )
    ax3.set_xlabel("y_h")
    ax3.set_xlim(20, -20)
    ax3.set_ylabel("y_l")
    ax3.set_ylim(-10, 10)
    ax3.set_zlabel("x")
    ax3.set_zlim(0, 1)

    '''Plot the losses.'''
    ax4 = plt.subplot(2, 2, 4)
    if not range_loss_plot:
        ax4.plot(range(len(losses)), losses)
    else:
        ax4.plot(
            range(len(losses))[range_loss_plot:-1],
            losses[range_loss_plot:-1]
        )
    ax4.set_xlabel("epoch")
    ax4.set_xlabel("loss")
