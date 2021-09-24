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
class Network():
    def __init__(
        self,
        vertical_distance_between_layers,
        horizontal_distance_between_neurons,
        neuron_radius,
        number_of_neurons_in_widest_layer,
        neuron_colors=("black", "green"),
        network_colors=("blue", "red"),
        width_scale=1,
    ):
        # Layers is a dictionary which keeps track of the number of
        # layers and the number of neurons in each layer.
        self.layers = {}
        self.vertical_distance_between_layers = vertical_distance_between_layers
        self.horizontal_distance_between_neurons = horizontal_distance_between_neurons
        self.neuron_radius = neuron_radius
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.neuron_colors = neuron_colors
        self.network_colors = network_colors
        self.width_scale=width_scale
        self.previous_layer = None

    ''' Function add_layer adds a new layer to the network, with a specified name and
        number of neurons, and, optionally, color to draw the neurons in. Returns the
        newly created layer. '''
    def add_layer(self, layer_name, number_of_neurons, neuron_colors=None):
        new_layer = Layer(
            number_of_neurons,
            layer_name,
            self.previous_layer,
            neuron_colors,
            self
        )
        self.layers[layer_name] = new_layer
        self.previous_layer = new_layer

    ''' Function __line_between_two_neurons draws the connection between two neurons
        using the specified edge color. '''
    def __line_between_two_neurons(self, neuron1, neuron2, linewidth, color_id):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius*.2 * sin(angle)
        y_adjustment = self.neuron_radius*.2 * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = plt.Line2D(
            line_x_data,
            line_y_data,
            linewidth=self.width_scale*linewidth,
            color=self.network_colors[color_id],
            zorder=5
        )
        plt.gca().add_line(line)

    ''' Function connect_layers connects two layers using the specified edge weights
        (to control thickness) and edge colors. '''
    def connect_layers(self, layer1_name, layer2_name, weights_matrix, color_ids_matrix):
        if layer1_name and layer2_name in self.layers:
            layer1 = self.layers[layer1_name]
            layer2 = self.layers[layer2_name]
        for (neuron2_index, neuron2) in enumerate(layer2.neurons):
            for (neuron1_index, neuron1) in enumerate(layer1.neurons):
                self.__line_between_two_neurons(
                    neuron1,
                    neuron2,
                    weights_matrix[neuron2_index, neuron1_index],
                    int(color_ids_matrix[neuron2_index, neuron1_index]),
                )

    ''' Function draw_neurons draws all neurons in the network.'''
    def draw_neurons(self):
        for (layer_key, layer) in self.layers.items():
            for neuron in layer.neurons:
                neuron.draw()
        plt.axis('scaled')
        plt.show()

class Layer():
    def __init__(
        self,
        number_of_neurons,
        name,
        previous_layer,
        neuron_colors,
        network
    ):
        self.name = name
        self.network = network
        self.previous_layer = previous_layer
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, neuron_colors)

    '''Initializes neurons in the layer.'''
    def __intialise_neurons(self, number_of_neurons, neuron_colors):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            if neuron_colors:
                neuron = Neuron(x, self.y, self.network, color=colors[iteration])
            else:
                neuron = Neuron(x, self.y, self.network)
            neurons.append(neuron)
            x += self.network.horizontal_distance_between_neurons
        return neurons

    '''Calculates y position for the layer's neurons'''
    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.network.vertical_distance_between_layers
        else:
            return 0

    '''Calculuates left margin to center the layers.'''
    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.network.horizontal_distance_between_neurons * \
            (self.network.number_of_neurons_in_widest_layer - number_of_neurons) \
            / 2

class Neuron():
    '''A Neuron instance draws itself in the correct position.'''
    def __init__(self, x, y, network, color="black"):
        self.x = x
        self.y = y
        self.color = color
        self.network = network

    def draw(self):
        circle = plt.Circle(
            (self.x, self.y),
            radius=self.network.neuron_radius,
            fill=True,
            color=self.color,
            zorder=10
        )
        plt.gca().add_patch(circle)
