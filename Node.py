import numpy as np


class Node:

    INPUT_NODE = "input"
    HIDDEN_NODE = "hidden"
    OUTPUT_NODE = "output"

    node_types = [INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE]

    TANH_ACTIVATION_FUNCTION = 1
    SIGMOID_ACTIVATION_FUNCTION = 2

    def __init__(self, node_id, layer_type, layer_id, activation_function):
        self.node_id = node_id
        self.node_type = layer_type
        self.layer_id = layer_id
        self.input_data = None
        self.output = None
        self.bias = None
        self.activation_function = activation_function
        self.generate_random_bias()

    def adjust_bias(self):
        if self.node_type == self.INPUT_NODE:
            self.bias = 0.0
        else:
            self.bias += np.random.uniform(-0.1, 0.1)

    def generate_random_bias(self):
        if self.node_type == self.INPUT_NODE:
            self.bias = 0.0
        else:
            self.bias = np.random.uniform(-1, 1)

    def __str__(self):
        return "Node id: " + str(self.node_id) + ", node type: " + self.node_type + ", layer id: " + str(self.layer_id) \
               + ", bias: " + str(self.bias) + ", input: " + str(self.input_data) + ", output: " + str(self.output)
