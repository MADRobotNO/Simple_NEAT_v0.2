from Node import Node
from Connection import Connection
import numpy as np


class Layer:

    INPUT_LAYER = "input"
    HIDDEN_LAYER = "hidden"
    OUTPUT_LAYER = "output"

    INPUT_LAYER_ID = 0
    OUTPUT_LAYER_ID = 1

    def __init__(self, layer_id, layer_type, number_of_nodes, list_of_all_nodes, list_of_all_connections, list_of_all_layers, list_of_innovations, activation_function=Node.TANH_ACTIVATION_FUNCTION):
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.number_of_nodes = number_of_nodes
        self.activation_function = activation_function
        self.list_of_layer_nodes = []
        self.list_of_layer_connections = []

        self.initialize_layer(list_of_all_nodes, list_of_all_connections, list_of_all_layers, list_of_innovations)

    def initialize_layer(self, list_of_all_nodes, list_of_all_connections, list_of_all_layers, list_of_innovations):
        self.initialize_nodes(list_of_all_nodes)

        if self.layer_type != self.INPUT_LAYER:
            self.initialize_connections(list_of_all_connections, list_of_all_layers, list_of_innovations)

    def initialize_nodes(self, list_of_all_nodes):
        for i in range(self.number_of_nodes):
            node = Node(len(list_of_all_nodes), self.layer_type, self.layer_id, self.activation_function)
            list_of_all_nodes.append(node)
            self.list_of_layer_nodes.append(node)

    def initialize_connections(self, list_of_all_connections, list_of_all_layers, list_of_innovations):
        # layer nearest to input layer, connecting to input
        for input_node in list_of_all_layers[self.INPUT_LAYER_ID].list_of_layer_nodes:
            for current_node in self.list_of_layer_nodes:
                innovation_id = self.get_innovation_id(list_of_innovations, input_node.node_id, current_node.node_id)
                connection = Connection(len(list_of_all_connections), input_node.node_id, current_node.node_id, innovation_id)
                list_of_all_connections.append(connection)
                self.list_of_layer_connections.append(connection)

    def get_innovation_id(self, list_of_innovations, from_node_id, to_node_id):
        new_connection = (from_node_id, to_node_id)
        for index, innovation in enumerate(list_of_innovations):
            if innovation == new_connection or reversed(innovation) == new_connection:
                return index
        list_of_innovations.append(new_connection)
        return list_of_innovations.index(new_connection)

    def generate_new_weights(self):
        for connection in self.list_of_layer_connections:
            connection.generate_random_weight()

    def generate_new_biases(self):
        for node in self.list_of_layer_nodes:
            node.generate_random_bias()

    def add_connection(self, from_node, to_node, innovation_id, list_of_all_connections):
        new_connection = Connection(len(list_of_all_connections), from_node.node_id, to_node.node_id, innovation_id)
        self.list_of_layer_connections.append(new_connection)
        list_of_all_connections.append(new_connection)
        return new_connection

    def add_new_node(self, input_layer, list_of_innovations, list_of_all_nodes, list_of_all_connections):
        node = Node(len(list_of_all_nodes), self.layer_type, self.layer_id, self.activation_function)
        for from_node in input_layer.list_of_layer_nodes:
            innovation_id = self.get_innovation_id(list_of_innovations, from_node.node_id, node.node_id)
            self.add_connection(from_node, node, innovation_id, list_of_all_connections)
        self.list_of_layer_nodes.append(node)
        list_of_all_nodes.append(node)
        return node

    def mutate(self, mutation_rate):
        for node in self.list_of_layer_nodes:
            # x % chance to adjust bias
            if np.random.random() < mutation_rate:
                node.adjust_bias()
            # x % chance to change bias
            elif np.random.random() < mutation_rate:
                node.generate_random_bias()
        for connection in self.list_of_layer_connections:
            # x % chance to adjust weight
            if np.random.random() < mutation_rate:
                connection.adjust_weight()
            # x % chance to change weight
            elif np.random.random() < mutation_rate:
                connection.generate_random_weight()

    def __str__(self):
        return_string = "Layer id: " + str(self.layer_id) + ", layer type: " + self.layer_type + ", number of connections: " \
                        + str(len(self.list_of_layer_connections)) + ", number of nodes: " + str(len(self.list_of_layer_nodes)) + "\n"
        return_string += "Nodes\n"
        if len(self.list_of_layer_nodes) > 0:
            for node in self.list_of_layer_nodes:
                return_string += node.__str__() + "\n"
        else:
            return_string += "No nodes in layer!\n"
        return_string += "Connections:\n"
        if len(self.list_of_layer_connections) > 0:
            for connection in self.list_of_layer_connections:
                return_string += connection.__str__() + "\n"
        else:
            return_string += "No connections!\n"
        return return_string
