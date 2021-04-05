from Connection import Connection
from Layer import Layer
import numpy as np
from Node import Node

class Model:

    def __init__(self, model_id, number_of_inputs, number_of_outputs, mutation_rate, list_of_innovations):
        self.model_id = model_id
        self.outputs = []
        self.score = 0.0
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.mutation_rate = mutation_rate
        self.list_of_all_nodes = []
        self.list_of_all_layers = []
        self.list_of_all_connections = []
        self.list_of_all_hidden_layers = []
        self.input_layer = None
        self.output_layer = None

        self.initialize_model(list_of_innovations)

    def initialize_model(self, list_of_innovations):
        input_layer = Layer(len(self.list_of_all_layers), Layer.INPUT_LAYER, self.number_of_inputs, self.list_of_all_nodes, self.list_of_all_connections, self.list_of_all_layers, list_of_innovations)
        self.list_of_all_layers.append(input_layer)
        self.input_layer = input_layer
        output_layer = Layer(len(self.list_of_all_layers), Layer.OUTPUT_LAYER, self.number_of_outputs, self.list_of_all_nodes, self.list_of_all_connections, self.list_of_all_layers, list_of_innovations, Node.SIGMOID_ACTIVATION_FUNCTION)
        self.list_of_all_layers.append(output_layer)
        self.output_layer = output_layer

    def regenerate_random_weights_bias(self):
        for layer in self.list_of_all_layers:
            layer.generate_new_weights()
            layer.generate_new_biases()

    def mutate(self, list_of_innovations, parent_model=None, parent_model_two=None):
        random_number = np.random.random()
        # x percent chance to mutate structure
        if random_number < self.mutation_rate:
            # print("Mutating structure on model", self.model_id)
            self.mutate_structure(list_of_innovations)
        else:
            # print("Mutating weights on model", self.model_id)
            for layer in self.list_of_all_layers:
                # each element of model has x percent chance to mutate
                layer.mutate(self.mutation_rate)

    def mutate_structure(self, list_of_innovations):
        nodes_connection_ratio = 0.7
        random_number = np.random.random()

        # create new connection or disable old
        if random_number <= nodes_connection_ratio:
            # print("Adding new connection")
            node_one = self.list_of_all_nodes[np.random.randint(0, len(self.list_of_all_nodes))]
            # first selected node cannot be an output node
            while node_one.node_type is Node.OUTPUT_NODE:
                node_one = self.list_of_all_nodes[np.random.randint(0, len(self.list_of_all_nodes))]
            node_two = self.list_of_all_nodes[np.random.randint(0, len(self.list_of_all_nodes))]
            # print("Node one selected:", node_one)
            # print("Node two selected:", node_two)
            # second selected node has to be:
            # 1. in lower layer to prevent reverse connections (exception: connection between input and output layers)
            # 2. different than first node
            # 3. of type hidden to prevent connection in the same layer for input/output layers
            # 4. other than input node as there can be only output connections from input layer
            while (node_two.layer_id > node_one.layer_id) \
                    or (node_two.node_id == node_one.node_id) \
                    or ((node_two.layer_id == node_one.layer_id) and (node_two.node_type is not Node.HIDDEN_NODE)) \
                    or (node_two.node_type is Node.INPUT_NODE):
                if node_two.layer_id > node_one.layer_id and (node_one.node_type == Node.INPUT_NODE
                                                              and node_two.node_type == Node.OUTPUT_NODE):
                    # print("Exception node one is input node two is output and node 2 layer is greater than node one")
                    break
                if (node_one.node_type is Node.INPUT_NODE) and (node_two.node_type is not Node.INPUT_NODE):
                    # print("Exception node one is input node two is other than input")
                    break
                # print("generate again node two")
                # print("Node one selected:", node_one)
                node_two = self.list_of_all_nodes[np.random.randint(0, len(self.list_of_all_nodes))]
                # print("Node selected for check", node_two)
            # print("Connection from:", node_one, ", to:", node_two)
            existing_connection_found = False
            for connection in self.list_of_all_layers[node_two.layer_id].list_of_layer_connections:
                if connection.from_node == node_one.node_id and connection.to_node == node_two.node_id:
                    # print("Existing connection found, mutating connection", connection, "to", (not connection.enabled))
                    if connection.enabled is True:
                        connection.enabled = False
                    else:
                        connection.enabled = True
                    existing_connection_found = True
                    break

            if not existing_connection_found:
                # new connection
                innovation_id = self.get_innovation_id(node_one, node_two, list_of_innovations)
                self.list_of_all_layers[node_two.layer_id].add_connection(node_one, node_two, innovation_id, self.list_of_all_connections)

        # create new node
        else:
            if len(self.list_of_all_hidden_layers) == 0:
                self.create_new_hidden_layer_with_node(list_of_innovations)
            else:
                # select random hidden layer or create a new one to add node
                random_number = np.random.random()
                if random_number > 0.3:
                    # print("Adding new node to existing layer")
                    hidden_layer = self.list_of_all_hidden_layers[np.random.randint(0, len(self.list_of_all_hidden_layers))]
                    # print("Hidden layer:", hidden_layer)
                    previous_layer = self.list_of_all_layers[hidden_layer.layer_id-1]
                    node = hidden_layer.add_new_node(self.input_layer, list_of_innovations, self.list_of_all_nodes, self.list_of_all_connections)
                    # print("New node:", node)
                    to_node = previous_layer.list_of_layer_nodes[np.random.randint(0, len(previous_layer.list_of_layer_nodes))]
                    previous_layer.add_connection(node, to_node, self.get_innovation_id(node, to_node, list_of_innovations), self.list_of_all_connections)
                else:
                    # new hidden layer
                    # print("Creating new hidden layer with new node")
                    self.create_new_hidden_layer_with_node(list_of_innovations)

    def create_new_hidden_layer_with_node(self, list_of_innovations):
        # create hidden layer with one new node
        # print("Adding new layer and new node")
        hidden_layer = Layer(len(self.list_of_all_layers), Layer.HIDDEN_LAYER, 1, self.list_of_all_nodes,
                             self.list_of_all_connections, self.list_of_all_layers, list_of_innovations)
        self.list_of_all_layers.append(hidden_layer)
        self.list_of_all_hidden_layers.append(hidden_layer)
        node = hidden_layer.list_of_layer_nodes[len(hidden_layer.list_of_layer_nodes) - 1]
        from_connections = hidden_layer.list_of_layer_connections
        # connect node to next layer
        previous_layer = self.list_of_all_layers[hidden_layer.layer_id - 1]
        to_node = previous_layer.list_of_layer_nodes[np.random.randint(0, len(previous_layer.list_of_layer_nodes))]
        previous_layer_connections = previous_layer.list_of_layer_connections
        previous_layer.add_connection(node, to_node, self.get_innovation_id(node, to_node, list_of_innovations), self.list_of_all_connections)

        # disable old connection
        for connection in previous_layer_connections:
            for from_connection in from_connections:
                if from_connection.from_node == connection.from_node and connection.to_node == to_node.node_id:
                    connection.enabled = False

    def get_innovation_id(self, node_from, node_to, list_of_innovations):
        new_connection = (node_from.node_id, node_to.node_id)
        if new_connection in list_of_innovations:
            return list_of_innovations.index(new_connection)
        list_of_innovations.append(new_connection)
        return len(list_of_innovations)

    def get_node_by_id(self, node_id):
        for node in self.list_of_all_nodes:
            if node_id == node.node_id:
                return node
        return None

    def feed_forward(self, input_data):
        self.outputs = []
        # input layer
        for input_index, input_node in enumerate(self.input_layer.list_of_layer_nodes):
            input_node.output = input_data[input_index]
        # hidden layers
        for hidden_index, hidden_layer in reversed(list(enumerate(self.list_of_all_hidden_layers))):
            for node in hidden_layer.list_of_layer_nodes:
                pass
        # output layer
        if len(self.list_of_all_layers) == 2:
            for output_index, node in enumerate(self.output_layer.list_of_layer_nodes):
                for output_conn_index, connection in enumerate(self.output_layer.list_of_layer_connections):
                    if connection.to_node == node.node_id:
                        input_node = self.get_node_by_id(connection.from_node)
                        node.input_data += input_node.output * connection.weight
                node.calculate_output()
                self.outputs.append(node.output)
        return self.outputs

    def fit(self, input_data, target):
        outputs = self.feed_forward(input_data)
        for output_index, output in enumerate(outputs):
            if output > 0.5:
                temp_output = 1.0
            else:
                temp_output = 0.0
            self.reward(output, temp_output, target[output_index])

    def reward(self, output, temp_output, target):
        if temp_output == target:
            # reward
            self.score += abs((2 * output) - 1)

    def __str__(self):
        return_string = "Model id: " + str(self.model_id) + ", number of layers: " + str(len(self.list_of_all_layers)) \
                        + ", score: " + str(self.score) + "\nLayers:\n"
        for layer in self.list_of_all_layers:
            return_string += layer.__str__() + "\n"
        return return_string
