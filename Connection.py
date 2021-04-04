from Node import Node
import numpy as np


class Connection:
    def __init__(self, connection_id, from_node, to_node, innovation_id):
        self.connection_id = connection_id
        self.innovation_id = innovation_id
        self.from_node = from_node
        self.to_node = to_node
        self.enabled = True
        self.weight = None
        self.generate_random_weight()

    def adjust_weight(self):
        self.weight += np.random.uniform(-0.1, 0.1)

    def generate_random_weight(self):
        self.weight = np.random.uniform(-1, 1)

    def __str__(self):
        return "Connection id: " + str(self.connection_id) + ", innovation id: " + str(self.innovation_id) + \
               ", from node: " + str(self.from_node) + ", to node: " + str(self.to_node) + ", enabled: " + \
               str(self.enabled) + ", weight: " + str(self.weight)
