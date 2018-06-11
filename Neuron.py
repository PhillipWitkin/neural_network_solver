import copy
import math
import random

class Neuron:

    INPUT1 = "InputNode1"
    INPUT2 = "InputNode2"

    FORWARDS = "forwards"
    BACKWARDS = "backwards"

    def __init__(self, name):
        # self.sigmoid = 0
        self.yValue = Neuron.random_number()
        self.threshold = Neuron.random_number()
        self.delta = 0
        self.ahead_of_nodes = []
        self.behind_nodes = []
        self.weight_to_nodes = {}
        self.name = name

    def add_node_ahead(self, *args):
       for node in args:
          self.behind_nodes.append(node)



    def add_node_behind(self, *args):
        for node in args:
            self.ahead_of_nodes.append(node)


    def get_weight_to_node(self, to_node):
        return self.weight_to_nodes.get(to_node.name)

    def set_weight_to_node(self, to_node, value):
        self.weight_to_nodes[to_node.name] = value



    def display_weights(self):
        # output = self.name
        for name, value in self.weight_to_nodes.items():
            print(self.name + " to " + name + " = " +  str(value))


     # provides a random number for weights and thresholds
    @staticmethod
    def random_number():
        max = 2.4/2
        min = -2.4/2
        return min + (max - min)*random.random()

    @staticmethod
    def sigmoid(num):
        return 1 / (1 + math.exp(-num))





