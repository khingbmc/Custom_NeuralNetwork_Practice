import numpy as np
import pandas as pd
from random import random 
from random import randint
import math
from random import seed 
from random import shuffle
import statistics

class NeuralNetwork:
#%% Initialize Neural Class
    def __init__(self, inputs, hiddens, ouputs, weight1, weight2):
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.network = []
        self.network.append(weight1)
        self.network.append(weight2)
        self.inputs = []
        self.num_class = int()

        self.iteration = int()
        self.check_status_hidden = [[{'mean' : 0, 'min' : 1000, 'max' : 0, 'std' : []} for i in range(self.hiddens)] for i in range(self.outputs)]
        self.condition = [[] for _ in range(self.hiddens)]
        self.check_condition = False

#%% activation function
    def sigmoid(self, net_input):
        return 1.0/(1.0+math.exp(-net_input))

    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        num = 0

        for i in range(len(weight)):

            if(type(weight[i])) == float or type(weight[i]) == np.float64:
                net_input += weight[i]*inputs[i]

            else:

                if self.check_condition:
                    num += 1
                    gaussian_answer = []

                    for j in range(self.outputs):
                        gaussian_answer.append(self.condition[i][j](inputs[i]))

                    weight_used = gaussian_answer.index(max(gaussian_answer))
                    net_input += weight[i][weight_used] * inputs[i]
                else:
                    net_input += weight[i][self.num_class] * inputs[i]
        return net_input

    def forward_propagation(self, data):
        self.inputs = data
        self.num_class = self.inputs[-1]
        for layer in range(len(self.network)):
            next_inputs = []
            
            for neuron in range(len(self.network[layer])):
                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, layer, neuron)

                self.network[layer][neuron]['output'] = self.sigmoid(net_input)
                if layer == 0 and self.iteration == 500:
                    output = self.network[layer][neuron]['output']
                    