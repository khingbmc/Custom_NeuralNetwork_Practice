import numpy as np
import pandas as pd
from random import random
import math

class NeuralNetwork:
    def __init__(self, inputs, hiddens, outputs):
        self.n_input = inputs
        self.n_hidden = hiddens
        self.n_output = outputs
        self.network = []
        hidden_layer = [{'weights':[random() for i in range(self.n_input)]} for i in range(self.n_hidden)] #random from num of inputlayer and hiddenlayer (input * hidden)
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(self.n_hidden)]} for i in range(self.n_output)]
        self.network.append(output_layer)
        self.inputs = []

    def compute_net_input(self, weight, input):
        net_input = 0
        for i in range(len(weight)):
            net_input += weight[i]*input[i]
        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0 + math.exp(-net_input))

    def forward_propagate(self, data):
        self.inputs = data
        for layer in self.network:
            next_inputs = []
            for neuron in layer:
                net_input = self.compute_net_input(neuron['weights'], data)
                neuron['output'] = self.sigmoid(net_input)
                next_inputs.append(neuron['output'])
            self.inputs = next_inputs

    #BackPropagation
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def back_propagate(self, expected):
        #backprop is begin in outputLayer
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1: #Hidden Layer
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += neuron['weights'][j] * neuron['errors']
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['errors'] = errors[j] * self.transfer_derivative(neuron['output'])
            
    

network = NeuralNetwork(2, 1, 2)
for layer in network.network:
    print(layer)
network.forward_propagate([1, 0])
print(network.inputs)
print(network.network)
print('=========')
network.back_propagate([0, 1])
print(network.network)
