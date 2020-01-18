import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle

class TestNeural:
    def __init__(self, inputs, hiddens, outputs, num_class):
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.network = []
        hidden_layer = [{'weights': [0.5 for i in range(self.inputs)]} for i in range(self.hiddens)]
        output_layer = [{'weights':[0.5 for i in range(self.hiddens)]} for i in range(self.outputs)]
        self.network.append(hidden_layer)
        self.network.append(output_layer)
        self.inputs = []
        self.data = []
        self.best_network = []
        self.testing = []
        self.num_class = num_class
        file = open("test_iris/init_weight"+str(self.num_class)+".txt", "w")
        file.write(str(self.network))
        file.close()
        

    def compute_net_input(self, weight, inputs):
        net_input = 0
        for i in range(len(weight)):
            net_input += weight[i]*inputs[i]
        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0+math.exp(-net_input))

    def forward_propagate(self, data):
        self.inputs = data
        self.data = data
        for layer in self.network:
            next_inputs = []
            for neuron in layer:
                net_input = self.compute_net_input(neuron['weights'], self.inputs)
                neuron['output'] = self.sigmoid(net_input)
                next_inputs.append(neuron['output'])
            self.inputs = next_inputs

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

    def update_weights(self, learn_rate):
        for i in range(len(self.network)):
            inputs = self.data[:-1]
            # print(inputs)
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += learn_rate * neuron['errors'] * inputs[j]
                neuron['weights'][-1] += learn_rate * neuron['errors']

    def training(self, dataset, learn_rate, num_iteration, num_output):

        for iterate in range(num_iteration):
            sum_error = 0
            for row in dataset:
                self.forward_propagate(row)
                expected = [0 for i in range(num_output)]
                expected[row[-1]] = 1
                # print("this is expect ", expected)

                sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])
                
                self.back_propagate(expected)
                self.update_weights(learn_rate)
           
            print('iteration=%d   learning_rate=%.4f   error=%.4f' % (iterate, learn_rate, sum_error))
        file = open("test_iris/end_weight"+str(self.num_class)+".txt", "w")
        file.write(str(self.network))
        file.close()
