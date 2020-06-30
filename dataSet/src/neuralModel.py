import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle
import statistics

class TestNeural:
    def __init__(self, inputs, hiddens, outputs, w1, w2):
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.network = []
        hidden_layer = w1
        output_layer = w2
        self.network.append(hidden_layer)
        self.network.append(output_layer)
        self.inputs = []
        self.data = []
        self.best_network = []
        self.testing = []
        self.num_class = -1
        self.iteration = 0
        self.check_all_weight = [[{'output':[]} for i in range(self.hiddens)] for i in range(self.outputs)]
        self.condition = [{} for _ in range(self.hiddens)]
        self.check_condition = False
        self.gaussian = []
        self.checking = False
      
        file = open("test_iris/init_weight"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()
        

    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        num = 0

        for i in range(len(weight)):

            net_input += weight[i]*inputs[i]

        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0+math.exp(-net_input))

    def forward_propagate(self, data):
        self.inputs = data
        self.data = data

        num = 0
        for layer in range(len(self.network)):
            next_inputs = []
            for neuron in range(len(self.network[layer])):

                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, num)
                self.network[layer][neuron]['output'] = self.sigmoid(net_input)

                next_inputs.append(self.network[layer][neuron]['output'])

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
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            # print(inputs)
            for neuron in self.network[i]:
                for j in range(len(inputs)):

                    neuron['weights'][j] += learn_rate * neuron['errors'] * inputs[j]
                    neuron['weights'][-1] += learn_rate * neuron['errors']


    def training(self, dataset, learn_rate, num_iteration, num_output, test):

        self.testing = test
        for iterate in range(int(num_iteration)+1):
            self.iteration = iterate
            sum_error = 0
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagate(row)
                if iterate != num_iteration:
                    expected = [0 for i in range(num_output)]
                    expected[int(row[-1])] = 1

                    sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])

                    self.back_propagate(expected)
                    self.update_weights(learn_rate)

            if iterate != num_iteration:
                print('iteration=%d   learning_rate=%.4f   rmse=%.4f' % (iterate, learn_rate, math.sqrt(sum_error)))
        
        file = open("test_iris/end_weight"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()
        return self.check_all_weight

    def predict(self, row):
        
        self.forward_propagate(row)
        return self.inputs
