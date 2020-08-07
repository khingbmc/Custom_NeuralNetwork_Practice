import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle
import statistics

class NeuralModel:
    def __init__(self, number_input_layer, number_hidden_layer, number_output_layer, weight_input_hidden, weight_hidden_output):
        self.number_inputs_layer = number_input_layer
        self.number_hidden_layer = number_hidden_layer
        self.number_output_layer = number_output_layer
        self.network = []
        self.network.append(weight_input_hidden)
        self.network.append(weight_hidden_output)
        self.inputs = []
        self.data = []
        self.testing = []
        self.num_class = -1 #Store class data of each row is training now
        self.iteration = 0 #Store round of loop training data
        self.phase = 'Training' #Checking phase of datamining
        self.hidden_data = []
        self.check = []

        file = open("../../newWay/iris_init_weight_1"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()


    def compute_net_input(self, weight, inputs, layer, neuron):
        net_input = 0
        self.hidden_data = []
        for i in range(len(weight)):
            if layer == 0:
                if self.phase == 'Training':
                    net_input += weight[i][round(inputs[i], 1)]*inputs[i]
                else:
                    hidden_key = list(weight[i].keys())
                    hidden_key.sort()
                    x1, x2 = round(inputs[i], 1), float()
                    if inputs[i] > hidden_key[0] and inputs[i] < hidden_key[-1]:
                        x2 = round(round(inputs[i], 1) - 0.1, 1) if inputs[i] < round(inputs[i], 1) else round(round(inputs[i], 1) + 0.1, 1)
                    else:
                        x1 = hidden_key[0] if inputs[i] <= hidden_key[0] else hidden_key[1]
                        x2 = hidden_key[0] + 0.1 if inputs[i] <= hidden_key[0] else round(hidden_key[1] - 0.1, 1)
                    m = (weight[i][x2] - weight[i][x1])/(x2-x1)
                    c = (-x1*m)+weight[i][x1]
                    net_input += self.select_weight(m, inputs[i], c)
            else:
                if self.phase == 'Training':
                    if round(inputs[i], 2) not in weight[i].keys():
                        self.network[layer][neuron]['weights'][i][round(inputs[i], 2)] = self.network[layer][neuron]['weights'][i]['default']
                    net_input += self.network[layer][neuron]['weights'][i][round(inputs[i], 2)]*inputs[i]

                else:
                    hidden_key = list(weight[i].keys())
                    hidden_key.remove('default')
                    hidden_key.sort()
                    x1, x2 = round(inputs[i], 1), float()
                    if inputs[i] > hidden_key[0] and inputs[i] < hidden_key[-1]:
                        for j in range(len(hidden_key)):
                            if hidden_key[j] <= inputs[i]:
                                continue
                            else:
                                x2 = hidden_key[j]
                                x1 = hidden_key[j-1]
                                break
                    else:
                        x1 = hidden_key[0] if inputs[i] <= hidden_key[0] else hidden_key[-2]
                        x2 = hidden_key[1] if inputs[i] <= hidden_key[0] else hidden_key[-1]
                    m = (weight[i][x2] - weight[i][x1])/(x2-x1)
                    c = (-x2*m)+weight[i][x2]
                    net_input += self.select_weight(m, inputs[i], c)
                    file = open("../../newWay/check_use_weight"+".txt", "a")
                    file.write(str(x1)+"  ")
                    file.write(str(x2)+"  ")
                    file.write(str(inputs[i])+"  ")
                    file.write(str(hidden_key)+" \n\n\n")
                    file.close()

        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0+math.exp(-net_input))

    def forward_propagation(self, data):
        self.data = data
        self.inputs = data
        self.check = []
        check = []
        for layer in range(len(self.network)):
            next_inputs = []
            for neuron in range(len(self.network[layer])):
                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, layer, neuron)
                self.network[layer][neuron]['output'] = self.sigmoid(net_input)
                if layer == 0:
                    check.append(self.sigmoid(net_input))
                next_inputs.append(self.network[layer][neuron]['output'])
            self.inputs = next_inputs
        self.check = check

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def back_propagation(self, expected):
        # Begin at output layer (Backpropagation Steps)
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += neuron['weights'][j][round(self.check[j], 2)] * neuron['errors']
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['errors'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, learning_rate):
        inputs = self.data[:-1]
        for i in range(len(self.network)):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            # print(inputs)
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    # if i != 0:
                    if i != 0:
                        neuron['weights'][j][round(inputs[j], 2)] += learning_rate * neuron['errors'] * inputs[j]
                    else:
                        neuron['weights'][j][round(inputs[j], 1)] += learning_rate * neuron['errors'] * inputs[j]

    def training(self, dataset, learn_rate, num_iteration, num_output, test):

        self.testing = test
        for iterate in range(int(num_iteration)):
            self.iteration = iterate
            sum_error = 0
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagation(row)
                if iterate != num_iteration:
                    expected = [0 for i in range(num_output)]
                    expected[int(row[-1])] = 1

                    sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])

                    self.back_propagation(expected)
                    self.update_weights(learn_rate)

            if iterate != num_iteration:
                print('iteration=%d   learning_rate=%s   rmse=%.4f' % (iterate, str(learn_rate), math.sqrt(sum_error)))

        file = open("../../newWay/end_weight"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()

    def predict(self, row):
        self.phase = 'Testing'
        self.forward_propagation(row)
        return self.inputs

    def select_weight(self, m, x, c):
        return (m*x)+c

