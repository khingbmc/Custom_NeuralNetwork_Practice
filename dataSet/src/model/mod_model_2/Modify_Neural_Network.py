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
        self.phase = 'Training'
        self.hidden_data = []

        file = open("../../newWay/iris_init_weight_1"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()


    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        num = 0
        self.hidden_data = []
        for i in range(len(weight)):
            if self.phase == 'Training':
                if round(inputs[i], 2) not in weight[i].keys() and layer == 1:
                    self.hidden_data.append(round(inputs[i], 2))
                    net_input += weight[i]['base']*inputs[i]
                else:
                    net_input += weight[i][round(inputs[i], 1)]*inputs[i]
            else:
                hidden_key = list(weight[i].keys())
                if layer == 1:
                    hidden_key.remove('base')
                    hidden_key.sort()
                    x1, x2 = round(inputs[i], 2), float()
                    if inputs[i] > hidden_key[0] and inputs[i] < hidden_key[-1]:
                        x2 = round(round(inputs[i], 2) - 0.01, 2) if inputs[i] < round(inputs[i], 2) else round(round(inputs[i], 2) + 0.01, 2)
                    else:
                        print(inputs[i])
                        x1 = hidden_key[0] if inputs[i] <= hidden_key[0] else hidden_key[1]
                        x2 = round(hidden_key[0] + 0.01, 2) if inputs[i] <= hidden_key[0] else round(hidden_key[1] - 0.01, 2)
                    m = (weight[i][x2] - weight[i][x1])/(x2-x1)
                    c = (-x1*m)+weight[i][x1]
                    net_input += self.select_weight(m, inputs[i], c)
                else:
                    hidden_key.sort()
                    x1, x2 = round(inputs[i], 1), float()
                    if inputs[i] > hidden_key[0] and inputs[i] < hidden_key[-1]:
                        x2 = round(round(inputs[i], 1) - 0.1, 1) if inputs[i] < round(inputs[i], 1) else round(round(inputs[i], 1) + 0.1, 1)
                    else:
                        print(inputs[i])
                        x1 = hidden_key[0] if inputs[i] <= hidden_key[0] else hidden_key[1]
                        x2 = hidden_key[0] + 0.1 if inputs[i] <= hidden_key[0] else round(hidden_key[1] - 0.1, 1)
                    m = (weight[i][x2] - weight[i][x1])/(x2-x1)
                    c = (-x1*m)+weight[i][x1]
                    net_input += self.select_weight(m, inputs[i], c)
            # else:
            #     if self.phase == 'Training':
            #         net_input += weight[i][round(inputs[i], 1)]*inputs[i]
            #     else:
            #         print(inputs[i])
            #         print(inputs[i] > 0.0 and inputs[i] < 1.0)
            #         x1, x2 = round(inputs[i], 1), float()
            #         if inputs[i] > 0.0 and inputs[i] < 1.0:
            #             x2 = round(round(inputs[i], 1) - 0.1, 1) if inputs[i] < round(inputs[i], 1) else round(round(inputs[i], 1) + 0.1, 1)
            #         else:
            #             x1 = 0.0 if inputs[i] <= 0.0 else 1.0
            #             x2 = 0.1 if inputs[i] <= 0.0 else 0.9
            #         m = (weight[i][x2] - weight[i][x1])/(x2-x1)
            #         c = (-x1*m)+weight[i][x1]
            #         net_input += self.select_weight(m, inputs[i], c)



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

                net_input = self.compute_net_input(self.network[layer][neuron]['weights'], self.inputs, layer)
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
            # print(i)
            if i != len(self.network) - 1: #Hidden Layer
                for j in range(len(layer)):
                    error = 0.0
                    # print(self.hidden_data[j])
                    # print('\n\n')
                    # print(layer[j]['output'])
                    for neuron in self.network[i + 1]:
                        # print(neuron)
                        # print(j)
                        # print(self.hidden_data)
                        # print(len(self.hidden_data))
                        if round(layer[j]['output'], 1) not in neuron['weights'][j].keys():
                            error += neuron['weights'][j]['base'] * neuron['errors']
                        else:
                            error += neuron['weights'][j][round(layer[j]['output'], 1)] * neuron['errors']
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['errors'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, learn_rate):
        # print(self.num_class)
        # print(learn_rate)
        inputs = self.data[:-1]
        for i in range(len(self.network)):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            # print(inputs)
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    # if i != 0:
                    if round(inputs[j], 1) not in neuron['weights'][j].keys():
                        neuron['weights'][j][round(inputs[j], 1)] = neuron['weights'][j]['base']
                    neuron['weights'][j][round(inputs[j], 1)] += learn_rate * neuron['errors'] * inputs[j]
                    # neuron['weights'][-1][round(inputs[j], 1)] += learn_rate * neuron['errors']


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
                print('iteration=%d   learning_rate=%s   rmse=%.4f' % (iterate, str(learn_rate), math.sqrt(sum_error)))

        file = open("../../newWay/iris_end_weight_1"+".txt", "a")
        file.write(str(self.network)+"\n\n")
        file.close()
        return self.check_all_weight

    def predict(self, row):
        self.phase = 'Testing'
        self.forward_propagate(row)
        return self.inputs

    def select_weight(self, m, x, c):
        return (m*x)+c