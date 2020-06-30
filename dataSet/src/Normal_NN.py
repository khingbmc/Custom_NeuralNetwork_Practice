import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle
import statistics

class NormalNeural:
    def __init__(self, input, hiddens, outputs, w1, w2):
        self.inputs = input
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
        self.iteration = 0

    def training(self, dataset, learn_rate, num_iteration, num_output, test):
        self.testing = test
        for iterate in range(int(num_iteration)+1):
            self.iteration = iterate
            sum_error = 0
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagate(row)
                expected = [0 for i in range(num_output)]
                expected[row[-1]] = 1
                # print("this is expect ", expected)
    
                sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])
                    
                self.back_propagate(expected)
                self.update_weights(learn_rate)
            print('iteration=%d   learning_rate=%.4f   error=%.4f' % (iterate, learn_rate, sum_error))
            return self.check_all_weight

    def compute_net_input(self, weight, inputs, layer):
        net_input = 0
        num = 0
           
        for i in range(len(weight)):
            
     
            if type(weight[i]) == float or type(weight[i]) == np.float64:
                
                net_input += weight[i]*inputs[i]
            
            else:
#%% Used gaussian to predict weight                
                if self.check_condition: 
                
                    gaussian_answer = []
                    for j in range(self.outputs):
                        mean, std = self.condition[i][j]['mean'], self.condition[i][j]['std']
                        gaussian_answer.append(self.gaussian_function(mean, std, inputs[i]))
                    weight_used = gaussian_answer.index(max(gaussian_answer))
                    net_input += weight[i][weight_used]*inputs[i]
# %% Normal Training
                else:
                    net_input += weight[i][self.num_class]*inputs[i]
                
               
        return net_input

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


    def predict(self, row):
        
        self.forward_propagate(row)
        return self.inputs