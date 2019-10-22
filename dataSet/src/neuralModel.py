import numpy as np
import pandas as pd
from random import random
import math
from random import seed

class NeuralNetwork:
    def __init__(self, inputs, hiddens, outputs):
        print(inputs)
        self.n_input = inputs
        self.n_hidden = hiddens
        self.n_output = outputs
        self.network = []
        hidden_layer = [{'weights':[random() for i in range(self.n_input)]} for i in range(self.n_hidden)] #random from num of inputlayer and hiddenlayer (input * hidden)
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(self.n_hidden)]} for i in range(self.n_output)]
        self.network.append(output_layer)
        self.inputs = []
        self.data = []
        self.best_network = []

    def compute_net_input(self, weight, input):
        net_input = 0
        for i in range(len(weight)):
            net_input += weight[i]*input[i]
        return net_input

    def sigmoid(self, net_input):
        return 1.0/(1.0 + math.exp(-net_input))

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

    def predict(self, row):
        self.forward_propagate(row)
        print(self.inputs)
        return self.inputs.index(max(self.inputs))
    
normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

data = pd.read_csv("../wdbc.csv", index_col=0)
num = 0
max_val, min_val = [0 for i in range(30)], [0 for i in range(30)]
for i in data:
    if(i != 'class'):
        if(num == 30):
            break
        max_val[num] = max(data[i])
        min_val[num] = min(data[i])
        num += 1

ID = data.index.values
data_key = []
for j in ID:
    format_data = []
    for i in data:
        format_data.append(data[i][j])
    data_key.append(format_data)

num_inputs = len(data_key[0]) 
num_outputs = len(set([row[-1] for row in data_key]))
print(num_inputs)





for i in range(len(data_key)):
    class_val = data_key[i][0]
    del data_key[i][0]
    data_key[i].append(1 if class_val == 'M' else 0)

print(data_key[0])
# print(max_val)
# print(min_val[0])

for i in range(len(data_key)-1):
    for j in range(len(max_val)):
        
        data_key[i][j] = normalized(data_key[i][j], max_val[j], min_val[j])
print("After Normalized")
print(data_key[0])


num_inputs = len(data_key[0]) 
num_outputs = len(set([row[-1] for row in data_key]))
# print(num_inputs)
# print(num_outputs)
# seed(1)
network = NeuralNetwork(num_inputs, 3, num_outputs)
network.training(data_key, 0.5, 5000, num_outputs)
print("\n\nModel")
print(network.network)

for row in data_key:
    prediction = network.predict(row)
    print("Expect=%d  Output=%d" % (row[-1], prediction))

# B   M
#[_   _]
