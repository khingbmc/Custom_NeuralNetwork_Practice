import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle

class NeuralNetwork:
    def __init__(self, inputs, hiddens, outputs, w1, w2):
   
        self.n_input = inputs
        self.n_hidden = hiddens
        self.n_output = outputs
        self.network = []
        hidden_layer = w1 #random from num of inputlayer and hiddenlayer (input * hidden)
        self.network.append(hidden_layer)
        output_layer = w2
        self.network.append(output_layer)
        self.inputs = []
        self.data = []
        self.best_network = []
        self.testing = []
        
    def compute_net_input(self, weight, inputs):
        net_input = 0
     
        for i in range(len(weight)):
            net_input += weight[i]*inputs[i]
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

    def training(self, dataset, learn_rate, num_iteration, num_output, tenflow_iterate):
        number_testing = [35*(tenflow_iterate+1), 21*(tenflow_iterate+1)] if tenflow_iterate != 9 else [7+(35*(tenflow_iterate+1)), 2+(21*(tenflow_iterate+1))]
        testing = [[x for x in range(35*tenflow_iterate, number_testing[0])], [x for x in range(21*tenflow_iterate, number_testing[1])]]
        training = []
   
        for i in range(num_output):
#             for i in range(3):
#                 testing.append(start_index+i)
#                 testing.append(start_index+50+i)
#                 testing.append(start_index+100+i)

            for j in range(len(dataset[i])):
                if j not in testing[i]:
                    training.append(dataset[i][j])
                else:
                    self.testing.append(dataset[i][j])
                   
        
        for iterate in range(num_iteration):
            sum_error = 0
            

            for row in training:
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
      
        return self.inputs.index(max(self.inputs))

normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)

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

num_hidden = 10
num_inputs = len(data_key[0])-1 

num_outputs = len(set(data['class']))
print(num_inputs, num_outputs)

for i in range(len(data_key)):
    class_val = data_key[i][0]
    del data_key[i][0]
    data_key[i].append(1 if class_val == 'M' else 0)

#tenflow 35 and 21 and last iteration is 42 and 23

for i in range(len(data_key)):
    for j in range(len(max_val)):
        
        data_key[i][j] = normalized(data_key[i][j], max_val[j], min_val[j])

print("Number of Input Layer: ", num_inputs)
print("Number of Output Layer: ", num_outputs)


print(data_key[0])
shuffle(data_key)
input_data = [[] for _ in range(num_outputs)]
for i in data_key:
    if i[-1] == 0:
        input_data[0].append(i)
    else:
        input_data[1].append(i)
print(len(input_data[0]), len(input_data[1]))

networks = []
accuracy = []

weight1 = [{'weights':[random() for i in range(num_inputs)]} for i in range(num_hidden)]
weight2 = [{'weights':[random() for i in range(num_hidden)]} for i in range(num_outputs)]

for i in range(10):
    networks.append(NeuralNetwork(num_inputs, num_hidden, num_outputs, weight1, weight2))
    networks[i].training(input_data, 0.1, 500, num_outputs, i)

    
    num = 0
    for row in networks[i].testing:
        print("this is test: "), len(networks[i].testing)
        
        prediction = networks[i].predict(row)
        if row[-1] == prediction:
            num += 1
        print("Expect=%d  Output=%d" % (row[-1], prediction))
    if i != 9:
        accuracy.append(num/56*100)
    else :
        accuracy.append(num/65*100)

print(accuracy)

for i in range(10):
    print("Model ", i)
    # print(networks[i].network, end='\n\n')
    print("Accuracy : ", accuracy[i])

print("Mean Accuracy: " ,sum(accuracy)/10)
