import numpy as np
import pandas as pd
from random import random
from random import randint
import math
from random import seed
from random import shuffle


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

    def training(self, dataset, learn_rate, num_iteration, num_output, start_index):
        testing = []
        training = []
        for i in range(5):
            testing.append(start_index+i)
            testing.append(start_index+50+i)
            testing.append(start_index+100+i)

        for i in range(len(data_key)):
            if i not in testing:
                training.append(data_key[i])
            else:
                self.testing.append(data_key[i])
        
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

# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]

data = pd.read_csv("../wdbc.csv", index_col=0)
# data = pd.read_csv("../iris.csv")
num = 0
# max_val, min_val = [0 for i in range(4)], [0 for i in range(4)]
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


num_inputs = len(data_key[0])-1 
print(num_inputs)

# num_outputs = len(set([row[-1] for row in data_key]))
num_outputs = len(set(data['class']))

# print(data_key)




for i in range(len(data_key)):
    class_val = data_key[i][0]
    # del data_key[i][-1]
    del data_key[i][0]
    data_key[i].append(1 if class_val == 'M' else 0)
    # if(class_val == 'Iris-setosa'):
    #     data_key[i].append(0)
    # elif(class_val == 'Iris-versicolor'):
    #     data_key[i].append(1)
    # elif(class_val == 'Iris-virginica'):
    #     data_key[i].append(2)


# print(max_val)
# print(min_val[0])

for i in range(len(data_key)):
    for j in range(len(max_val)):
        
        data_key[i][j] = normalized(data_key[i][j], max_val[j], min_val[j])
print("After Normalized")


# for i in range(51):
#     print(data_key[i])
# num_inputs = len(data_key[0]) 
# num_outputs = len(set([row[-1] for row in data_key]))
print("Number of Input Layer: ", num_inputs)
print("Number of Output Layer: ", num_outputs)
# seed(1)

print(data_key[0])
shuffle(data_key)
print(data_key[0])
num0, num1 = 0, 0
for i in data_key:
    if(i[-1] == 0):
        num0+=1
    else:
        num1+=1

print(num0)
print(num1)
networks = []
accuracy = []
# network = NeuralNetwork(num_inputs, 10, num_outputs)
# for i in range(10):
#     networks.append(NeuralNetwork(num_inputs, 10, num_outputs))
#     networks[i].training(data_key, 0.1, 500, num_outputs, 5*i)
    
#     num = 0
#     for row in networks[i].testing:
#         prediction = networks[i].predict(row)
#         if row[-1] == prediction:
#             num += 1
#         print("Expect=%d  Output=%d" % (row[-1], prediction))
#     accuracy.append(num/15*100)


# for i in range(10):
#     print("Model ", i)
#     print(networks[i].network, end='\n\n')
#     print("Accuracy : ", accuracy[i])

# print("Mean Accuracy: " ,sum(accuracy)/10)
  
    

# B   M
#[_   _]
