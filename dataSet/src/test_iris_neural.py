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
      
        file = open("test_iris/init_weight"+".txt", "w")
        file.write(str(self.network))
        file.close()
        

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
                        # gaussian_answer.append(self.condition[i][j](inputs[i]))
                        # self.gaussian.append(self.condition[i][j](inputs[i]))
                        
                    
                        
                    weight_used = gaussian_answer.index(max(gaussian_answer))
                    net_input += weight[i][weight_used]*inputs[i]
                    
                    
# %% Normal Training
                else:
                    net_input += weight[i][self.num_class]*inputs[i]
                
               
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
                if layer == 0 and self.iteration == 500:
                    
                    output = self.network[layer][neuron]['output']
                    self.check_all_weight[self.num_class][neuron]['output'].append(output)
                    # self.check_all_weight[self.num_class][neuron]['min'] = output if self.check_all_weight[self.num_class][neuron]['min'] > output else self.check_all_weight[self.num_class][neuron]['min'] 
                    # self.check_all_weight[self.num_class][neuron]['max'] = output if self.check_all_weight[self.num_class][neuron]['max'] < output else self.check_all_weight[self.num_class][neuron]['max']  
                    
                   
                    
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
                        if type(neuron['weights'][j]) == float or type(neuron['weights'][j]) == np.float64:
                            error += neuron['weights'][j] * neuron['errors']
                        else:
                            error += neuron['weights'][j][self.num_class] * neuron['errors']
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
                    if type(neuron['weights'][j]) == float or type(neuron['weights'][j]) == np.float64:
                        neuron['weights'][j] += learn_rate * neuron['errors'] * inputs[j]
                        neuron['weights'][-1] += learn_rate * neuron['errors']
                    else:
                       
                        neuron['weights'][j][self.num_class] += learn_rate * neuron['errors'] * inputs[j]
                        neuron['weights'][-1][self.num_class] += learn_rate * neuron['errors']
                

    def training(self, dataset, learn_rate, num_iteration, num_output, test):
        
        self.testing = test
        for iterate in range(num_iteration+1):
            self.iteration = iterate
            sum_error = 0
            for row in dataset:
                self.num_class = row[-1]
                self.forward_propagate(row)
                if iterate != num_iteration:
                    expected = [0 for i in range(num_output)]
                    expected[row[-1]] = 1
                    # print("this is expect ", expected)
    
                    sum_error += sum([(expected[i] - self.inputs[i])**2 for i in range(len(expected))])
                    
                    self.back_propagate(expected)
                    self.update_weights(learn_rate)
           
            if iterate != num_iteration:
                print('iteration=%d   learning_rate=%.4f   error=%.4f' % (iterate, learn_rate, sum_error))
        
        file = open("test_iris/end_weight"+".txt", "w")
        file.write(str(self.network))
        file.close()
        return self.check_all_weight
    
    def create_condition(self):
        print("this is check all weight")
        print(self.check_all_weight, end = '\n\n\n')
        for i in range(len(self.check_all_weight)):
            for j in range(len(self.check_all_weight[i])):
                print(i ,j)
                print(self.check_all_weight[i][j]['output'])
                mean = statistics.mean(self.check_all_weight[i][j]['output'])
                std = statistics.stdev(self.check_all_weight[i][j]['output'], xbar=mean) 
                print("Mean Std")
                print(mean, std, end='\n\n\n')
                self.condition[j][i] = {'mean':mean, 'std':std}
                
        self.check_condition = True
        self.iteration = 0
        print(self.check_all_weight)
        return self.condition
    
    def predict(self, row):
        
        self.forward_propagate(row)
        return self.inputs

    def gaussian_function(self, means, std, output):
        return (1/(std*math.sqrt(2*math.pi)))*math.exp((-1/2)*((output-means)/std)**2)
                
        
    
    
    
    
    
    
