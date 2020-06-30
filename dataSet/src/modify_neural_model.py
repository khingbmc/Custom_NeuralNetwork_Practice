import pandas as pd
import math
import statistics
import numpy as np

from random import randint, shuffle, random

# Create Class Neural Network
class NeuralNetwork:
# %% Class Constructor
    def __init__(self, number_node_input, number_node_hidden, number_node_output, weight_input_hidden, weight_hidden_output):
        self.networks = [weight_input_hidden, weight_hidden_output]
        self.number_node_input = number_node_input
        self.number_node_output = number_node_output
        self.number_node_hidden = number_node_hidden
        self.datarow = []
        self.inputs_data = []
        self.gaussian = [] # use this array to collect the gaussian elimination each connection of hidden to output layer (because all of this connection is multi-connection in each node)
        hidden_output = [{'output':[]} for _ in range(number_node_hidden)]
        self.hidden_layer_output = [hidden_output for _ in range(number_node_output)] # use this array to calculate statistics (max, min, mean, std) of data output on hidden layer
        self.class_condition = [[[] for _ in range(self.number_node_output)] for _ in range(self.number_node_hidden)]
        self.created_gaussian_checking = False
        self.end_of_tenfold = False
        self.num = -1
        file = open('nw_init.txt', 'a')
        file.write(str(self.networks))
        file.close()



# %% Computation network input (Before compute in every node)
    def compute_network_input(self, weights, data_input, data_class, number_layer):
        network_inputs = 0
        # add in function guassian (if it is hidden - output layer)
        for node in range(len(weights)):
            #check weight layer
            if(type(weights[node]) == float or type(weights[node]) == np.float64):
                network_inputs += weights[node]*data_input[node]
            else:
                #Hidden - Output Layer use gaussian to select multiple weight
                #checking if gaussian elimination is created
                #testing data in tenfold
                if(self.created_gaussian_checking):
                    gaussian_answer = []
                    for class_output in range(self.number_node_output):
                        mean, std = self.class_condition[node][class_output]['mean'], self.class_condition[node][class_output]['std']
                        gaussian_answer.append(self.gaussian_function(mean, std, data_input[node]))

                    weight_used = gaussian_answer.index(max(gaussian_answer))
                    network_inputs += weights[node][weight_used]*data_input[node]
                #training data in tenfold
                else:
                    network_inputs += weights[node][data_class]*data_input[node]
        return network_inputs
# %% sigmoid function
    def sigmoid(self, network_input):
        return (1.0)/(1.0+math.exp(-network_input))
# %% forward propagation function ( forward data from input to output layer )
    def forward_propagation(self, data_row, data_class):
        self.inputs_data = data_row
        self.datarow = data_row
        for layer in range(len(self.networks)):
            # This variable is collect data each row after compute with weight
            data_after_compute = []
            for neuron in range(len(self.networks[layer])):
                network_inputs = self.compute_network_input(self.networks[layer][neuron]['weights'], self.inputs_data, data_class, layer)

                #update output data from node in network
                self.networks[layer][neuron]['output'] = self.sigmoid(network_inputs)

                #Calculate static of output in hidden layer
                if self.end_of_tenfold and layer == 0:
                    output = self.networks[layer][neuron]['output']
                    self.hidden_layer_output[data_class][neuron]['output'].append(output)
                #This is data after compute each layer (use this data to calculate in next layer)
                data_after_compute.append(self.networks[layer][neuron]['output'])
            self.inputs_data = data_after_compute


# %% transfer derivative function (use in backpropagation scope)
    def transfer_derivative(self, output):
        return output * (1.0 - output)

# %% backpropagation scope use this scope to re-value all weight
    def backpropagation(self, expected, data_class):
        #back propagation is begin at output layer
        for backward_layer in reversed(range(len(self.networks))):
            layer = self.networks[backward_layer]
            errors = []
            if backward_layer != len(self.networks) - 1:
                for i_neuron in range((len(layer))):
                    error = 0.0
                    for j_neuron in self.networks[backward_layer + 1]:
                        # if(type(j_neuron['weights'][i_neuron]) == float or type(j_neuron['weights'][i_neuron]) == np.float64):
                        #     print("eiei\n\n\n")
                        #     error += j_neuron['weights'][i_neuron] * j_neuron['errors']
                        # else:
                        error += j_neuron['weights'][i_neuron][data_class] * j_neuron['errors']
                    errors.append(error)
            else: # output layer
                for output_neuron_index in range(len(layer)):
                    neuron = layer[output_neuron_index]
                    errors.append(expected[output_neuron_index] - neuron['output'])
            for neuron_index in range(len(layer)):
                neuron = layer[neuron_index]
                neuron['errors'] = errors[neuron_index] * self.transfer_derivative(neuron['output'])

# %% update weight in model scope
    def update_weights(self, learning_rate, data_class):
        for neuron_index in range(len(self.networks)):
            inputs = self.datarow[:-1] # data each row
            if neuron_index != 0: # check layer is not input layer
                inputs = [neuron['output'] for neuron in self.networks[neuron_index - 1]]
            for neuron in self.networks[neuron_index]:
                for connection_index in range(len(inputs)):
                    # each iteration is each connection in i to j layer
                    if type(neuron['weights'][connection_index]) == float or type(neuron['weights'][connection_index]) == np.float64: # check present layer is a hidden layer?
                        neuron['weights'][connection_index] += learning_rate * neuron['errors'] * inputs[connection_index]
                        neuron['weights'][-1] += learning_rate * neuron['errors']
                    else: # input layer
                        neuron['weights'][connection_index][data_class] += learning_rate * neuron['errors'] * inputs[connection_index]
                        neuron['weights'][-1][data_class] += learning_rate * neuron['errors']


    def training(self, dataSet, learning_rate, number_of_iteration, number_of_outputs):
        for iterate in range(number_of_iteration+1): # +1 is 151 round final round is collect statistics of output in hidden layer
            summation_of_error = 0 # this value collect sum of error in each iteration
            if iterate == number_of_iteration:
                self.end_of_tenfold = True
            for row in dataSet:

                dataClass = int(row[-1])
                self.forward_propagation(row, dataClass)
                if iterate != number_of_iteration: # checking is iteration not final round (151)
                    expected = [0 for i in range(number_of_outputs)]
                    expected[dataClass] = 1

                    summation_of_error += sum([(expected[i] - self.inputs_data[i])**2 for i in range(len(expected))])

                    self.backpropagation(expected, dataClass)
                    self.update_weights(learning_rate, dataClass)
            if iterate != number_of_iteration:
                print('iteration=%d   learning_rate=%.4f   rmse=%.4f' % (iterate, learning_rate, math.sqrt(summation_of_error)))

    def create_gaussian_elimination(self):
        file = open('hidden_output.txt', 'a')
        file.write(str(self.hidden_layer_output))
        file.write('\n')
        file.close()
        for i in range(self.number_node_output):
            for j in range(self.number_node_hidden):
                print(i, j)
                mean = sum(self.hidden_layer_output[i][j]['output'])/len(self.hidden_layer_output[i][j]['output'])
                std = statistics.stdev(self.hidden_layer_output[i][j]['output'], xbar=mean)

                self.class_condition[j][i] = {'mean':mean, 'std':std}
                print(self.class_condition)

        self.created_gaussian_checking = True
        return self.class_condition


    def predict(self, row, dataClass):
        self.forward_propagation(row, dataClass)
        return self.inputs_data

    def gaussian_function(self, means, std, output):
        return (1/(std*math.sqrt(2*math.pi)))*math.exp((-1/2)*((output-means)/std)**2)




