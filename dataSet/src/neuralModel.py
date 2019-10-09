import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, inputs, hiddens, outputs):
        self.n_input = inputs
        self.n_hidden = hiddens
        self.n_output = outputs
        self.inputLayer = np.zeros(inputs) #init neurons in inputLayer
        self.hiddenLayer = np.zeros(hiddens) #init neurons in hiddenLayer
        self.outputLayer = np.zeros(outputs) #init neurons in outputLayer
        self.target = ''
        self.errors = np.array([])
        self.weights_ij = np.random.rand(hiddens, inputs)
        self.weights_jk = np.random.rand(outputs, hiddens)
        self.weights = [self.weights_ij, self.weights_jk]

    def feedForward(self, row):
        inputs = row
        new_inputs = []
        for layer in self.weights:
            activation = []
            for neuron in layer:
                print("=================")
                print(neuron)
                activation.append(self.activate(neuron, inputs))
            new_inputs = self.sigmoid(activation)
            print(">>>>>>>>>>>>>>")
            print(new_inputs)
            print(activation)
            inputs = new_inputs
        return inputs

    def activate(self, weight, inputs):
        activation = inputs * weight
        return sum(activation)


    def check(self):
        print(self.weights_ij)
        print(self.weights_jk)
        return 0

    def sigmoid(self, array):
        outArray = np.zeros(len(array))
        for i in range(len(array)):
            outArray[i] = (1/(1+np.exp(-array[i])))
        return outArray

    