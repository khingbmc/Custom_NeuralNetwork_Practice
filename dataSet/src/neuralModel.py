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

    def feedForward(self, inputArray):
        self.target = inputArray[0]
        self.inputLayer = inputArray[1:]
        _hiddenLayer = np.dot(self.weights_ij, self.inputLayer) #output of input layer i
        self.hiddenLayer = self.sigmoid(_hiddenLayer)
        print("---------- INPUT -> HIDDEN (ij) ----------")
        print("SIZE")
        print(self.weights_ij.size)
        print("Array of Weight IJ")
        print(self.weights_ij)
        print("DOT")
        print(self.inputLayer)
        print("EQUALS (Input of HiddenLayer)")
        print(self.hiddenLayer)
        
        _outputLayer = np.dot(self.weights_jk, self.hiddenLayer)
        self.outputLayer = self.sigmoid(_outputLayer)
        print("---------- HIDDEN -> OUTPUT (jk) ----------")
        print("Array of Weight JK")
        print(self.weights_jk)
        print("DOT")
        print(self.hiddenLayer)
        print("EQUALS (Input of OutputLayer)")
        print(self.outputLayer)

    def sigmoid(self, array):
        outArray = np.zeros(array.size)
        for i in range(array.size):
            outArray[i] = (1/(1+np.exp(-array[i])))
        return outArray

    