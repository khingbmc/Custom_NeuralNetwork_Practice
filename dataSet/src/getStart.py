import neuralModel as Ne
def main():
    NN = Ne.NeuralNetwork(3, 3, 3)
    NN.feedForward([4, 2, 5])