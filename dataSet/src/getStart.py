import neuralModel
import numpy as np
def main():
    model = neuralModel.NeuralNetwork(4, 3, 2)
    # model.check()
    output = model.feedForward(np.array([15.26, 14.84, 0.871, 5.763]))
    print(output)
main()