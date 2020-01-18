import pandas as pd
from random import shuffle
from test_iris_neural import TestNeural
from sklearn.utils import shuffle
from random import random
from random import randint
import math

def main(dataname, num_inputs, num_hiddens, learning_rate):
    """eiei"""
    # normalize data
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    # read data .csv
    data = pd.read_csv("../"+dataname+".csv")


    max_val, min_val = [0 for i in range(num_inputs)], [0 for i in range(num_inputs)]

    for i in range(len(data.keys())):
        if(data.keys()[i] != 'class'):
            max_val[i] = max(data[data.keys()[i]])
            min_val[i] = min(data[data.keys()[i]])

    # shuffle row in dataframe
    index = data.index
    data = shuffle(data)
    data.index = index

    ID = data.index.values
    data_key = []
    for j in ID:
        format_data = []
        for i in data:
            format_data.append(data[i][j])
        data_key.append(format_data)

    # print(data_key)


    num_outputs = len(set(data['class']))

    class_set = list(set(data['class']))

    for i in range(len(data_key)):
        class_val = data_key[i][num_inputs]
        del data_key[i][num_inputs]
        data_key[i].append(class_set.index(class_val))
    
    # normalize data
    for i in range(len(data_key)):
        for j in range(len(max_val)):
            
            data_key[i][j] = normalized(data_key[i][j], max_val[j], min_val[j])

    input_data = [[] for _ in range(num_outputs)]
    for i in data_key:
        input_data[i[-1]].append(i)
   
   
    num_data = 0
    num_each_group = []
    for i in range(len(input_data)):
       num_data += len(input_data[i])
       num_each_group.append(len(input_data[i]))

    accuracy = []

    weight1 = [{'weights':[0.5 for i in range(num_inputs)]} for i in range(num_hiddens)]
    weight2 = [{'weights':[0.5 for i in range(num_hiddens)]} for i in range(num_outputs)]
    networks = []
    for i in range(num_outputs):
        networks.append(TestNeural(num_inputs, num_hiddens, num_outputs, i))
        networks[i].training(input_data[i], learning_rate, 500, num_outputs)
        print(networks[i].network)
    # networks = TestNeural(num_inputs, num_hiddens, num_outputs, weight1, weight2)
    # num_training_each_group = [50, 50, 50]
    # networks.training(input_data, learning_rate, 500, num_outputs, 0, num_training_each_group, num_each_group)
    
main(input("Type File csv name: "), int(input("Type number of attribute in data: ")), int(input("Type node number of hidden layers: ")), float(input("Type learning rate: ")))
