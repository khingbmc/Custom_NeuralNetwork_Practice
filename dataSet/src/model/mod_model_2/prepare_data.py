#Library
import pandas as pd
import math
import numpy as np

from sklearn.utils import shuffle
from random import random
from random import randint

# Neural Model Class
from Mod_Neural_Model import NeuralModel




def prepareData(num_inputs, number_node_hidden, learning_rate, number_of_iterate):
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    train_path = input("Type path of train file: ")
    test_path = input("Type path of test file: ")

    training_data = pd.read_csv("../../../dataFile/"+train_path+".csv")
    testing_data = pd.read_csv("../../../dataFile/"+test_path+".csv")


    max_val, min_val = [0 for i in range(num_inputs)], [0 for i in range(num_inputs)]

    for i in range(len(training_data.keys())):
        if(training_data.keys()[i] != 'class'):
            max_val[i] = max(training_data[training_data.keys()[i]])
            min_val[i] = min(training_data[training_data.keys()[i]])

    # count each data class
    list_data_class = [x for x in set(training_data['class'])]
    number_of_class = len(list_data_class)
    list_split_data = [[] for _ in range(number_of_class)]
    # split data in each class list
    for i in list_data_class:
        list_split_data[i] = training_data[training_data['class'] == i]

    # calculate learning rate in each class
    list_learning_rate = [0 for _ in range(number_of_class)]

    for class_index in range(len(list_split_data)):
        list_learning_rate[class_index] = (1-len(list_split_data[class_index])/len(training_data))*learning_rate

    # shuffle row in dataframe
    index = training_data.index
    training_data = shuffle(training_data)
    training_data.index = index

    index = testing_data.index
    testing_data = shuffle(testing_data)
    testing_data.index = index

    ID = training_data.index.values
    list_train = []

    for j in ID:
        format_data = []
        for i in training_data:
            format_data.append(training_data[i][j])
        list_train.append(format_data)

    ID2 = testing_data.index.values
    list_test = []
    for j in ID2:
        format_data = []
        for i in testing_data:
            format_data.append(testing_data[i][j])
        list_test.append(format_data)


    # print(data_key)

    num_outputs = len(set(training_data['class']))

    if(input("Do you want to nomalized this data?? (Yes or No) : ") == 'Yes'):
        print("Eiei")
        for i in range(len(list_train)):
            for j in range(len(max_val)):

                list_train[i][j] = normalized(list_train[i][j], max_val[j], min_val[j])

        for i in range(len(list_test)):
            for j in range(len(max_val)):

                list_test[i][j] = normalized(list_test[i][j], max_val[j], min_val[j])


# %% Init Weight in neural network model
    weight1 = []
    for i in range(number_node_hidden):
        weight = {'weights':[]}
        # nm_weight = {'weights':[]}
        for j in range(num_inputs):
            weight_random = random()
            sub_weight = {}
            for k in np.arange(0.0, 1.1, 0.1):
                sub_weight[round(k, 1)] = weight_random
            weight['weights'].append(sub_weight)
        weight1.append(weight)
    # weight in layer 2 is multi-weight / connection
    # weight2 = [{'weights':[{'base': random()} for i in range(number_node_hidden)]} for i in range(num_outputs)]
    weight2 = [{'weights':[random() for i in range(number_node_hidden)]} for i in range(num_outputs)]

# %% Model part
    network = NeuralModel(num_inputs, number_node_hidden, num_outputs, weight1, weight2)

    network.training(list_train,learning_rate , number_of_iterate, num_outputs, list_test)


    accuracy = 0
    for row in list_test:
        prediction = network.predict(row)
        print("Apply Model")
        print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))
        # file_object = open('../../newWay/Diabetes/results/Diabetes_prediction3'+'.txt', 'a')
        # file_object.write("Expect=%d  Output=%d\n" % (row[-1], prediction.index(max(prediction))))
        # file_object.write(str(row)+"\n\n")
        # file_object.close()

        if row[-1] == prediction.index(max(prediction)):
            accuracy += 1
        sum_accuracy = accuracy/len(list_test)*100
    print("Mean Accuracy = ", sum_accuracy)
    # file_object = open('../../newWay/Diabetes/results/Diabetes_prediction3'+'.txt', 'a')
    # file_object.write("Accuracy : "+ str(sum_accuracy))
    # file_object.close()


prepareData(int(input('Please fill number of node input: ')), int(input('Please fill number of node hidden: ')),float(input('Please fill number of learning rate: ')), int(input('Please fill round of iteration in training phase: ')))