import pandas as pd
from random import shuffle
from ModNeuralModel import TestNeural
from sklearn.utils import shuffle
from random import random
from random import randint
import math
import statistics

import csv

def main(train_path, test_path, num_inputs, num_hiddens, learning_rate, round, check_normalized):
    """eiei"""
    # read data .csv
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    training_data = pd.read_csv("../dataFile/"+train_path+".csv")
    testing_data = pd.read_csv("../dataFile/"+test_path+".csv")


    max_val, min_val = [0 for i in range(num_inputs)], [0 for i in range(num_inputs)]

    for i in range(len(training_data.keys())):
        if(training_data.keys()[i] != 'class'):
            max_val[i] = max(training_data[training_data.keys()[i]])
            min_val[i] = min(training_data[training_data.keys()[i]])

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

    if(check_normalized):
        for i in range(len(list_train)):
            for j in range(len(max_val)):

                list_train[i][j] = normalized(list_train[i][j], max_val[j], min_val[j])

        for i in range(len(list_test)):
            for j in range(len(max_val)):

                list_test[i][j] = normalized(list_test[i][j], max_val[j], min_val[j])





# %% init Weight layer 1 and layer w
    weight1 = [{'weights':[random() for i in range(num_inputs)]} for i in range(num_hiddens)]

    weight2 = []
    # nm_weight2 = []
    for i in range(num_outputs):
        weight = {'weights':[]}
        # nm_weight = {'weights':[]}
        for j in range(num_hiddens):
            weight_random = random()
            weight['weights'].append([weight_random for _ in range(num_outputs)])
            # nm_weight['weights'].append(weight_random)
        # nm_weight2.append(nm_weight)
        weight2.append(weight)


# %% Model Process

    networks = TestNeural(num_inputs, num_hiddens, num_outputs, weight1, weight2)

    # file_object = open('../present/Diabetes/modify/initial_weight/Diabetes_init_weight10'+'.txt', 'a')

    # # Append 'hello' at the end of file
    # file_object.write(str(networks.network)+"\n\n")

    # # Close the file
    # file_object.close()

    networks.training(list_train, learning_rate, 500, num_outputs, list_test)
    c = networks.create_condition()

    accuracy = 0
    for row in networks.testing:
        prediction = networks.predict(row)
        if row[-1] == prediction.index(max(prediction)):
            accuracy += 1
        print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))
        # file_object = open('../present/Diabetes/modify/result/Diabetes_prediction10'+'.txt', 'a')

        # # Append 'hello' at the end of file
        # file_object.write("Expect=%d  Output=%d\n" % (row[-1], prediction.index(max(prediction))))
        # file_object.write(str(row)+"\n\n")
        # # Close the file
        # file_object.close()
    accuracy = accuracy/len(list_test)*100
    print("Accuracy : ", accuracy)
    # file_object = open('../present/Diabetes/modify/result/Diabetes_prediction10'+'.txt', 'a')

    #  # Append 'hello' at the end of file
    # file_object.write("Accuracy : "+ str(accuracy))

    # # Close the file
    # file_object.close()

    # file_object = open('../present/Diabetes/modify/network/Diabetes_network10'+'.txt', 'a')

    #  # Append 'hello' at the end of file
    # file_object.write(str(networks.network))

    # # Close the file
    # file_object.close()

train_path = input("Type path of train file: ")
test_path = input("Type path of test file: ")
num_attribute = int(input("Type number of attribute in data: "))
num_hidden_layer = int(input("Type node number of hidden layers: "))
learn_rate = float(input("Type learning rate: "))

# for i in range(10):
main(train_path, test_path, num_attribute, num_hidden_layer, learn_rate, 0, bool(input("Do you want to nomalized this data?? (True/False)")))
