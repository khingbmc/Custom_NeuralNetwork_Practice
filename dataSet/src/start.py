import pandas as pd
from random import shuffle
from neuralModel import NeuralNetwork
from sklearn.utils import shuffle
from random import random
from random import randint
import math

def main(dataname, num_inputs, num_hiddens, learning_rate):

    # normalize data
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    # read data .csv
    data = pd.read_csv("../"+dataname+".csv")


    max_val, min_val = [0 for i in range(num_inputs)], [0 for i in range(num_inputs)]

    # for i in data:
    #     if(i != 'class'):
    #         if(num == 30):
    #             break
    #         max_val[num] = max(data[i])
    #         min_val[num] = min(data[i])
    #         num += 1

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
    # print(num_inputs, num_outputs)
    # print(list(class_set))


    
    for i in range(len(data_key)):
        class_val = data_key[i][num_inputs]
        del data_key[i][num_inputs]
        data_key[i].append(class_set.index(class_val))
    # print(data_key[0])

    #tenflow 35 and 21 and last iteration is 42 and 23
    
    # normalize data
    for i in range(len(data_key)):
        for j in range(len(max_val)):
            
            data_key[i][j] = normalized(data_key[i][j], max_val[j], min_val[j])
    # print(data_key[0])


    input_data = [[] for _ in range(num_outputs)]
    for i in data_key:
        input_data[i[-1]].append(i)
    print(input_data)
   
    num_data = 0
    num_each_group = []
    for i in range(len(input_data)):
       num_data += len(input_data[i])
       num_each_group.append(len(input_data[i]))
   


    networks = []
    accuracy = []

    weight1 = [{'weights':[random() for i in range(num_inputs)]} for i in range(num_hiddens)]
    weight2 = [{'weights':[random() for i in range(num_hiddens)]} for i in range(num_outputs)]

    for i in range(10):
        networks.append(NeuralNetwork(num_inputs, num_hiddens, num_outputs, weight1, weight2))

    for i in range(10):
        # if i != 9:
        #     num_training_tenflow = num_data//10
        
        num_training_each_group = [len(input_data[x])//10 for x in range(len(input_data))]
        if i == 9:
            num_training_each_group = [len(input_data[x])-len(input_data[x])//10*9 for x in range(len(input_data))]
        
            # num_training_each_group = [len(input_data[x])-len(input_data[x])//10*9 for x in range(len(input_data))]
        #     print(num_training_each_group)
        # else :
        #     num_training_tenflow = num_data-num_data//10*9
        #     num_training_each_group = [len(input_data[x])-len(input_data[x])//10*9 for x in range(len(input_data))]
        
        networks[i].training(input_data, learning_rate, 500, num_outputs, i, num_training_each_group, num_each_group)
        
        num = 0
        
        for row in networks[i].testing:
            
            print("this is test: "), len(networks[i].testing)
            
            prediction = networks[i].predict(row)
            if row[-1] == prediction:
                num += 1
            print("Expect=%d  Output=%d" % (row[-1], prediction))
      
        accuracy.append(num/(sum(num_training_each_group))*100)
        print("this is training")
        print(num_training_each_group)
        

    print(accuracy)

    for i in range(10):
        print("Model ", i)
        # print(networks[i].network, end='\n\n')
        print("Accuracy : ", accuracy[i])

    print("Mean Accuracy: " ,sum(accuracy)/10)

main(input("Type File csv name: "), int(input("Type number of attribute in data: ")), int(input("Type node number of hidden layers: ")), float(input("Type learning rate: ")))