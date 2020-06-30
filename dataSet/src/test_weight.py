import pandas as pd
from random import shuffle
from test_iris_neural import TestNeural
from sklearn.utils import shuffle
from random import random
from random import randint
import math
import statistics
from neuralModel import NormalNeural
import csv

def main(dataname, num_inputs, num_hiddens, learning_rate, round):
    """eiei"""
    # normalize data
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    # read data .csv
    data = pd.read_csv("../dataFile/"+dataname+".csv")


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


    # input_data = [[] for _ in range(num_outputs)]
    input_data = []
    for i in data_key:
        input_data.append(i)
   
   
    num_data = 0
    num_each_group = [0 for _ in range(num_outputs)]
    index_each_group = [[] for _ in range(num_outputs)]
    num_test_each_rounds = [0 for _ in range(num_outputs)]
    

    for i in range(len(input_data)):
       num_data += len(input_data[i])
       num_each_group[input_data[i][-1]] += 1
       index_each_group[input_data[i][-1]].append(i)



#%% Random index number of testing (20 %)

#%% tenfold data testing
    # print(num_each_group)
    for i in range(len(num_each_group)):
        num_test_each_rounds[i] = num_each_group[i]//10
        index_each_group[i] = shuffle(index_each_group[i])
    
    file = open("test_iris/tenfold_data"+".txt", "w")
    file.write(str(index_each_group))
    file.close()


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

    # file = open("test_iris/nm_weight"+".txt", "w")
    # file.write(str(nm_weight2))
    # file.close()
        
    #data Test
    file_object = open('test_iris/test/testing'+str(round)+'.txt', 'w')
    file_object.write("")
    file_object.close()
    #data Train
    file_object = open('test_iris/train/training'+str(round)+'.txt', 'w')
    file_object.write("")
    file_object.close()
# %% init Network and Train
    

    

    
    # for j in range(15):
    #     testing.append(input_data[randint(0, 150)])
    networks = []
    # nm_networks = []
    # for i in range(num_outputs):
    num_testing = []

    num_data = 0
    num_each_group = [0 for _ in range(num_outputs)]
    index_each_group = [[] for _ in range(num_outputs)]
    
    

    for i in range(len(input_data)):
       num_data += len(input_data[i])
       num_each_group[input_data[i][-1]] += 1
       index_each_group[input_data[i][-1]].append(i)


# %% select train and test data
    for i in range(10):
        print("Tenfold Round "+str(i), end='\n\n')

        testing = []
        training_index = [int(x) for x in range(len(input_data))]
        

        for j in range(num_outputs):
            if i != 9:
                # print(len(input_data))
                # print(len(index_each_group[j]))
                for k in range(num_test_each_rounds[j]*i, num_test_each_rounds[j]*(i+1)):
                    testing.append(input_data[index_each_group[j][k]])
                    training_index.remove(index_each_group[j][k])
            else:
                
                for k in range(num_test_each_rounds[j]*i, len(index_each_group[j])):
                    testing.append(input_data[index_each_group[j][k]])
                    training_index.remove(index_each_group[j][k])
            
        training_data = list()
        for c in training_index:
            training_data.append(input_data[c])
        print(training_data)
        print(testing)

        num_testing.append(len(testing))
        
        

        
        # print(len(training_data))
     

        file_object = open('test_iris/test/testing'+str(i)+'.txt', 'a')
 
        # Append 'hello' at the end of file
        file_object.write(str(testing)+"\n\n")
        
        # Close the file
        file_object.close()

        file_object = open('test_iris/train/training'+str(i)+'.txt', 'a')
 
        # Append 'hello' at the end of file
        file_object.write(str(training_data)+"\n\n")
        
        # Close the file
        file_object.close()

# %%

        

        networks.append(TestNeural(num_inputs, num_hiddens, num_outputs, weight1, weight2))
        # nm_networks.append(NormalNeural(num_inputs, num_hiddens, num_outputs, weight1, nm_weight2))
        
        networks[i].training(training_data, learning_rate, 500, num_outputs, testing)
        # nm_networks[i].training(training_data, learning_rate, 500, num_outputs, testing)

        
        c = networks[i].create_condition()
    
    sum_accuracy = []
    # nm_sum_accuracy = []
    for j in range(len(networks)): 
        accuracy = 0
        # nm_accuracy = 0
        print("Predict Set "+str(j))
        for row in networks[j].testing:
                
                prediction = networks[j].predict(row)
                # nm_prediction = nm_networks[j].predict(row)
                
                print("Apply Model")
                print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))

                file = open('test_iris/predict.txt', 'a')
                file.write(str(row))
                file.write('\n')
                file.write("Expect=%d  Output=%d\n\n" % (row[-1], prediction.index(max(prediction))))
                file.close()


                # print("Normal Model")
                # print("Expect=%d  Output=%d" % (row[-1], nm_prediction.index(max(nm_prediction))), end='\n\n')
                # file = open('test_iris/nm_predict.txt', 'a')
                # file.write(str(row))
                # file.write('\n')
                # file.write("Expect=%d  Output=%d\n\n" % (row[-1], nm_prediction.index(max(nm_prediction))))
                # file.close()

                if row[-1] == prediction.index(max(prediction)):
                    accuracy += 1
                # if row[-1] == nm_prediction.index(max(nm_prediction)):
                #     nm_accuracy += 1
        
        sum_accuracy.append(accuracy/num_testing[j]*100)
        # nm_sum_accuracy.append(nm_accuracy/num_testing[j]*100)
    for i in range(len(sum_accuracy)):
        print("Accuracy Set "+str(i)+" : "+str(sum_accuracy[i]))

    # for i in range(len(sum_accuracy)):
    #     print("Normal Accuracy Set "+str(i)+" : "+str(nm_sum_accuracy[i]))

    print("Mean Accuracy : "+str(statistics.mean(sum_accuracy)) )
    # print("Normal Mean Accuracy : "+str(statistics.mean(nm_sum_accuracy)) )
    
    file_object = open('test_iris/accuracy.txt', 'a')
 
    # Append 'hello' at the end of file
    file_object.write(str(sum_accuracy)+"\n")
    file_object.write("Mean Accuracy : "+str(statistics.mean(sum_accuracy))+"\n\n")
    
    # Close the file
    file_object.close()

    # file_object = open('test_iris/nm_accuracy.txt', 'a')
 
    # Append 'hello' at the end of file
    # file_object.write(str(nm_sum_accuracy)+"\n")
    # file_object.write("Mean Accuracy : "+str(statistics.mean(nm_sum_accuracy))+"\n\n")
    
    # # Close the file
    # file_object.close()
            
    
file_name = input("Type File csv name: ")
num_attribute = int(input("Type number of attribute in data: "))
num_hidden_layer = int(input("Type node number of hidden layers: "))
learn_rate = float(input("Type learning rate: "))

# for i in range(10):
main(file_name, num_attribute, num_hidden_layer, learn_rate, 0)
