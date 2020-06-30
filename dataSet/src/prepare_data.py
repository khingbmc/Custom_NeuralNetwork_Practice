#Library
import pandas as pd
import math

from sklearn.utils import shuffle
from random import random
from random import randint

# Neural Model Class
from neuralModel import TestNeural




def prepareData(num_inputs, number_node_hidden, learning_rate, number_of_iterate):
    normalized = lambda x, maxv, minv : (x-minv*0.95)/(maxv*1.05-minv*0.95)
    train_path = input("Type path of train file: ")
    test_path = input("Type path of test file: ")

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

    if(bool(input("Do you want to nomalized this data?? (True/False)"))):
        for i in range(len(list_train)):
            for j in range(len(max_val)):

                list_train[i][j] = normalized(list_train[i][j], max_val[j], min_val[j])

        for i in range(len(list_test)):
            for j in range(len(max_val)):

                list_test[i][j] = normalized(list_test[i][j], max_val[j], min_val[j])


# %% Init Weight in neural network model
    weight1 = [{'weights': [0.685884522959965, 0.5004877793869557, 0.5493324818169311, 0.9168289471677233, 0.6165113649173493, 0.5097458565385011, 0.6502103693513898, 0.6716081148430538]}, {'weights': [0.35772423166862166, 0.06449875424855767, 0.6656938468137663, 0.20712385970342873, 0.4694803495977382, 0.07810377437953031, 0.844393497983296, 0.5756023916206351]}, {'weights': [0.39803517587118076, 0.9931595637248957, 0.9387952120377739, 0.39406911728567706, 0.48178033434557443, 0.6971504731148207, 0.540508884860321, 0.6686160121942201]}, {'weights': [0.21661966958309087, 0.6271763636674476, 0.41002171709352664, 0.8848279335850445, 0.23106178866278027, 0.8679941403048321, 0.963466898287706, 0.5561158051665889]}, {'weights': [0.042965552339581325, 0.0031584437287864864, 0.21373692025555935, 0.4736289861491132, 0.6516396790117789, 0.40907481068883955, 0.7740204243228116, 0.416069772662349]}, {'weights': [0.9971937308431256, 0.0835754196523033, 0.2824341285692601, 0.7131135043974371, 0.8406367789271569, 0.5413866479187055, 0.5585086524639331, 0.2263309787280724]}, {'weights': [0.0588878621967569, 0.820075999878592, 0.5740789093488201, 0.5970130202116981, 0.14318210003536747, 0.22165705381837197, 0.9333723678692607, 0.6054940183269326]}, {'weights': [0.34077969147051135, 0.9738640965665113, 0.8882586411876305, 0.7243293114804243, 0.6374844540723572, 0.3593374323934294, 0.9663457980592923, 0.5844535709498352]}, {'weights': [0.6920824742426916, 0.8593283560557613, 0.7386164957078946, 0.20012668202051265, 0.7468494075567791, 0.8968340686183378, 0.09620146015518427, 0.2946342383320547]}, {'weights': [0.256099818912192, 0.4262955312925317, 0.31255306772315283, 0.8964475855306947, 0.47646807601763264, 0.20301592710575422, 0.8690506360038713, 0.1001146889742075]}, {'weights': [0.839475398784809, 0.8352886136889345, 0.50949239261281, 0.22854340712324162, 0.5417137974152579, 0.3442864730521903, 0.862714588223163, 0.7611413252382551]}]
    # weight in layer 2 is multi-weight / connection
    weight2 = [{'weights': [0.8763699815432019, 0.3837972973359153,  0.8726021409043736,  0.4552517447555178,  0.2360630198657533, 0.43490978808501424, 0.20521860819599635,  0.6779091288172228,  0.9078148977952254,  0.9736629086385454, 0.09476612618692215]}, {'weights': [ 0.5187683284411252, 0.10730655081054341, 0.615686878774022,  0.5519456807817172,  0.5227840321658236, 0.02308201919404973, 0.667329114197327,  0.8796669000167747, 0.39047318908057504, 0.44460112817913233,  0.7763909650135756]}]
# %% Model part
    network = TestNeural(num_inputs, number_node_hidden, num_outputs, weight1, weight2)
    file_object = open('../present/Diabetes/normal/initial_weight/Diabetes_init_weight10'+'.txt', 'a')

    # Append 'hello' at the end of file
    file_object.write(str(network.network)+"\n\n")

    # Close the file
    file_object.close()

    network.training(list_train,learning_rate, number_of_iterate, num_outputs, list_test)


    accuracy = 0
    for row in list_test:
        prediction = network.predict(row)
        print("Apply Model")
        print("Expect=%d  Output=%d" % (row[-1], prediction.index(max(prediction))))
        file_object = open('../present/Diabetes/normal/result/Diabetes_prediction10'+'.txt', 'a')

        # Append 'hello' at the end of file
        file_object.write("Expect=%d  Output=%d\n" % (row[-1], prediction.index(max(prediction))))
        file_object.write(str(row)+"\n\n")
        # Close the file
        file_object.close()

        if row[-1] == prediction.index(max(prediction)):
            accuracy += 1
        sum_accuracy = accuracy/len(list_test)*100
    print("Mean Accuracy = ", sum_accuracy)
    file_object = open('../present/Diabetes/normal/result/Diabetes_prediction10'+'.txt', 'a')

     # Append 'hello' at the end of file
    file_object.write("Accuracy : "+ str(sum_accuracy))

    # Close the file
    file_object.close()

    file_object = open('../present/Diabetes/normal/network/Diabetes_network10'+'.txt', 'a')

     # Append 'hello' at the end of file
    file_object.write(str(network.network))

    # Close the file
    file_object.close()

# Define function normalization

prepareData(int(input('Please fill number of node input: ')), int(input('Please fill number of node hidden: ')),float(input('Please fill number of learning rate: ')), int(input('Please fill round of iteration in training phase: ')))