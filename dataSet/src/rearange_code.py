#%% import zone
import pandas as pd

from sklearn.utils import shuffle

from random import random
from random import randint
from random import shuffle

import math

#%% main function

def main(dataname, number_inputs, number_hiddens, leaning_rate):
    """ Main Function is reading Data and 
        Put Data to Neural Model in Neural 
        Network Class in another file and 
        print out reusult of prediction"""

    # define normalized function (lambda function)
    normalized = lambda x, maximum, minimum : (x - minimum*0.95)/(maximum*1.05 - minimum*0.95)

    # read data (.csv data)
    dataset = pd.read_csv("../"+dataname+".csv")

    # define maximum and minimum array
    maximum, minimum = [0 for _ in range(number_inputs)], [0 for _ in range(number_inputs)]

    for i in range(len(data.keys())):
        if(data.keys()[i] != 'class'):
            maximum[i] = max(data[data.keys()[i]])
            minimum[i] = min(data[data.keys()[i]])

    # shuffle row in dataframe
    index = dataset.index
    data = shuffle(dataset)
    data.index = index

    ID = data.index.values
    data_key = []
    for j in ID:
        format_data = []
        for i in data:
            format_data.append(data[i][j])
        data_key.append(format_data)

    # assign output
    number_outputs = len(set(data['class']))

    set_of_class = list(set(data['class']))

    
