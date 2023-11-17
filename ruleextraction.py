# Library
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt #Visualization
from keras.utils import np_utils
import tensorflow as tf
import pprint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras #library for neural network
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np # linear algebra
import copy

#Configuration
DATASET = "iris" #Default iris - options: "iris", "wbc"
EXECUTION_MODE = 1 #Default 1 (complete algorithm), options: 1, 2 (No Updation of rules), 3 (Initial rules only, no pruning nor updation)
TRAIN_PERCENT = 0.8 #Default 0.8 - % of data used for training, 0.8 = 80%
TEST_PERCENT = 0.2 #Default 0.2 - $ of data used for testing, 0.2 = 20%



#from FORxREN import FORxREN
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras #library for neural network
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np # linear algebra
from keras.utils import np_utils
import copy

class Iris_extractor(FORxREN):
    """
    A rule extractor for the Iris dataset.

    Inherits from the FORxREN class, which provides the basic funcionality
    for extracting rules from dataset.

    """

    def get_model(self):
        return keras.models.load_model('/content/rule_extraction/models/iris_model')

    def process_data(self):
        data=pd.read_csv(self.path_data)
        data = data.sample(frac=1)

        #Set classes with 0, 1 and 2
        data.loc[data["species"]=="setosa","species"]=0
        data.loc[data["species"]=="versicolor","species"]=1
        data.loc[data["species"]=="virginica","species"]=2

        #Convert data to handle it with the network
        self.X = data.iloc[:,0:4].values
        self.Y = data['species'].values

        return data

#from FORxREN import FORxREN
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras #library for neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import copy

class WBC_extractor(FORxREN):
    """
    A rule extractor for the WBC dataset.

    Inherits from the FORxREN class, which provides the basic funcionality
    for extracting rules from dataset.

    """

    def get_model(self):
        return keras.models.load_model('/content/rule_extraction/models/wbc_model')

    def process_data(self):
        data=pd.read_csv(self.path_data)
        data = data.sample(frac=1)

        #Eliminamos la columna ID innecesaria
        data.drop(["id"], axis=1, inplace=True)

        #Seteamos las clases con 0 y 1
        data.diagnosis = [1 if each == "B" else 0 for each in data.diagnosis]

        #Convertimos los datos para manejarlos con la red
        self.X = data.iloc[:,1:31].values
        self.Y = data.diagnosis.values
        return data

#from Iris_extractor import Iris_extractor
#from WBC_extractor import WBC_extractor
import argparse

def main():

  # Extract and process data for the iris dataset
  if (DATASET == "iris"):
    CANT_CLASSES = 3
    INPUT_DIM = 4
    FIRST_LAYER_SIZE = 5
    PATH_DATA = "/content/rule_extraction/data/iris_data.csv"


    iris = Iris_extractor(CANT_CLASSES,INPUT_DIM,FIRST_LAYER_SIZE, TRAIN_PERCENT, TEST_PERCENT,PATH_DATA,EXECUTION_MODE)
    iris.start()

  # Extract and process data for the WBC dataset
  elif (DATASET == "wbc"):
    CANT_CLASSES = 2
    INPUT_DIM = 30
    FIRST_LAYER_SIZE = 10
    PATH_DATA = "/content/rule_extraction/data/wisc_bc_data.csv"

    wbc = WBC_extractor(CANT_CLASSES,INPUT_DIM,FIRST_LAYER_SIZE, TRAIN_PERCENT, TEST_PERCENT,PATH_DATA,EXECUTION_MODE)
    wbc.start()

main()