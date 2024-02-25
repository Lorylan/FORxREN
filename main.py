from  FORxREN import FORxREN
import configuration as conf
import copy
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras import utils
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
import tensorflow as tf
from tensorflow.keras.models import load_model

def build_model(hidden_layer_sizes ,input_dim ,X ,Y ,persentage_test ,loss,classes):

    end_idx=round(len(X)*1-persentage_test)
    
    X_train = X[:end_idx]
    Y_train = Y[:end_idx]
    y_train = utils.to_categorical(copy.deepcopy(Y_train),num_classes=classes)

    model = Sequential()

    model.add(Dense(hidden_layer_sizes[0],input_dim=input_dim,activation='relu',name="First_layer"))

    #Arranca desde la segunda, porque la primera ya la agregue
    idx_layer = 1
    for hidden_layer_size in hidden_layer_sizes[1:-1]:
        model.add(Dense(hidden_layer_size, activation='relu',name=f'Hidden_layer_{idx_layer}'))
        idx_layer+=1
    
    model.add(Dense(hidden_layer_sizes[-1],activation='softmax', name="Output_layer"))

    model.compile(loss=loss,optimizer='adam',metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=0)
    
    return model

if (conf.DATASET == "iris"):
    iris= load_iris()
    CANT_CLASES = len(iris.target_names)
    CLASS_NAMES = iris.target_names
    INPUT_DIM = len(iris.feature_names)
    ATTRIBUTES = iris.feature_names
    LOSS_FUNCTION = 'categorical_crossentropy'
    X, Y = shuffle(iris.data , iris.target)
  
elif (conf.DATASET == "wbc"):
    wbc = load_breast_cancer()
    CANT_CLASES = len(wbc.target_names)
    CLASS_NAMES = wbc.target_names
    INPUT_DIM = len(wbc.feature_names)
    ATTRIBUTES = wbc.feature_names
    LOSS_FUNCTION = 'binary_crossentropy'
    X, Y = shuffle(wbc.data , wbc.target)
       
else:
    # conf.DATASET == "wine"
    wine= load_wine()
    CANT_CLASES = len(wine.target_names)
    CLASS_NAMES = wine.target_names
    INPUT_DIM = len(wine.feature_names)
    ATTRIBUTES = wine.feature_names
    LOSS_FUNCTION = 'categorical_crossentropy'
    X, Y = shuffle(wine.data , wine.target)
    

if (conf.CREATE_NN):
    HIDDEN_LAYER_SIZES = [INPUT_DIM//2, INPUT_DIM*2, INPUT_DIM,CANT_CLASES]
    KERAS_MODEL = build_model(HIDDEN_LAYER_SIZES ,INPUT_DIM ,X ,Y ,conf.TEST_PERCENT,LOSS_FUNCTION, CANT_CLASES)
    FIRST_LAYER_SIZE = HIDDEN_LAYER_SIZES[0]

else:
    KERAS_MODEL = load_model('{}_model'.format(conf.DATASET))
    FIRST_LAYER_SIZE = KERAS_MODEL.get_layer("First_layer").output_shape[1]

keras_model = FORxREN().extract_rules(KERAS_MODEL ,X ,Y ,INPUT_DIM ,FIRST_LAYER_SIZE , conf.EXECUTION_MODE, conf.TEST_PERCENT, conf.MAX_FIDELITY_LOSS, ATTRIBUTES, CANT_CLASES, CLASS_NAMES, conf.SHOW_STEPS)

if (conf.SAVE_NN):
    keras_model.save('{}_model'.format(conf.DATASET))
    