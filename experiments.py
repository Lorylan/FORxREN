from  FORxREN import FORxREN
import configuration as conf
import copy
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.datasets import load_wine, load_iris, load_breast_cancer

# Pasar el y con np_utils.to_categorical(copy.deepcopy(Y_train),num_classes=classes)

def build_model(hidden_layer_sizes,input_dim, classes, X , Y, persentage_test ,loss ):

    end_idx=len(X)*1-persentage_test
    
    X_train = X[:end_idx]
    Y_train = Y[:end_idx]

    model = Sequential()

    model.add(Dense(hidden_layer_sizes[0],input_dim=input_dim,activation='relu',name="First_layer"))

    #Arranca desde la segunda, porque la primera ya la agregue
    for hidden_layer_size in hidden_layer_sizes[1:-1]:
        model.add(Dense(hidden_layer_size, activation='relu'),name=f'Hidden_layer_{hidden_layer_sizes.index(hidden_layer_size)}')
    
    model.add(Dense(hidden_layer_sizes[-1],activation='softmax'), name="Output_layer" )

    model.compile(loss=loss,optimizer='adam',metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32,epochs=200,verbose=1)
    
    return model

if (conf.DATASET == "iris"):
    iris= load_iris()
    CANT_CLASES = len(iris.target_names)
    INPUT_DIM = len(iris.feature_names)
    ATTRIBUTES = iris.feature_names
    LOSS_FUNCTION = ''
    X, Y = shuffle(iris.data , iris.target)
    
    HIDDEN_LAYER_SIZES= [5,CANT_CLASES]
    
    
elif (conf.DATASET == "wbc"):
    wbc = load_breast_cancer()
    CANT_CLASES = len(wbc.target_names)
    INPUT_DIM = len(wbc.feature_names)
    ATTRIBUTES = wbc.feature_names
    LOSS_FUNCTION = ''
    X, Y = shuffle(wbc.data , wbc.target)
    
    HIDDEN_LAYER_SIZES = [10,CANT_CLASES]
    
else:
    # conf.DATASET == "wine"
    wine= load_wine()
    CANT_CLASES = len(wine.target_names)
    INPUT_DIM = len(wine.feature_names)
    ATTRIBUTES = wine.feature_names
    LOSS_FUNCTION = ''
    X, Y = shuffle(wine.data , wine.target)
    
    HIDDEN_LAYER_SIZES = [10, CANT_CLASES]


KERAS_MODEL = build_model(HIDDEN_LAYER_SIZES,INPUT_DIM,CANT_CLASES,X,Y, LOSS_FUNCTION)

forxren = FORxREN().extract_rules(KERAS_MODEL, X, Y, INPUT_DIM, conf.FIRST_LAYER_SIZE, conf.EXECUTION_MODE, 
                                 conf.TEST_PERCENT, conf.MAX_FIDELITY_LOSS, ATTRIBUTES)
