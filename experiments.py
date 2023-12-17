from  FORxREN import FORxREN
import configuration as conf
from sklearn.utils import shuffle
from sklearn.datasets import load_wine, load_iris, load_breast_cancer

if (conf.DATASET == "iris"):
    iris= load_iris()
    CANT_CLASES = len(iris.target_names)
    INPUT_DIM = len(iris.feature_names)
    ATTRIBUTES = iris.feature_names
    X, Y = shuffle(iris.data , iris.target)
    
    FIRST_LAYER_SIZE = 5
    #keras_model
    
elif (conf.DATASET == "wbc"):
    wbc = load_breast_cancer()
    CANT_CLASES = len(wbc.target_names)
    INPUT_DIM = len(wbc.feature_names)
    ATTRIBUTES = wbc.feature_names
    X, Y = shuffle(wbc.data , wbc.target)
    
    FIRST_LAYER_SIZE = 10
    
else:
    wine= load_wine()
    CANT_CLASES = len(wine.target_names)
    INPUT_DIM = len(wine.feature_names)
    ATTRIBUTES = wine.feature_names
    X, Y = shuffle(wine.data , wine.target)
    
    FIRST_LAYER_SIZE = 10
    

#forxren = FORxREN().extract_rules(KERAS_MODEL, X, Y, INPUT_DIM, conf.FIRST_LAYER_SIZE, conf.EXECUTION_MODE, 
#                                 TEST_PERCENT, conf.MAX_FIDELITY_LOSS, ATTRIBUTES)
