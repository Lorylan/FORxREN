from  FORxREN import FORxREN
import configuration as conf


if (conf.DATASET == "iris"):
    pass
elif (conf.DATASET = "wbc"):
    pass
else:
    pass

keras_model, X,Y

forxren = FORxREN().extract_rules(KERAS_MODEL, X, Y, conf.INPUT_DIM, conf.FIRST_LAYER_SIZE, conf.EXECUTION_MODE, 
                                  conf.TEST_PERCENT, conf.MAX_FIDELITY_LOSS, conf.ATTRIBUTES)
