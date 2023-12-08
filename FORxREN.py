
import numpy as np
import copy
import tensorflow as tf
from keras.utils import np_utils
import pandas as pd


class FORxREN():
    """
    Abstract class for rule extraction

    This class defines the interface for extracting rules from a dataset, but does not provide a concrete implementation.
    Subclasses of this class must implement the build_model and process_data method.

    """
    
    def __init__(self):
        """
        Contructs all the necesary attibutes for the Rule Extractor object

        Args:
            classes (int): Amount of clasification classes in the dataset
            input_dim (int): Size of the input layer of the NN
            first_layer_size (int): Size of the first hidden layer of the NN
            per_train (int): Percentage of data used for training of the NN
            per_test (int): Percentage of data used for testing of the NN
            path_data (str): Directory where the dataset file is located
            mode (int) : COMPLETAR

        """
        
    @property
    def X(self):
        return self.__X

    @property
    def Y(self):
        return self.__Y

    @property
    def model(self):
        return self.__model

    @property
    def error_examples(self):
        return self.__error_examples

    @property
    def acc_origin(self):
        return self.__acc_origin

    @property
    def null_neurons(self):
        return self.__null_neurons

    @property
    def rule_mat(self):
        return self.__rule_mat

    @property
    def rule_order(self):
        return self.__rule_order

    @property
    def network_y(self):
        return self.__network_y


    #####################################################################################

    @X.setter
    def X(self,X):
        self.__X= X

    @Y.setter
    def Y(self,Y):
        self.__Y= Y

    @model.setter
    def model(self,model):
        self.__model= model

    @error_examples.setter
    def error_examples(self,error_examples):
        self.__error_examples= error_examples

    @acc_origin.setter
    def acc_origin(self,acc_origin):
        self.__acc_origin= acc_origin

    @null_neurons.setter
    def null_neurons(self,null_neurons):
        self.__null_neurons= null_neurons

    @rule_mat.setter
    def rule_mat(self,rule_mat):
        self.__rule_mat= rule_mat

    @rule_order.setter
    def rule_order(self, rule_order):
        self.__rule_order= rule_order

    @network_y.setter
    def network_y(self, network_y):
        self.__network_y = network_y

    #####################################################################################

    def __network_output(self):
        """
        Classificates every element in the dataset and returns a list with the output class from the network for each element in the dataset.

        Returns:
            new_y : List of integers from the network output, where each number represents the class assigned to an element from the dataset
        """
        data = self.X
        x_pred = self.model.predict(data, verbose=0)

        return [np.argmax(x_pred[elem]) for elem in range(len(data)) ]

    def calculate_train(self, data):
        """
        Splits the input data into training and validation sets based on the percentage of data to use for training.

        Args:
            data: A list or array of the data to be split.

        Returns:
            train_data: A list or array of the training data.
        """
        per_train = self.per_train

        end_idx = int(len(data)*per_train)
        train_data =  data[:end_idx]
        return train_data

    def calculate_test(self, data):
        """
        Splits the input data into training and validation sets based on the percentage of data to use for testing.

        Args:
            data: A list or array of the data to be split.

        Returns:
            test_data: A list or array of the testing data.
        """
        per_test = self.per_test

        start_idx = int(len(data) - len(data)*per_test)
        test_data = data[start_idx:]
        return test_data

    def model_accuracy(self, use_y_from_network):
        """
        Calculates the accuracy of the model on the test data.

        Args:
            use_y_from_network: A boolean indicating whether to use the output from the network for Y_test.

        Returns:
            test_accuracy: A float indicating the accuracy of the model on the test data.
        """
        model = self.model
        X_test = self.calculate_test(self.X)

        if use_y_from_network:
            Y_test = self.calculate_test(self.network_y)
        else:
            Y_test = self.calculate_test(self.Y)

        Y_test=np_utils.to_categorical(Y_test, num_classes=self.classes)
        result = model.evaluate(X_test,Y_test, verbose=0)
        #print(("Network test: {} - Network accuracy: {}").format(result[0],result[1]))
        return result[1]

    def __missclassified_counter(self):
        """
        Removes each neuron in the first layer of the model and calculates the number of misclassified elements for each input.

        Returns:
            e: A list of lists of integers. Each sublist contains the indices of the misclassified elements for a particular input.
        """
        model= self.model
        input_dim = self.input_dim
        first_layer_size =self.first_layer_size
        X = self.X
        Y = self.network_y

        e=[]

        layer1 = model.get_layer("layer1")
        weights = layer1.get_weights()

        #---------------------------
        for i in range(input_dim):
            elems = []
            no_neuron_i = copy.deepcopy(weights)
            no_neuron_i[0][i] = tf.convert_to_tensor([0 for each in range(first_layer_size)])
            layer1.set_weights(no_neuron_i)

            # missclassified elements
            for i in range(len(X)):
                data = tf.convert_to_tensor([X[i]])
                result = model.predict(data, verbose=0) # i get an array
                class_result =  np.argmax(result)# i get the id of the class with the maximum probability
                if(Y[i] != class_result):
                    elems.append(i)
            e.append(elems)
        layer1.set_weights(weights)
        #print("--------------------------------")
        #print("Amount of missclasified elements: ")
        #for i in range(input_dim):
        #    print("Input nº {} has {} missclasified elements ".format(i, len(e[i])))
        return e


    def __min_len(self,erased):
        """
        Returns the minimum number of misclassified elements for the inputs not in the 'erased' list.

        Args:
            erased: A list of integers representing the indices of inputs that have been erased.

        Returns:
            min: An integer representing the minimum number of misclassified elements for the inputs not in the 'erased' list.
        """
        e = self.error_examples

        min = len(self.X) + 1
        for i in range(len(e)):
            if i not in erased:
                if(len(e[i]) < min):
                    min = len(e[i])
        return min

    def __neuron_filter(self):
        """
        Filters out the neurons in the first layer of the neural network whose activation is not necessary for the
        network's high accuracy / fidelity. The function uses a threshold to determine the number of missclassified examples that
        a neuron is responsible for, and if this number is lower than the threshold, the neuron is removed.
        The function returns a list with the indices of the removed neurons.

        Returns:
            erased (list): A list with the indices of the removed neurons
        """
        e = self.error_examples
        first_layer_size = self.first_layer_size
        input_dim = self.input_dim
        acc_origin = self.acc_origin
        model = self.model

        layer1 = model.get_layer("layer1")
        weights = layer1.get_weights()
        fidelity = 1
        min_fidelity = fidelity-0.05
        new_weights = copy.deepcopy(weights)
        next = True
        another_erased = False
        erased = []

        #print("Network Accuracy: ", acc_origin)

        while(next):
            threshold = self.__min_len(erased)
            #print("-----------------------")
            #print("Threshold: ", threshold)

            B=[]
            another_erased = False

            for x in range(input_dim):
                if(len(e[x]) <= threshold and x not in erased):
                    print("Add neuron {} to B".format(x))
                    B.append(x)
                    another_erased = True

            if another_erased:
                aux = copy.deepcopy(new_weights)
                for x in range(len(B)):
                    idx_err = B[x]
                    aux[0][idx_err] = tf.convert_to_tensor([0 for each in range(first_layer_size)])
                    #print("Deactivated Neuron: ",idx_err)
                layer1.set_weights(aux)

                # We find out the accuracy
                #print(" Partial fidelity ")
                fid_aux = self.model_accuracy(True)

                if((fid_aux >= min_fidelity and not(len(erased) >= input_dim - 1 ))):
                    fidelity = fid_aux
                    new_weights=copy.deepcopy(aux)
                    for x in range(len(B)):
                        #print(("Neuron {} is insignificant").format(B[x]))
                        erased.append(B[x])
                else:
                    next = False
            else:
                next = False

        layer1.set_weights(new_weights)
        return erased


    def __build_matrix(self):
        """
         Build a matrix of rules based on the input data and the misclassified examples.

        Returns:
            rule_mat : A matrix of rules with dimensions (input_dim, classes).
        """
        input_dim = self.input_dim
        classes = self.classes
        B = self.null_neurons
        e = self.error_examples
        Y = self.Y
        X = self.X

        alpha = 0.25
        class_groups = []
        rule_mat = np.empty((input_dim,classes),dtype=tuple)
        #For each input neuron
        for i in range(input_dim):
            #If the neuron is not insignificant
            if(not i in B):
            #For each class we add an empty array
                [class_groups.append([]) for j in range(classes)]
                #For each missclassified example without the neuron I we add it to it's class array
                [class_groups[Y[x]].append(x) for x in e[i]]
                #For each class
                for j in range(classes):
                    #If neuron is significant in the error for that class,  calculate max and min, then add it to matrix.
                    if(len(class_groups[j]) > alpha * len(e[j])):
                        # Check if the data is continuous or discrete
                        
                        max = -9999999999
                        min = 999999999999
                        for w in class_groups[j]:
                            if(X[w][i] > max):
                                max = X[w][i]
                            if(X[w][i] < min):
                                min = X[w][i]
                        rule_mat[i][j] = (min,max)
                        
        #print("-----Obtained Matrix-----")
        #pprint.pprint(rule_mat)
        #print("#################################")
        #print("-----First Rules-----")
        self.write_rules(rule_mat, True)
        return rule_mat

    def write_rules(self, mat, write):
        """
        Writes the generated rules from a matrix to a rule list,
        and sorts the rules in decreasing order of complexity.

        Args:
            mat (numpy.ndarray): The matrix containing the rules.
            write (boolean ): A boolean value that indicates whether the rules should be printed to the console or not.
        """
        classes = self.classes
        input_dim = self.input_dim
        rules = []
        r_order = []

        #For each class
        for k in range(classes):
            #Set up condition counter and rule
            j = 0
            rule = ""
            data = []
            attributes = ["sepal_length","sepal_width","petal_length","petal_width"]
            #For each input neuron
            for i in range(input_dim):
                #If neuron has an interval for that class
                if(mat[i][k] != None):
                    if(mat[i][k][0] != None or mat[i][k][1] != None):
                        #We add the condition, if is the first one add it one way, else concatenate it to the partial rule
                        if(mat[i][k][0] != None and mat[i][k][1] != None):
                            aux_rule = '({} ≥ {} ∧ {} ≤ {})'.format(attributes[i], mat[i][k][0], attributes[i] ,mat[i][k][1])
                        elif(mat[i][k][0] != None):
                            aux_rule = '{} ≥ {}'.format(attributes[i], mat[i][k][0])
                        else:
                            aux_rule = '{} ≤ {}'.format(attributes[i], mat[i][k][1])
                       
                        if(j != 0):
                            rule = '{} ∧ {}'.format(rule, aux_rule)
                        else:
                            rule = aux_rule
                        j = j+1

            #Once the rule is done save it, we add a second version in case is the last rule and we add the amount of conditions (J)
            data.append('if({}) then Class = {}'.format(rule,k))
            data.append('else Class = {}'.format(k))
            data.append(j)
            #We add the information in a list
            rules.append(data)
        #For each rule
        for i in range(len(rules)):
            max_count = -1
            rule = ""
            max_idx = -1
            #Find the rule with most conditions
            for j in range(len(rules)):
                if(rules[j][2] > max_count):
                    rule = rules[j][0]
                    max_count = rules[j][2]
                    max_idx = j
            #Remove it from the list
            poped_rule = rules.pop(max_idx)
            r_order.append(int(poped_rule[1][-1]))
            #If is the last one then turn it into an else
            if(len(rules) == 0):
                rule = poped_rule[1]
            #Print the rule
            if(write):
                print(rule)
        #if(write):
            #print("###########################################")
        self.rule_order = r_order

    def __rule_prune(self, rule_idx, input_idx, sub_idx):
        """
        Prunes a rule from the rule matrix and returns the accuracy of the classification result.


        Args:
            rule_idx (int): Index of the rule to be pruned.
            input_idx (int): Index of the input for which the rule is to be pruned.
            sub_idx (int): Index of the sub-rule to be pruned. Default is -1, indicating that the entire rule is to be pruned.

        Returns:
            float: The accuracy of the classification result after pruning the specified rule from the rule matrix.
        """
        rule_mat = self.rule_mat
        Y_test = self.network_y
        X_test = self.X

        aux_mat = copy.deepcopy(rule_mat)

        if(sub_idx != -1):
            if(sub_idx == 1):
                new_value = (aux_mat[input_idx][rule_idx][0], None)
            else:
                new_value = (None, aux_mat[input_idx][rule_idx][1])
            aux_mat[input_idx][rule_idx] = new_value
        else:
            aux_mat[input_idx][rule_idx] = None
        self.write_rules(aux_mat, False)
        return self.__classify(aux_mat, self.rule_order, X_test, Y_test,False)[1]

    def __rule_pruning(self):
        """
        Prunes rules that do not contribute to classification accuracy, by iteratively
        removing individual conditions from each rule, and comparing the classification
        performance before and after the removal.

        Returns:
            rule_mat : The updated rule matrix after pruning.
        """
        rule_mat = self.rule_mat
        rule_order = self.rule_order
        
        Y_test = self.network_y
        X_test = self.X

        prune_matrix = [0 for x in range(self.classes)]

        for j in range(self.classes):
            for i in range(self.input_dim):
                if(rule_mat[i][j] != None):
                    prune_matrix[j] += 1

        r_fid = self.__classify(rule_mat, rule_order, X_test, Y_test,False)[1]
        for j in range(len(rule_order)):
            for i in range(self.input_dim):
                if(prune_matrix[j] > 1):
                    if(rule_mat[i][rule_order[j]] != None):
                        new_fid = self.__rule_prune(rule_order[j], i, -1)
                        if  new_fid >= r_fid:
                            rule_mat[i][rule_order[j]] = None
                            r_fid = new_fid
                            prune_matrix[j] -= 1
                        else:
                            new_fid = self.__rule_prune(rule_order[j], i, 0)
                            if new_fid >= r_fid:
                                rule_mat[i][rule_order[j]] = (None, rule_mat[i][rule_order[j]][1])
                                r_fid = new_fid
                                prune_matrix[j] -= 0.5
                            new_fid = self.__rule_prune(rule_order[j], i, 1)
                            if new_fid >= r_fid:
                                rule_mat[i][rule_order[j]] = (rule_mat[i][rule_order[j]][0], None)
                                r_fid = new_fid
                                prune_matrix[j] -= 0.5

        #print("-----Pruned Rules-----")
        self.write_rules(rule_mat, True)
        return rule_mat

    def __classify(self,mat, rule_order, x_test, y_test, should_print):
        """
        Classifies the input data based on the given rule order and membership matrix.

        Args:
            mat (list of list): Membership matrix.
            rule_order (list): Order of the rules to be applied.
            x_test (list): Input data to be classified.
            y_test (list): arget output corresponding to the input data.

        Returns:
            tuple: A tuple containing the indices of the misclassified samples and the classification accuracy.
        """

        input_dim = self.input_dim
        
        total = len(y_test)
        correct_class = 0
        wrong_class = []

        for x in range(len(x_test)):
            for j in range(len(rule_order)):
                ok = True
                if(j != len(rule_order) - 1):
                    for i in range(input_dim):
                        if(mat[i][rule_order[j]] != None):
                            if(mat[i][rule_order[j]][0] != None and not(x_test[x][i] >= mat[i][rule_order[j]][0])):
                                ok = False
                            if(mat[i][rule_order[j]][1] != None and not(x_test[x][i] <= mat[i][rule_order[j]][1])):
                                ok = False
                          
                if(ok or j == (len(rule_order) - 1)):
                    if(rule_order[j] == y_test[x]):
                        correct_class += 1
                    else:
                        wrong_class.append(x)
                    break

        #if(should_print):
          #print(("Elements correctly classified: {}%").format(round(correct_class/ total, 4)*100))
        return (wrong_class, round(correct_class / total, 4))

    def __rule_updation(self):
        """
        This method updates the rules matrix by identifying the missclassified samples and searching for new
        lower and upper boundaries to improve the classification fidelity.

        Returns:
             tuple: The updated rule matrix.
        """
        classes = self.classes
        input_dim = self.input_dim
        rule_mat = self.rule_mat
        rule_order = self.rule_order

        missclassified = []

        Y_test = self.network_y
        X_test = self.X
        data = self.__classify(rule_mat, rule_order, X_test, Y_test,False)
        pruned_fidelity = data[1]

        for k in range(classes):
            missclassified.append([])

        for x in data[0]:
            missclassified[Y_test[x]].append(x)

        for k in range(classes):
            for i in range(input_dim):
                if(rule_mat[i][k] != None):
                    
                    if(rule_mat[i][k][0] != rule_mat[i][k][1]):
                        origin_max = rule_mat[i][k][1]
                        origin_min = rule_mat[i][k][0]
                        max = -99999999
                        min = 99999999
                        for x in missclassified[k]:
                            if(origin_max != None):
                                if(X_test[x][i] > max and X_test[x][i] < origin_max):
                                    max = X_test[x][i]
                            if(origin_min != None):
                                if(X_test[x][i] < min and X_test[x][i] > origin_min):
                                    min = X_test[x][i]
                        if(max != -99999999):
                            origin_max = max
                            aux_mat = copy.deepcopy(rule_mat)
                            aux_mat[i][k] = (aux_mat[i][k][0], max)
                            new_data = self.__classify(aux_mat, rule_order, X_test, Y_test, False)
                            new_fid = new_data[1]
                            if(new_fid > pruned_fidelity):
                                #print(("El valor {} en la posicion [{},{}] fue cambiado por {}").format(rule_mat[i][k][1], i, k, max))
                                rule_mat[i][k] = (rule_mat[i][k][0], max)
                                pruned_fidelity = new_fid
                        if(min != 99999999):
                            origin_min = min
                            aux_mat = copy.deepcopy(rule_mat)
                            aux_mat[i][k] = (min, aux_mat[i][k][1])
                            new_data = self.__classify(aux_mat, rule_order, X_test, Y_test, False)
                            new_fid = new_data[1]
                            if(new_fid > pruned_fidelity):
                                #print(("El valor {} en la posicion [{},{}] fue cambiado por {}").format(rule_mat[i][k][0], i, k, min))
                                rule_mat[i][k] = (min, rule_mat[i][k][1])
                                pruned_fidelity = new_fid

        #print("-----Updated Rules-----")
        #self.write_rules(rule_mat, True)
        return rule_mat

    def extract_rules(self, keras_model, X,Y, input_dim, first_layer_size, execution_mode):
        """
        Trains and evaluates a model using a rule-based approach for classification.
        This method processes the input data, builds a model, and then performs k-fold cross-validation.
        During each fold, it computes the accuracy, fidelity, and a rule matrix for the model.
        At the end of the k-fold cross-validation, it selects the best rule matrix based on fidelity,
        writes the rules for the selected matrix, and prints the average accuracy and fidelity across all folds.
        
        #TODO: Revisar la explicación 
        
        Args:

            keras_model (): 
            X ():
            Y ():
            input_dim (int): Size of the input layer of the NN
            first_layer_size (int): Size of the first hidden layer of the NN
            classes (int): Amount of clasification classes in the dataset
            execution_mode (int) : COMPLETAR
        """
        
        #+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++
        
        self.model = keras_model
        self.X = X
        self.Y = Y
        self.input_dim = input_dim
        self.first_layer_size = first_layer_size
        self.mode = execution_mode
        self.classes = pd.unique(Y)
        
        #+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++#+#++

        self.acc_origin = self.model_accuracy(False)
        self.network_y = self.__network_output()
        self.error_examples = self.__missclassified_counter()

        if(execution_mode in [1,2,3]):
            self.null_neurons = self.__neuron_filter()
        else:
            self.null_neurons = []

        self.rule_mat = self.__build_matrix()

        if(execution_mode in [1,2]):
            self.rule_mat = self.__rule_pruning()

        if(execution_mode == 1):
            self.rule_mat = self.__rule_updation()
            

        #print("-----Accuracy-----")
        #self.__classify(self.rule_mat, self.rule_order, self.X, self.Y,True)[1]
        #print("-----Fidelity-----")
        #self.__classify(self.rule_mat, self.rule_order, self.X, self.network_y,True)[1]
        #print("-----Final Matrix-----")
        #pprint.pprint(self.rule_mat)
        
