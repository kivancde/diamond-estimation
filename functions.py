# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:29:10 2022

@author: kvc
"""

import numpy as np

#Score Calculating
def R_squared(prediction_y, real_y):
    above = np.sum(np.square(real_y - prediction_y))
    below = np.sum(np.square(prediction_y - np.mean(real_y)))
    return (1- above/below)

#Ridge Regression
def ridge_regression_algorithm_direct(X, Y, lamda_value):
    I = np.identity(X.shape[1]) #identity matrix
    beta_values = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lamda_value * I), X.T), Y)   #Direct formula
    return 0, beta_values, 0

def ridge_regression_algorithm(train_Y, train_X, decided_learning_rate, decided_lamda, iteration):
    feature_number = 9
    data_number = 53940
    bias_value = np.random.normal(0, 0.1, 1)
    beta_values = np.random.normal(0, 0.1, feature_number)
    MSE_values_array = np.zeros(iteration)
    
    for i in range(0, iteration):
        y_predicted = train_X.dot(beta_values) + bias_value
        
        error_value = train_Y - y_predicted             #Direct error
        squared_error_value = np.square(error_value)    #Take square
        MSE_value = np.mean(squared_error_value)        #Take mean
        MSE_values_array[i] = MSE_value
    
        Derivated_MSE = -2 * (train_X.T).dot(error_value)
        Derivated_Lamda = 2 * decided_lamda * beta_values
        Derivated_Beta = (Derivated_MSE + Derivated_Lamda) / data_number
        Derivated_Bias = -2 * np.mean(np.sum(error_value)) / data_number
        
        beta_values = beta_values - decided_learning_rate * Derivated_Beta
        bias_value = bias_value - decided_learning_rate * Derivated_Bias
        
    return MSE_values_array, beta_values, bias_value


#Neural Network
def relu(x_value):                #relu activation func
    return np.maximum(0, x_value)
    
def relu_grad(x_value): 
    return (x_value > 0) * 1 

class NeuralNetwork(): 
    def __init__(self, neuron_number_in_hidden_layer, num_of_features, learning_rate):
        self.bias_hidden_layer = np.random.normal(0, 0.3, size = (1, neuron_number_in_hidden_layer))                #Bias for hidden layer
        self.weight_hidden_layer= np.random.normal(0, 0.3, size = (num_of_features, neuron_number_in_hidden_layer)) #Weight for hidden layer
        self.bias_output = np.random.normal(0, 0.3, size = (1, 1))                                  #Bias for output
        self.weight_output = np.random.normal(0, 0.3, size = (neuron_number_in_hidden_layer, 1))    #Weight for output
        self.learning_rate = learning_rate      #learning rate
        
    def train_one_iteration(self, train_X, train_Y): 
        loss = 0 
        num_of_samples = train_Y.shape[0] 
        x_value = np.zeros((1, train_X.shape[1])) 
        
        for i in range(num_of_samples): 
            x_value[0, :] = train_X[i, :] 
            y_value = train_Y[i] 
            
            hidden_layer_before = x_value.dot(self.weight_hidden_layer) + self.bias_hidden_layer 
            hidden_layer = relu(hidden_layer_before)                            #putting into ReLU function
            output = hidden_layer.dot(self.weight_output) + self.bias_output    #output
            test_Y_predicted = relu(output)                                     #putting into ReLU function
            
            error_value = y_value - test_Y_predicted                #Error calculation
            loss = loss + error_value * error_value/num_of_samples  #Loss calculation
            
            weight_output_grad = -2 * error_value * hidden_layer.T * relu_grad(output)  #derivative
            bias_output_grad = -2 * error_value * relu_grad(output)                     #derivative
            weight_hidden_layer_gradient = -2 * error_value * relu_grad(output) * x_value.T * (self.weight_output.T * relu_grad(hidden_layer_before)) #derivative
            bias_hidden_layer_gradient = -2 * error_value * relu_grad(output) * (self.weight_output.T * relu_grad(hidden_layer_before))         #derivative
    
            self.weight_output = self.weight_output - self.learning_rate * weight_output_grad   #update
            self.bias_output = self.bias_output - self.learning_rate * bias_output_grad         #update
            self.weight_hidden_layer = self.weight_hidden_layer - self.learning_rate * weight_hidden_layer_gradient #update
            self.bias_hidden_layer = self.bias_hidden_layer - self.learning_rate * bias_hidden_layer_gradient       #update
            
        return loss 
    
    def training(self, train_X, train_Y, maximum_epoch = 25, threshold_to_stop = 0.00005): 
        loss_record = [] 
        for i in range(maximum_epoch):                                            #Iteration on each epoch
            loss = NeuralNetwork.train_one_iteration(self, train_X, train_Y)      #Training each epoch
            loss_record.append(loss[0][0])                                       #Finding and recording loss after each epoch
            if (i >= 1) and (loss_record[-2] - loss_record[-1] < threshold_to_stop):  #Condition to stop the training when converged
                break        
        return loss_record 
        
    def predict(self, train_X): 
        test_Y_predicted_array = np.zeros(train_X.shape[0])     #create the array
        x_value = np.zeros((1, train_X.shape[1]))               #create the array
        
        for i in range(train_X.shape[0]): 
            x_value[0, :] = train_X[i, :] 
            
            hidden_layer_before = x_value.dot(self.weight_hidden_layer) + self.bias_hidden_layer 
            hidden_layer = relu(hidden_layer_before)            #hidden layer update
            
            output = hidden_layer.dot(self.weight_output) + self.bias_output 
            test_Y_predicted = relu(output)         #Getting the predictions
            test_Y_predicted_array[i] = test_Y_predicted 
            
        return test_Y_predicted_array
    
#Decision Tree
class DecisionTree(): 
     def __init__(self): 
         self.first = None              # first (left) node
         self.sec = None                # second (right) node
         self.feature = None 
         self.threshold_to_stop = None  #threshold value  
         self.prediction_y = None 
         
def RSS(first_node, sec_node):
    first_rss = np.sum(np.square(first_node - np.mean(first_node)))
    sec_rss = np.sum(np.square(sec_node - np.mean(sec_node)))
    rss = first_rss + sec_rss
    return rss

def basepredict_dt(test, dt):               #predicts the initial condition
    while (dt.prediction_y is None):
        feature = dt.feature
        threshold_to_stop = dt.threshold_to_stop
        if test[feature] < threshold_to_stop:
            dt = dt.sec
        else:
            dt = dt.first 
    return dt.prediction_y

def predict_set(test_X, dt):              #next prediction acc. to previous
    test_Y_predicted_array = np.zeros(np.shape(test_X)[0])
    for i in range(np.shape(test_X)[0]):
        test_Y_predicted_array[i] = basepredict_dt(test_X[i, :], dt)
    return test_Y_predicted_array

def training_dt(train_X, train_Y, depth, max_depth = 10):
    if  depth >= max_depth: 
        rule = DecisionTree()
        rule.prediction_y = np.mean(train_Y)
        return rule
    
    rule = DecisionTree()
    rule_rss = np.inf
    feature_numb = np.shape(train_X)[1]
    
    for feature_no in range(feature_numb):
        feature_value = np.unique(train_X[:, feature_no])
        feature_value = np.sort(feature_value)
        feature_value = feature_value[1: -1]
        for val in feature_value:
            sec_datum = train_X[:, feature_no] > val
            sec_Y = train_Y[sec_datum]
            first_datum = ~sec_datum
            first_Y = train_Y[first_datum]
            node_rss = RSS(sec_Y, first_Y)
            if rule_rss > node_rss:
                rule_rss = node_rss
                rule.feature = feature_no
                rule.threshold_to_stop = val
    
    if rule.threshold_to_stop is None or rule.feature is None:
        rule.prediction_y = np.mean(train_Y)
        return rule
    
    sec_datum = train_X[:, rule.feature] > rule.threshold_to_stop
    sec_train_X = train_X[sec_datum, :]
    sec_train_Y = train_Y[sec_datum]    
    first_datum = ~sec_datum
    first_train_X = train_X[first_datum, :]
    first_train_Y = train_Y[first_datum]      
    depth = depth + 1

    rule.sec = training_dt(sec_train_X, sec_train_Y, depth, max_depth)    
    rule.first = training_dt(first_train_X, first_train_Y, depth, max_depth)
    return rule
         
