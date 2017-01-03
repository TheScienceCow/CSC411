""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    y = []
    for i in range(len(data)):
        sigmoid_value = sigmoid(np.dot(weights[:-1].T,data[i])+weights[-1])
        y.append(sigmoid_value)
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    correct = 0
    for i in range(len(targets)):
        if ((y[i]) >= 0.5) and targets[i]==1:
            correct += 1
        elif ((y[i]) < 0.5) and targets[i]==0:
            correct += 1
    frac_correct = float(correct) / len(targets)

    y = np.array(y)
    ce = -np.sum(targets*np.log(y+1e-100)+(1-targets)*np.log(1-y+1e-100))/len(targets)
    
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        sigmoid_value_list = []
        
        for i in range(len(targets)):
            sigmoid_value = sigmoid(np.dot(weights[:-1].T,data[i])+weights[-1])
            sigmoid_value_list.append(sigmoid_value)

        f = -1*np.sum(targets*np.log(np.array(sigmoid_value_list).reshape(-1,1)+0.001)+(1-targets)*np.log(1-np.array(sigmoid_value_list).reshape(-1,1)+0.001))

        df = np.append(-1 * np.dot((targets - np.array(sigmoid_value_list).reshape(-1,1)).T, data), -1 * np.sum(targets - sigmoid_value_list)).reshape(-1, 1) 
        
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    sigmoid_value_list = []
    alpha = hyperparameters["weight_decay"]
        
    for i in range(len(targets)):
        sigmoid_value = sigmoid(np.dot(weights[:-1].T,data[i])+weights[-1])
        sigmoid_value_list.append(sigmoid_value)

    f = -1*np.sum(targets*np.log(np.array(sigmoid_value_list).reshape(-1,1)+0.001)+(1-targets)*np.log(1-np.array(sigmoid_value_list).reshape(-1,1)+0.001))
    f += alpha * np.sum(np.dot(weights[:-1].transpose(), weights[:-1])/2)
    df = np.append(-1 * np.dot((targets - np.array(sigmoid_value_list).reshape(-1,1)).T, data), -1 * np.sum(targets - sigmoid_value_list)).reshape(-1, 1) 
    df = df + hyperparameters['weight_decay']*weights   
    
    return f, df
