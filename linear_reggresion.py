# linear reggresion for n*m
import numpy as np
import matplotlib.pyplot as plt

"""
LINEAR REGGRESION | ASSUMING INPUTS AS ROW MATRICES 



------------------------------------------------------------------------------------------------
In case of input being Column Matries Hypothesis function will be changed from wx^T to w^T x
------------------------------------------------------------------------------------------------
"""


def hypothesis(weights, inputs):
    """ 
    h(w) = w^T * X 
    w = (W_0; w_1; w_2 ; w_3 ; ... ;w_n) 
    x = (1; x_1; x_2; x_3; ... ; x_n) 
    """
    return weights * inputs.transpose()

def optimized_term(weights, X, Y, ChainX, M):
    """
    J(w) = 1/(M) * Sigma{i=1}{m} (h(x^(i)) - y(i)) * x(i)(j)
    """
    result = 0 
    for row_index in range(M):
        result += (hypothesis(weights, X[row_index]) - Y[row_index])*ChainX
    return result/M

def gradient_descent(weight_input, inputs, values, learning_rate):
    """
    repeat{
     w_i := w_i - alpha * d/d(w_1)J(w_1) 
     ...
     }
    """

    weights = weight_input
    row_size = inputs.shape[0]
    column_size = inputs.shape[1]


    #Learning Process 
    for row_index in range(row_size):

        #Updating weights
        for column_index in range(column_size):
            weights[0,column_index] = weights[0, column_index]  - \
            learning_rate * optimized_term(weights, inputs, values, inputs[row_index, column_index], column_size)

    return weights