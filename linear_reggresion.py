# linear reggresion for n*m
import numpy as np
import matplotlib.pyplot as plt


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
    for example_index in range(M):
        result += (hypothesis(weights, X[example_index]) - Y[example_index])*ChainX
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
    for example_index in range(row_size):

        #Updating weights
        for weight_index in range(column_size):
            weights[0,weight_index] = weights[0, weight_index]  - \
            learning_rate * optimized_term(weights, inputs, values, inputs[example_index, weight_index], column_size)

    return weights



def dummy_dataset():

    W = np.matrix([100, 100])
    x = list()
    y = list()
    for i in range(1000):
        x.append(1)
        x.append(i)
        y.append(i*4 + 3)

    _length = len(x)//2
    X = np.matrix(x).reshape(_length,2)
    Y = np.matrix(y).reshape(_length,1)
    A = 0.00000001
    


    W = gradient_descent(W, X, Y, A)
    xtrain = np.matrix([1, 30]).reshape(2,1)
    
    guess = float(W * xtrain)
    assert(guess - 123) <= 1
    print(guess)

    
    x = [i for i in range(1000)]
    plt.plot(np.array(x), np.array(y))
    plt.plot(30,guess, 'X')
    plt.show()


def dummy_bigdataset():
    
    X = np.matrix([1,2104,5,1,45,1,1416,3,2,40,1,1534,3,2,30,1,852,2,1,36]).reshape(4,5)
    Y = np.matrix([460,232,315,178]).reshape(4,1)
    W = np.matrix([1, 10, 5, 2.5,1]) 
    A = 0.000001

    W = gradient_descent(W, X, Y, A)
    xtrain = np.matrix([1,852,2,1,36]).reshape(5,1)

    print(float(W * xtrain))



if __name__ == '__main__':
    dummy_dataset()