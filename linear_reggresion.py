# linear reggresion for n*m
import numpy as np

def hypothesis(weights, inputs):
    ''' 
     h(w) = w^T * X 
     w = (W_0; w_1; w_2 ; w_3 ; ... ;w_n) 
     x = (1; x_1; x_2; x_3; ... ; x_n) 

    '''
    # apppend column(1,1,...,1) to inputs
    return weights * inputs.transpose()



def gradient_descent(weight_input, inputs, values, learning_rate):
    ''' J(w) = 1/(2M) * Sigma{i=1}{m} (h(x^(i)) - y(i))^2
    repeat{ w_i := w_i - alpha * d/d(w_1)J(w_1) }
    n+1
    '''
    weights = weight_input
    row_size = inputs.shape[0]
    column_size = inputs.shape[1]


    #Learning Process 
    for example_index in range(row_size):

        #Calculating cost function term for x[i]
        sigma_term = 0  
        for i in range(row_size):
            sigma_term += float(hypothesis(weights, inputs[i]) - values[i])

        DJDW =  learning_rate*sigma_term/row_size
        
        #print(DJDW)
        
        #Updating weight
        for weight_index in range(column_size):
            weights[0,weight_index] = weights[0, weight_index]  - DJDW * inputs[example_index, weight_index ]

    return weights



def dummy_dataset():

    W = np.matrix([3, 4])
    X = np.matrix([1,10, 1, 20]).reshape(2,2)
    Y = np.matrix([43, 83]).reshape(2,1)
    A = 0.00001
    
    W = gradient_descent(W, X, Y, A)

    xtrain = np.matrix([1, 30]).reshape(2,1)
    
    assert(float(W * xtrain)) == 123
    print(float(W * xtrain))

if __name__ == '__main__':
    dummy_dataset()




    