# linear reggresion for n*m
import numpy as np

def h(weights, inputs):
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

    print(row_size)

    #Learning Process 
    for example_index in range(row_size):

        #Calculating cost function term for x[i]
        sigma_term = 0  
        for i in range(row_size):
            sigma_term += float(h(weights, inputs[i]) - values[i])
        print(sigma_term)
        DJDW =  learning_rate*sigma_term/row_size
        
        #print(DJDW)
        
        #Updating weight
        for weight_index in range(column_size):
            weights[0,weight_index] = weights[0, weight_index]  - DJDW * inputs[example_index, weight_index ]

    return weights

X = np.matrix([1,2104,5,1,45,1,1416,3,2,40,1,1534,3,2,30,1,852,2,1,36]).reshape(4,5)
Y = np.matrix([460,232,315,178]).reshape(4,1)
W = np.matrix([1, 10, 5, 2.5,1]) 
A = 0.01



#print(W * X[0].transpose()) 
gradient_descent(W, X, Y, A)