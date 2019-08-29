import numpy as np


def perceptron_trains(data: np.array, max_iter: int):
    """
    w_d := 0 for all d = 1, ... D
    b := 0

    for iter = 1 to maxIter do
        for all (x, y) in D
            a := sum_{d = 1}^D ( w_d . x_d )  + b
            if y.a  leq 0 then
                w_d := w_d + y.x_d, for all d = 1 to D
                b :+= y
            end if
        end for
    end for
    return w_d[for d = 1 to D] , b
    """
    weights = np.zeros(data.shape)
    bias = 0

    for i in range(max_iter):
        data = np.random.permutation(data)
        for x, y in data:
            a = weights.dot(x) + bias
            if not a * y > 0:
                weights += y * x
                bias += y
    return weights, bias


def perceptron_test(weights: np.array, bias: float, x_predict: np.array):
    """
    a : = sum {d = 1 to D } ( w_d x_predict_d ) + b
    return SIGN ( a )
    """
    a = weights.dot(x_predict) + bias
    return np.sign(a)
