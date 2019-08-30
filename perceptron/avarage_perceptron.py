import numpy as np


def average_perceptron_train(data: np.array, max_iter: int) -> tuple:
    """
    w := [ 0, ... 0]
    b := 0
    u := [0, ... 0]
    d := 0

    for iter = 1 to maxIter do
        for all (x, y) in D
            a := sum_{d = 1}^D ( w_d . x_d )  + b
            if y.a  leq 0 then
                w := w + yx
                b := b + y
                u := w + ycx
                d := d + cy
            end if
            c := c + 1
        end for
    end for
    return w - u/c , b - d/c
    """
    weights, bias = np.zeros(data.shape), 0
    u, b = weights, bias
    c = 1
    for _ in range(max_iter):
        data = np.random.permutation(data)
        for x, y in data:
            if not y * (weights.dot(x) + bias) > 0:
                weights += y * x
                bias += y
                u += y * c * x
                b += y * c
            c += 1
    return weights - u / c, bias - b / c
