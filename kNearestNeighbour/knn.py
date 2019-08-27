"""
KNN with binary( pos , neg )  classifier

!!! assuming 2 dimensionality (x, y)

Pseudo:
    KNN-Predict(D, K, x)
    S := [ ]
    for n = 1 to N do
        S := S (+) < d{x_n, x}, n>
    end for

    S := sort( S )

    y := 0
    for k = 1 to K do
        < dist, n> := S_k
        y := y + y_n
    end for
    return SIGN( y )
"""

from numpy import sign


def knn_predict(data_set, hyper_k, x):
    s_by_euclid = list()
    for i, n in enumerate(data_set):
        s_by_euclid.append([abs(n[1] - x), i])
        s_by_euclid.sort(key=lambda f: f[1])
    y = 0
    for i in range(hyper_k):
        y += s_by_euclid[i]
    return sign(y)
