import numpy as np
from .knn_clustering import d_euclid


def knn_train_loo(data: np.array, k: int) -> int:
    err = list()
    for _ in range(k):
        error = 0
        for n, d in enumerate(data):
            s = np.array([[d_euclid(d, d_m), m] for m, d_m in enumerate(data)].pop())
            s = np.sort(s)
            y = 0
            for li in s:
                dist, m = li
                y_m = data[m]
                y += y_m
                if y != y_m:
                    error += 1
        err.append({
            'error': error,
            'k': k
        })
    return min(err, key=lambda x: x['error'])['k']
