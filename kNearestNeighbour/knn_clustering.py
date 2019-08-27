import sys

sys.path.append('..')

import random
import numpy as np
from metrics.minkowski import minkowski_distance
from typing import Tuple


def d_euclid(*args, **kwargs):
    return minkowski_distance(p=2, *args, **kwargs)


def arg_min(centers: list, x: tuple) -> int:
    distance = d_euclid(centers[0], x)
    miu: int = 0

    for i in range(len(centers)):
        centroid = centers[i]
        dist = d_euclid(centroid, x)
        if dist < distance:
            distance = dist
            miu = i

    return miu


def k_means(data: list, k: int, limit: float = 0.00001) -> Tuple[list, list]:
    """
    algorithm K-Means(D, K):
        for k = 1 to K do
            m_k := some random location
        end for
        repeat
            for n = 1  to N do
                z_n := arg_min_k { || m_k - x_n || }
            end for
            for k = 1 to K do
                X_k := {x_n | z_n = k }
                x_k := mean{ X_k }
            end for
        until m's stop changing
        return z
    """
    centers = random.sample(data, k)
    final_clustering = []
    z = [arg_min(centers, x) for x in data]
    while len(final_clustering) < 2:
        z = [arg_min(centers, x) for x in data]
        for i in range(k):
            cluster = [data[n] for n in range(len(data)) if z[n] == i]
            miu = np.mean(cluster, axis=0)
            dist = d_euclid(miu, centers[i])
            if dist < limit:
                # this is good clustering center
                final_clustering.append(miu)
            centers[i] = miu
    return z, centers


if __name__ == '__main__':
    # testing out
    d = [
        np.array([3, 4]),
        np.array([4, 3]),
        np.array([3, 3]),
        np.array([5, 6]),
        np.array([5, 7]),
        np.array([5, 6.5]),
        np.array([5.5, 7]),
        np.array([5.5, 6.5]),
    ]
    clustering, centroids = k_means(d, 2)
    print(f'centroids: {centroids} clustering: {clustering}')
