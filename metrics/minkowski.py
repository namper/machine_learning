def minkowski_distance(u: list, v: list, p: int) -> float:
    depth = len(u)
    assert len(u) == len(v)
    sigma = 0
    for i in range(depth):
        sigma += abs(u[i] - v[i]) ** p
    return sigma ** (1 / p)
