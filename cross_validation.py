import numpy as np
from typing import Callable, List, Dict


def cross_validate(learning_algorithm: Callable, data: np.array, hyper_parameters: np.array, k: int, *args,
                   **kwargs) -> List[Dict]:
    error = np.inf
    hyper_li = list()
    for hyper_parameter in hyper_parameters:
        err = list()
        for i in range(k):
            train = [d for n, d in enumerate(data) if n % hyper_parameter != hyper_parameter - 1]
            test = [d for n, d in enumerate(data) if n % hyper_parameter == hyper_parameter - 1]
            model_predict = learning_algorithm(data=train, *args, **kwargs)
            err.append(model_predict(data=test))
        av_err = np.mean(err)
        if av_err < error:
            error = av_err
            hyper_li.append({
                'hyper': error
            })
    return hyper_li
