import numpy as np


def consecutive_slices(data):
    runs = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
    return [slice(min(run), max(run) + 1) for run in runs]


def stratified_continuous_folds(y, nfold):
    """
    Create primitive stratified folds for continuous data.
    """
    elements = len(y)
    assert elements >= nfold, "more splits then elements."
    sorted = np.argsort(y)
    val = [sorted[range(i, len(y), nfold)] for i in range(nfold)]
    return val


def _ensure_2d_array(a):
    if a.ndim == 1:
        return a.reshape(-1, 1)
    else:
        return a


class Logger:
    def __init__(self, verbose):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)
