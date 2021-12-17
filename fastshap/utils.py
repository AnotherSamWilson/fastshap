import numpy as np


def print_big_num(x):
    x = 10030000000
    n = ""
    billion = 1000000000
    million = 1000000
    thousand = 1000

    if x > billion:
        billions = int(x / billion)
        n = n + f"{str(billions)} Billion\n"
        x -= billions * billion

    if x > million:
        millions = int(x / million)
        n = n + f"{str(millions)} million\n"
        x -= millions * million


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


class Logger:
    def __init__(self, verbose):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)
