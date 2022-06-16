"""Compatibility library."""

"""pandas"""
try:
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series
    from pandas import concat

    def _assign_pd(table, location, values):
        table.iloc[location] = values

    def _view_pd(table, location):
        return table.iloc[location]

    def _concat_pd(pd_list, axis):
        return concat(pd_list, axis=axis)

    PANDAS_INSTALLED = True

except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:  # type: ignore
        """Dummy class for pandas.Series."""

        pass

    class pd_DataFrame:  # type: ignore
        """Dummy class for pandas.DataFrame."""

        pass


"""numpy"""
import numpy as np


def _assign_np(array, location, values):
    array[location] = values


def _view_np(array, location):
    return array[location]


def _concat_np(np_list, axis):
    if axis == 0:
        return np.vstack(np_list)
    else:
        return np.hstack(np_list)


"""Other"""


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, pd_DataFrame):
        return x.to_numpy()
    elif isinstance(x, pd_Series):
        if x.dtype.name == "category":
            return x.cat.codes.to_numpy()
        elif x.dtype.name == "object":
            return x.fillna("").to_numpy()
        else:
            return x.to_numpy()
    else:
        raise ValueError("Unknown datatype")


def _repeat(x, repeats, axis=0):
    if isinstance(x, pd_DataFrame) or isinstance(x, pd_Series):
        newind = np.arange(x.shape[0]).repeat(repeats)
        return x.iloc[newind].reset_index(drop=True)
    else:
        return np.repeat(x, repeats, axis=axis)


def _tile(x, reps):
    if isinstance(x, pd_DataFrame) or isinstance(x, pd_Series):
        new_ind = np.tile(np.arange(x.shape[0]), reps[0])
        return x.iloc[new_ind].reset_index(drop=True)
    else:
        return np.tile(x, reps)
