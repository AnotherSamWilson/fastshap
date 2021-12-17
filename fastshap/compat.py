"""Compatibility library."""
"""Stolen from lightgbm"""

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
    if isinstance(x, pd_DataFrame):
        return x.values

    else:
        return x



