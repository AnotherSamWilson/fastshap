from .compat import pd_Series, pd_DataFrame, _view_pd, _view_np, _to_numpy
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


def _get_variable_name_index(variable, data):
    """
    Returns the variable name as a string, and the variable index as an int,
    whether a string or int is passed to a pd.DataFrame or numpy.ndarray

    :param variable: str, int
        The variable we want to return the name and index for.
    :param data:
        The data in which the variable resides.

    :return:
        variable_name (str), variable_index (int)
    """
    if isinstance(data, pd_DataFrame):
        if isinstance(variable, str):
            variable_name = variable
            variable_index = data.columns.tolist().index(variable)
        else:
            variable_name = str(data.columns.tolist()[variable])
            variable_index = variable

    elif isinstance(data, np.ndarray):
        assert isinstance(
            variable, int
        ), "data was numpy array, variable must be an integer"
        variable_name = str(variable)
        variable_index = variable
    else:
        raise ValueError("data not recognized. Must be numpy array or pd.DataFrame")

    return variable_name, variable_index


def _safe_isnan(x):
    if isinstance(x, pd_DataFrame) or isinstance(x, pd_Series):
        return x.isnull().values
    elif isinstance(x, np.ndarray):
        return np.isnan(x)
    else:
        raise ValueError("x not recognized")


def _ensure_2d_array(a):
    if a.ndim == 1:
        return a.reshape(-1, 1)
    else:
        return a


def _fill_missing_cat(x, s):
    assert isinstance(
        x, pd_Series
    ), "Can only fill cat on pandas object or categorical series"
    x_nan = x.isnull()
    if x_nan.sum() > 0:
        if x.dtype.name == "category":
            x = x.cat.add_categories(s).fillna(s)
        elif x.dtype.name == "object":
            x = x.fillna(s)
        else:
            raise ValueError("Series datatype must be object or category.")
    return x


def _keep_top_n_cats_unique(x, n, s, m, codes=False):
    """
    Groups least popular categories together.
    Can return the category codes.

    :param x: pd.Series
        The series to be grouped
    :param n: int
        The number of categories to leave unique (including nans)
    :param s:
        The value to impute non-popular categories as
    :param m:
        The value to impute missing values as.
    :return:
    """

    return_type = "category" if codes else x.dtype.name
    x = _fill_missing_cat(x, m)
    c = x.value_counts().sort_values(ascending=False).index.to_numpy()[:n]
    d = pd_Series(np.where(x.isin(c), x, s), dtype=return_type)
    if codes:
        d = d.cat.codes
    return d


def ampute_data(
    data,
    variables=None,
    perc=0.1,
    random_seed=None,
):
    """
    Ampute Data

    Returns a copy of data with specified variables amputed.

    Parameters
    ----------
     data : Pandas DataFrame
        The data to ampute
     variables : None or list
        If None, are variables are amputed.
     perc : double
        The percentage of the data to ampute.
    random_state: None, int, or np.random.RandomState

    Returns
    -------
    pandas DataFrame
        The amputed data
    """
    amputed_data = data.copy()
    data_shape = amputed_data.shape
    amp_rows = int(perc * data_shape[0])
    random_state = np.random.RandomState(random_seed)

    if len(data_shape) > 1:
        if variables is None:
            variables = [i for i in range(amputed_data.shape[1])]
        elif isinstance(variables, list):
            if isinstance(variables[0], str):
                variables = [data.columns.tolist().index(i) for i in variables]

        if isinstance(amputed_data, pd_DataFrame):
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data.iloc[na_ind, v] = np.NaN

        if isinstance(amputed_data, np.ndarray):
            amputed_data = amputed_data.astype("float64")
            for v in variables:
                na_ind = random_state.choice(
                    np.arange(data_shape[0]), replace=False, size=amp_rows
                )
                amputed_data[na_ind, v] = np.NaN

    else:

        na_ind = random_state.choice(
            np.arange(data_shape[0]), replace=False, size=amp_rows
        )
        amputed_data[na_ind] = np.NaN

    return amputed_data


class Logger:
    def __init__(self, verbose):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)
