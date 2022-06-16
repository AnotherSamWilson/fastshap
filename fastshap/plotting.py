# TODO:
# Code is a mess because categorical variables
# for numpy arrays was initially supported. Clean up.

from .compat import pd_DataFrame, _view_pd, _view_np, _to_numpy
from .utils import (
    _get_variable_name_index,
    _safe_isnan,
    _fill_missing_cat,
    _keep_top_n_cats_unique,
)
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
from warnings import warn


__BIN_CREDIBILITY_THRESHOLD = 50
__NUMERIC_MISSING_COUNT_IND_THRESHOLD = 30
__NUMERIC_MISSING_PERCENT_IND_THRESHOLD = 0.3
__MAX_AUTO_BINS = 10
__DOT_SIZE = 50
__MISSING_X_SIZE = 30


def __preprocess_data(data, variable):
    num_rows, num_columns = data.shape
    if isinstance(data, pd_DataFrame):
        data_context = "pd_DataFrame"
        col_names = data.columns.tolist()
        catvars = {
            col_names.index(col): data[col].dtype.name in {"category", "object"}
            for col in col_names
        }

        _view = _view_pd

    elif isinstance(data, np.ndarray):
        data_context = "np_array"
        _view = _view_np
        # Do not support categorical variables for numpy arrays
        catvars = {col: False for col in range(num_columns)}

    else:
        raise ValueError("data not recognized.")

    var_name, var_index = _get_variable_name_index(variable, data)

    return data_context, num_rows, num_columns, var_name, var_index, catvars, _view


def __determine_plotting_context(shap_values, output_index, interaction_variable):
    """
    There are 12 different plotting scenarios:
        Multiclass plotting of numeric variable shap values
        Multiclass plotting of categorical variable shap values
        Multiclass plotting a single output dimension numeric shap values with categorical interaction
        Multiclass plotting a single output dimension numeric shap values with categorical interaction
        Multiclass plotting a single output dimension categorical shap values with numeric interaction
        Multiclass plotting a single output dimension categorical shap values with categorical interaction
        Single output plotting categorical variable with categorical interaction
        Single output plotting categorical variable with numeric interaction
        Single output plotting categorical variable without interaction
        Single output plotting numeric variable with categorical interaction
        Single output plotting numeric variable with numeric interaction
        Single output plotting numeric variable without interaction
    """
    interact = interaction_variable is not None
    find_interaction = interaction_variable == "auto"
    multiclass = False

    # Model was multiclass
    if shap_values.ndim == 3:
        if output_index is None:
            output_index = [i for i in range(shap_values.shape[2])]
        elif isinstance(output_index, int):
            output_index = [output_index]
        assert isinstance(
            output_index, list
        ), "output_index must be an int or list of ints. If it is None, all classes are plotted."
        # If only 1 output dimension is being plotted, we can treat it like a standard
        # single output variable, and allow interaction plotting.
        if len(output_index) > 1:
            assert (
                interaction_variable == "auto" or interaction_variable is None
            ), "interaction_variable cannot be specified if plotting multiple output dimensions"
            multiclass = True

    return multiclass, interact, find_interaction, output_index


def get_variable_interactions(shap_values, data, variable, interaction_bins=5):
    """
    Returns an array representing the weighted average of each variables ability
    to explain the variance (r-square) in the shap values of our variable of interest
    over the different bins. Binning is done in the case of non-linear relationships.
    Variables with higher r-squares in this regard tend to have more interesting
    interactions with the variable of interest.

    :param shap_values:
    :param data:
    :param variable:
    :param categorical_variables:
    :param num_var_bins:
    :return:
    """
    (
        data_context,
        num_rows,
        num_columns,
        var_name,
        var_index,
        catvars,
        _view,
    ) = __preprocess_data(data, variable)
    linreg = LinearRegression(fit_intercept=True)
    variable_values = _view(data, (slice(None), var_index))
    variable_shap_values = shap_values[:, var_index]
    var_nan = _safe_isnan(variable_values)
    potential_interaction_vars = np.setdiff1d(list(range(num_columns)), [var_index])

    # If the variable is categorical, the bins will just be the top n categories.
    if catvars[var_index]:
        var_bins = _keep_top_n_cats_unique(
            x=variable_values, n=interaction_bins, s="OTHER", m="MISSING", codes=True
        ).values

    else:
        var_bins = np.zeros(shape=variable_values.shape).astype("int32")
        nona_bins = np.digitize(
            variable_values[~var_nan],
            np.unique(
                np.nanpercentile(
                    variable_values,
                    np.linspace(0, 100, interaction_bins, endpoint=False)[1:],
                )
            ),
            right=True,
        )
        var_bins[~var_nan] = nona_bins
        var_bins[var_nan] = -1

    bin_unq, bin_cnt = np.unique(var_bins, return_counts=True)
    min_bin_cnt = bin_cnt.min()
    if min_bin_cnt < 15:
        warn(
            f"WARNING: Lowest bin count is {min_bin_cnt}. Consider lowering interaction_bins."
        )
    vbc = bin_cnt[np.argsort(bin_unq)] / bin_cnt.sum()
    bin_unq.sort()

    # piv_r_squares ends up being the weighted average of each variables ability
    # to explain the variance (r-square) in the shap values of our variable of interest
    # over the different bins. Variables with higher r-squares in this regard tend
    # to have more interesting interactions with the variable of interest.
    piv_r_squares = np.zeros(num_columns)
    for piv in potential_interaction_vars:
        piv_values = _to_numpy(_view(data, (slice(None), piv)))

        for i in bin_unq:
            bin_ind = np.where(var_bins == i)[0]
            piv_bin = piv_values[bin_ind]

            # numpy.unique() fails quite miserably at handling nan values.
            # np.nan == np.nan -> False
            # We must replace them with -inf to make this work nicely.
            piv_bin_isnan = _safe_isnan(piv_bin)
            if piv_bin_isnan.sum() > 0:
                piv_bin[piv_bin_isnan] = -np.Inf

            if catvars[piv]:
                # We can either leave missing values empty or give a one-hot encoding.
                cats, indexes = np.unique(piv_bin, return_inverse=True)
                onehot = np.zeros((len(bin_ind), len(cats)))
                onehot[np.arange(len(bin_ind)), indexes] = 1

                # Delete the variable with the lowest count.
                onehot = np.delete(onehot, np.argmin(onehot.sum(0)), 1)

                # Break out if we have more columns than rows or NO columns
                onehot_cols = onehot.shape[1]
                if onehot_cols > onehot.shape[0] or onehot_cols == 0:
                    break

                # See how well the piv explains the variance in the shap values in this bin.
                linreg.fit(onehot, variable_shap_values[bin_ind])
                piv_r_squares[piv] += (
                    linreg.score(onehot, variable_shap_values[bin_ind]) * vbc[i]
                )

            else:

                piv_nonan_mean = np.mean(piv_bin[~piv_bin_isnan])

                # If there are enough missing data in the piv, give it it's own
                # categorical indicator, otherwise just impute with mean.
                if (
                    piv_bin_isnan.mean() > __NUMERIC_MISSING_PERCENT_IND_THRESHOLD
                    or piv_bin_isnan.sum() > __NUMERIC_MISSING_COUNT_IND_THRESHOLD
                ):
                    piv_bin[piv_bin_isnan] = piv_nonan_mean
                    lin_feat = (
                        np.concatenate([piv_bin, np.where(piv_bin_isnan, 1, 0)])
                        .reshape(2, -1)
                        .transpose()
                    )
                else:
                    lin_feat = piv_bin.reshape(-1, 1)
                    lin_feat[piv_bin_isnan] = piv_nonan_mean

                linreg.fit(lin_feat, variable_shap_values[bin_ind])
                piv_r_squares[piv] += (
                    linreg.score(lin_feat, variable_shap_values[bin_ind]) * vbc[i]
                )

        # Remove the value associated with the variable we are measuring interactions for.
        piv_r_squares = np.delete(piv_r_squares, var_index)

        return piv_r_squares, potential_interaction_vars


def plot_variable_effect_on_output(
    shap_values,
    data,
    variable,
    interaction_variable="auto",
    interaction_bins=None,
    output_index=None,
    class_labels=None,
    max_rows=10000,
    max_cat_levels=5,
    style_context="seaborn-deep",
    cmap=None,
    alpha=1.0,
    plot_adjust_func=None,
):
    assert (
        data.shape[0] == shap_values.shape[0]
    ), "These shap values didn't come from this data."
    assert (
        data.shape[1] + 1 == shap_values.shape[1]
    ), "These shap values didn't come from this data."
    (
        data_context,
        num_rows,
        num_columns,
        var_name,
        var_index,
        catvars,
        _view,
    ) = __preprocess_data(data, variable)

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("", ["royalblue", "fuchsia"])

    # Subset if we need to, can't get around copying here
    if num_rows > max_rows:
        ind = np.random.choice(range(num_rows), max_rows, replace=False)
        data = _view(data, (ind, slice(None))).copy()
        shap_values = shap_values[ind]

    var_values = _view(data, (slice(None), var_index))

    (
        multiclass,
        interact,
        find_interaction,
        output_index,
    ) = __determine_plotting_context(shap_values, output_index, interaction_variable)

    # Multiclass
    if multiclass:

        sv = shap_values[:, var_index, output_index]
        if class_labels is None:
            class_labels = [str(i) for i in output_index]
        else:
            assert len(class_labels) == len(output_index)

        if catvars[var_index]:
            assert (
                data_context == "pd_DataFrame"
            ), "Cannot plot categorical vars unless from pandas."
            plot_df = pd_DataFrame(sv, columns=class_labels)
            plot_df[var_name] = var_values
            plot_df[var_name] = _keep_top_n_cats_unique(
                plot_df[var_name], max_cat_levels, "OTHER", "MISSING"
            )
            _plot_multi_with_cat_var(
                var_name=var_name,
                plot_df=plot_df,
                style_context=style_context,
                plot_adjust_func=plot_adjust_func,
            )

        else:
            _plot_multi_with_num_var(
                var_name=var_name,
                var_values=var_values,
                sv=sv,
                output_index=output_index,
                class_labels=class_labels,
                style_context=style_context,
                cmap=cmap,
                alpha=alpha,
            )

    else:
        sv = shap_values[:, var_index]

        if not interact:
            iv_values = None
            iv_cat = None
            iv_name = None

        else:

            if find_interaction:

                if interaction_bins is None:
                    interaction_bins = int(num_rows / __BIN_CREDIBILITY_THRESHOLD)
                    interaction_bins = np.min([__MAX_AUTO_BINS, interaction_bins])

                piv_r_squares, interaction_vars = get_variable_interactions(
                    shap_values=shap_values,
                    data=data,
                    variable=variable,
                    interaction_bins=interaction_bins,
                )

                interaction_variable = interaction_vars[np.argmax(piv_r_squares)]
                iv_name, iv_index = _get_variable_name_index(
                    int(interaction_variable), data
                )

            else:
                iv_name, iv_index = _get_variable_name_index(interaction_variable, data)

            iv_values = _view(data, (slice(None), iv_index))
            iv_cat = catvars[iv_index]

        if catvars[var_index]:
            assert (
                data_context == "pd_DataFrame"
            ), "Cannot plot categorical variables unless from pandas."
            plot_df = pd_DataFrame(
                {var_name: var_values, f"{var_name} SHAP Values": sv}
            )
            plot_df[var_name] = _keep_top_n_cats_unique(
                plot_df[var_name], max_cat_levels, "OTHER", "MISSING"
            )
            if interact:
                plot_df[iv_name] = iv_values
                if iv_cat:
                    plot_df[iv_name] = _keep_top_n_cats_unique(
                        plot_df[iv_name], max_cat_levels, "OTHER", "MISSING"
                    )
            _plot_cat_var_shap_values(
                var_name=var_name,
                iv_name=iv_name,
                plot_df=plot_df,
                iv_cat=iv_cat,
                cmap=cmap,
                alpha=alpha,
                style_context=style_context,
                plot_adjust_func=plot_adjust_func,
            )

        else:
            _plot_numeric_var_shap_values(
                data_context=data_context,
                var_name=var_name,
                var_values=var_values,
                sv=sv,
                iv_name=iv_name,
                iv_values=iv_values,
                iv_cat=iv_cat,
                max_cat_levels=max_cat_levels,
                cmap=cmap,
                alpha=alpha,
                style_context=style_context,
                plot_adjust_func=plot_adjust_func,
            )


def _plot_multi_with_num_var(
    var_name,
    var_values,
    sv,
    output_index,
    class_labels,
    style_context,
    cmap,
    alpha,
):
    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        for oi in range(len(output_index)):
            cl = class_labels[oi]
            oind = output_index[oi]
            plt.scatter(
                x=var_values,
                y=sv[:, oind],
                label=cl,
                marker=".",
                cmap=cmap,
                alpha=alpha,
            )
        plt.legend(title="Class")
        plt.xlabel(var_name)
        plt.ylabel(f"{var_name} \n SHAP Values")
        plt.title("Dependence Plot")


def _plot_multi_with_cat_var(var_name, plot_df, style_context, plot_adjust_func):
    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        axes = plot_df.boxplot(by=var_name, layout=(1, -1), return_type="axes", rot=45)
        axes.iloc[0].set_ylabel(f"{var_name} \n SHAP Values")
        fig = axes[0].get_figure()
        fig.suptitle("Dependence Plot")
        if plot_adjust_func is not None:
            plot_adjust_func(plt, fig, axes)


def _plot_cat_var_shap_values(
    var_name, iv_name, plot_df, iv_cat, cmap, alpha, style_context, plot_adjust_func
):
    plot_interaction = iv_name is not None

    if plot_interaction:

        if iv_cat:
            _plot_cat_var_w_cat_iv(
                var_name, iv_name, plot_df, style_context, plot_adjust_func
            )
        else:
            _plot_cat_var_w_num_iv(
                var_name, iv_name, plot_df, cmap, alpha, style_context, plot_adjust_func
            )
    else:
        _plot_cat_var_wo_iv(var_name, plot_df, style_context, plot_adjust_func)


def _plot_cat_var_wo_iv(
    var_name,
    plot_df,
    style_context,
    plot_adjust_func,
):
    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        axes = plot_df.boxplot(by=var_name, layout=(1, -1), return_type="axes", rot=45)
        axes.iloc[0].set_ylabel(f"{var_name} \n SHAP Values")
        fig = axes[0].get_figure()
        fig.suptitle("Dependence Plot")
        plt.title("")
        if plot_adjust_func is not None:
            plot_adjust_func(plt, fig, axes)


def _plot_cat_var_w_cat_iv(var_name, iv_name, plot_df, style_context, plot_adjust_func):
    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        axes = plot_df.groupby(iv_name).boxplot(
            by=var_name, layout=(1, -1), return_type="axes", rot=45
        )
        axes[0][0].set_ylabel(f"{var_name} \n SHAP Values")
        fig = axes[0][0].get_figure()
        fig.suptitle(iv_name)
        if plot_adjust_func is not None:
            plot_adjust_func(fig, axes)


def _plot_cat_var_w_num_iv(
    var_name, iv_name, plot_df, cmap, alpha, style_context, plot_adjust_func
):
    sv_name = f"{var_name} SHAP Values"
    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        # cmap = plt.get_cmap(cmap)
        groups = plot_df.groupby(var_name)
        for name, group in groups:
            plt.scatter(
                x=group[iv_name],
                y=group[sv_name],
                label=name,
                marker=".",
                s=__DOT_SIZE,
                # cmap=cmap,
                alpha=alpha,
            )
        # produce a legend with the unique colors from the scatter
        plt.legend(title=var_name)
        plt.xlabel(iv_name)
        plt.ylabel(f"SHAP values \n {var_name}")
        plt.title("Dependence Plot")

        if plot_adjust_func is not None:
            plot_adjust_func(plt)


def _plot_numeric_var_shap_values(
    data_context,
    var_name,
    var_values,
    sv,
    iv_name,
    iv_values,
    iv_cat,
    max_cat_levels,
    cmap,
    alpha,
    style_context,
    plot_adjust_func,
):
    # Variable values are unknown. These might not exist
    var_nan = _safe_isnan(var_values)
    sideplot_ind = var_nan
    sideplot_sv = sv[sideplot_ind]
    plot_interaction = iv_values is not None

    # If we are plotting iv_values...
    if plot_interaction:
        iv_nan = _safe_isnan(iv_values)

        if iv_cat:
            assert (
                data_context == "pd_DataFrame"
            ), "iv is categorical, but data is not pandas."
            plot_df = pd_DataFrame(
                {
                    var_name: var_values,
                    iv_name: iv_values,
                    f"{var_name} SHAP Values": sv,
                }
            )
            plot_df[iv_name] = _keep_top_n_cats_unique(
                plot_df[iv_name], max_cat_levels, "OTHER", "MISSING"
            )
            _plot_num_var_w_cat_iv(
                var_name=var_name,
                iv_name=iv_name,
                plot_df=plot_df,
                sideplot_sv=sideplot_sv,
                cmap=cmap,
                alpha=alpha,
                style_context=style_context,
                plot_adjust_func=plot_adjust_func,
            )

        else:

            sctrplt_cc_ind = np.bitwise_and(~var_nan, ~iv_nan)
            sctrplt_ivnan_ind = np.bitwise_and(~var_nan, iv_nan)

            # Complete Case samples. These should definitely exist
            sctrplt_cc_var = var_values[sctrplt_cc_ind]
            sctrplt_cc_iv = iv_values[sctrplt_cc_ind]
            sctrplt_cc_sv = sv[sctrplt_cc_ind]

            # Interaction Variable unknown samples. These might not exist.
            # sctrplt_miss_color data
            plot_missing_color = iv_nan.sum() > 0
            sctrplt_ivnan_var = var_values[sctrplt_ivnan_ind]
            sctrplt_ivnan_sv = sv[sctrplt_ivnan_ind]

            _plot_num_var_w_num_iv(
                var_name=var_name,
                iv_name=iv_name,
                sctrplt_cc_var=sctrplt_cc_var,
                sctrplt_cc_iv=sctrplt_cc_iv,
                sctrplt_cc_sv=sctrplt_cc_sv,
                plot_missing_color=plot_missing_color,
                sctrplt_ivnan_var=sctrplt_ivnan_var,
                sctrplt_ivnan_sv=sctrplt_ivnan_sv,
                sideplot_sv=sideplot_sv,
                cmap=cmap,
                alpha=alpha,
                style_context=style_context,
                plot_adjust_func=plot_adjust_func,
            )

    else:
        sctrplt_cc_ind = np.bitwise_and(~var_nan)
        sctrplt_cc_var = var_values[sctrplt_cc_ind]
        sctrplt_cc_sv = sv[sctrplt_cc_ind]
        _plot_num_var_wo_iv(
            var_name=var_name,
            sctrplt_cc_var=sctrplt_cc_var,
            sctrplt_cc_sv=sctrplt_cc_sv,
            sideplot_sv=sideplot_sv,
            alpha=alpha,
            style_context=style_context,
            plot_adjust_func=plot_adjust_func,
        )


def _plot_num_var_w_num_iv(
    var_name,
    iv_name,
    sctrplt_cc_var,
    sctrplt_cc_iv,
    sctrplt_cc_sv,
    plot_missing_color,
    sctrplt_ivnan_var,
    sctrplt_ivnan_sv,
    sideplot_sv,
    cmap,
    alpha,
    style_context,
    plot_adjust_func,
):
    plot_sideplot = len(sideplot_sv) > 5

    if plot_sideplot:
        ncols = 2
        width_ratios = [4, 1]
    else:
        ncols = 1
        width_ratios = [1]

    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable

    with plt.style.context(style_context):
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(sctrplt_cc_sv.min(), sctrplt_cc_sv.max())
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            sharey=True,
            gridspec_kw=dict(
                width_ratios=width_ratios,
            ),
        )
        axs = axs if plot_sideplot else [axs]
        axs[0].scatter(
            x=sctrplt_cc_var,
            y=sctrplt_cc_sv,
            c=sctrplt_cc_iv,
            marker=".",
            s=__DOT_SIZE,
            cmap=cmap,
            alpha=alpha,
        )
        clb = fig.colorbar(sm, ax=axs[0], pad=0.0)
        clb.ax.set_title(iv_name, fontsize=10)
        axs[0].set_xlabel(var_name)
        axs[0].set_ylabel(f"SHAP values \n {var_name}")

        if plot_missing_color:
            axs[0].scatter(
                x=sctrplt_ivnan_var,
                y=sctrplt_ivnan_sv,
                color="black",
                marker="x",
                alpha=alpha,
                s=__MISSING_X_SIZE,
            )
        axs[0].set_title("Dependence Plot", fontsize=14)

        if plot_sideplot:
            axs[1].violinplot(dataset=sideplot_sv)
            axs[1].set_title(f"Missing \n {var_name}", fontsize=10)

        if plot_adjust_func is not None:
            plot_adjust_func(plt, fig, axs)

        return plt, fig, axs


def _plot_num_var_w_cat_iv(
    var_name,
    iv_name,
    plot_df,
    sideplot_sv,
    cmap,
    alpha,
    style_context,
    plot_adjust_func,
):
    plot_sideplot = len(sideplot_sv) > 5
    sv_name = f"{var_name} SHAP Values"

    if plot_sideplot:
        ncols = 2
        width_ratios = [4, 1]
    else:
        ncols = 1
        width_ratios = [1]

    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        # cmap = plt.get_cmap(cmap)
        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            sharey=True,
            gridspec_kw=dict(
                width_ratios=width_ratios,
                # wspace=0.1,
                # hspace=0.1,
            ),
        )
        axs = axs if plot_sideplot else [axs]
        groups = plot_df.groupby(iv_name)
        for name, group in groups:
            axs[0].scatter(
                x=group[var_name],
                y=group[sv_name],
                label=name,
                marker=".",
                s=__DOT_SIZE,
                # cmap=cmap,
                alpha=alpha,
            )
        # produce a legend with the unique colors from the scatter
        axs[0].legend(title=iv_name)
        axs[0].set_xlabel(var_name)
        axs[0].set_ylabel(f"SHAP values \n {var_name}")
        axs[0].title.set_text("Dependence Plot")

        if plot_sideplot:
            axs[1].violinplot(dataset=sideplot_sv)
            axs[1].set_title(f"Missing \n {var_name}", fontsize=10)

        if plot_adjust_func is not None:
            plot_adjust_func(plt, fig, axs)

    return plt, fig, axs


def _plot_num_var_wo_iv(
    var_name,
    sctrplt_cc_var,
    sctrplt_cc_sv,
    sideplot_sv,
    alpha,
    style_context,
    plot_adjust_func,
):

    plot_sideplot = len(sideplot_sv) > 5

    if plot_sideplot:
        ncols = 2
        width_ratios = [4, 1]
    else:
        ncols = 1
        width_ratios = [1]

    from matplotlib import pyplot as plt

    with plt.style.context(style_context):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            sharey=True,
            gridspec_kw=dict(
                width_ratios=width_ratios,
                # wspace=0.1,
                # hspace=0.1,
            ),
        )
        axs = axs if plot_sideplot else [axs]
        axs[0].scatter(
            x=sctrplt_cc_var, y=sctrplt_cc_sv, marker="o", s=__DOT_SIZE, alpha=alpha
        )
        axs[0].set_xlabel(var_name)
        axs[0].set_ylabel(f"SHAP values \n {var_name}")
        axs[0].title.set_text("Dependence Plot")

        if plot_sideplot:
            axs[1].violinplot(dataset=sideplot_sv)
            axs[1].set_title(f"Missing \n {var_name}", fontsize=10)

        if plot_adjust_func is not None:
            plot_adjust_func(plt, fig, axs)

    return plt, fig, axs
