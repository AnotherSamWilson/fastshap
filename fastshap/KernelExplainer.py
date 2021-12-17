from .compat import pd_DataFrame, pd_Series, _to_numpy
import numpy as np
from itertools import combinations
from datetime import datetime as dt
from scipy.special import binom
from sklearn.linear_model import LinearRegression
from .utils import stratified_continuous_folds, consecutive_slices, Logger


class KernelExplainer:
    def __init__(self, model, background_data):
        """
        The KernelExplainer is capable of calculating shap values for any arbitrary function.
        Multi dimension outputs are not supported yet.

        Parameters
        ----------

        model: Callable.
            Some function which takes the background_data and return
            a numpy array of shape (n, ).

        background_data: pandas.DataFrame or np.ndarray
            The background set which will be used to act as the "missing"
            data when calculating shap values. Smaller background sets
            will make the process fun faster, but may cause shap values
            to drift from their "true" values.

            It is possible to stratify this background set by running
            .stratify_background_set()

        """
        self.model = model
        self.background_data = [background_data]
        self.num_columns = background_data.shape[1]
        self.n_splits = 0

        if isinstance(background_data, pd_DataFrame):
            self.col_names = background_data.columns.tolist()
            self.dtypes = {col: background_data[col].dtype for col in self.col_names}
            from .compat import _assign_pd, _view_pd, _concat_pd
            self._assign = _assign_pd
            self._view = _view_pd
            self._concat = _concat_pd

        if isinstance(background_data, np.ndarray):
            from .compat import _assign_np, _view_np, _concat_np
            self._assign = _assign_np
            self._view = _view_np
            self._concat = _concat_np


    def stratify_background_set(self, n_splits=10):
        """
        Helper function that breaks up the background
        set into folds stratified by the model output
        on the background set. The larger n_splits,
        the smaller each background set is.

        Parameters
        ----------

        n_splits: int
            The number split datasets created. Raise
            this number to calculate shap values
            faster, at the expense of integrating
            over a smaller dataset.

        """

        self.background_data = self._concat(self.background_data, axis=0)
        background_preds = self.model(self.background_data)
        folds = stratified_continuous_folds(background_preds, n_splits)
        self.background_data = [self._view(self.background_data, f) for f in folds]
        self.n_splits = n_splits

    def get_theoretical_array_expansion_sizes(
            self,
            outer_batch_size=100,
            inner_batch_size=10,
            n_coalition_sizes=3,
            background_fold_to_use=None,
    ):
        """
        Gives the maximum expanded array sizes that can be
        expected for different parameters. Use this function
        to determine appropriate parameters for your machine.

        Parameters
        ----------

        outer_batch_size: int
            See calculate_shap_values()

        inner_batch_size: int
            See calculate_shap_values()

        n_coalition_sizes: int
            See calculate_shap_values()

        background_fold_to_use: int
            See calculate_shap_values()

        Returns
        -------

        Arrays returned are, in order:
            1) Mask Matrix
            2) Linear Target Size (outer batch)
                Depends on outer_batch_size, n_coalition_sizes
            3) Model Evaluation Set Size (inner batch)
                Depends on inner_batch_size, n_coalition_sizes
        """

        if background_fold_to_use is not None:
            assert background_fold_to_use < self.n_splits, (
                f"There are only {self.n_splits} splits of the background dataset"
            )
        else:
            background_fold_to_use = 0

        n_choose_k_midpoint = (self.num_columns - 1) / 2.0
        coalition_sizes_to_combinate = np.min([np.ceil(n_choose_k_midpoint), n_coalition_sizes]).astype("int32")
        symmetric_sizes_to_combinate = np.min([np.floor(n_choose_k_midpoint), n_coalition_sizes]).astype("int32")
        coalition_pared_ind = [
            (cs in range(symmetric_sizes_to_combinate))
            for cs in range(coalition_sizes_to_combinate)
        ]

        # Number of coalition combinations (including complement) per coalition size.
        total_coalitions_per_coalition_size = [
            binom(self.num_columns, i+1).astype("int32") * (2 if coalition_pared_ind[i] else 1)
            for i in range(coalition_sizes_to_combinate)
        ]

        mask_matrix_size = (
            np.sum(total_coalitions_per_coalition_size),
            self.num_columns
        )
        linear_target_size = (
            np.sum(total_coalitions_per_coalition_size),
            outer_batch_size
        )
        inner_model_eval_set_size = (
            inner_batch_size *
            self.background_data[background_fold_to_use].shape[0],
            self.num_columns
        )

        return mask_matrix_size, linear_target_size, inner_model_eval_set_size

    def calculate_shap_values(
            self,
            data,
            outer_batch_size=100,
            inner_batch_size=10,
            n_coalition_sizes=3,
            background_fold_to_use=None,
            verbose=True,
            linear_model=None
    ):
        """
        Calculates approximate shap values for data.


        Parameters
        ----------

        data: pandas.DataFrame or np.ndarray
            The data to calculate the shap values for

        outer_batch_size: int
            Shap values are calculated all at once in the outer batch.
            The outer batch requires the creation of the Linear Targets,
            which is an array of size(`Total Coalitions`, `outer_batch_size`)

            To determine an appropriate outer_batch_size, play around with
            the .get_theoretical_array_expansion_sizes() function.

        inner_batch_size: int
            To get the Linear Targets, an array of the following size must
            be evaluated by the model: (`inner_batch_size`, `# background samples`)
            and then aggregated.

            To determine an appropriate inner_batch_size, play around with
            the .get_theoretical_array_expansion_sizes() function.

        n_coalition_sizes: int
            The coalition sizes, starting at 1, and their complements which will
            be used to calculate the shap values.

            Not all possible column combinations can be evaluated to calculate
            the shap values. The shap kernel puts more weight on lower
            coalition sizes (and their complements). These also tend to have
            fewer possible combinations.

            For example, if our dataset has 10 columns, and we set
            n_coalition_sizes = 3, then the process will calculate the shap
            values by integrating over all column combinations of
            size 1, 9, 2, 8, 3, and 7.

        background_fold_to_use: None or int
            If the background dataset has been stratified, select one of
            them to use to calculate the shap values.

        verbose: bool
            Should progress be printed?

        Returns
        -------

        Returns an array of shape (# data rows, # columns + 1).
        The last column is the expected value.

        """

        logger = Logger(verbose)

        if background_fold_to_use is not None:
            assert background_fold_to_use < self.n_splits, (
                f"There are only {self.n_splits} splits of the background dataset"
            )
        else:
            background_fold_to_use = 0

        if linear_model is None:
            linear_model = LinearRegression(fit_intercept=False)
        else:
            assert hasattr(linear_model, "fit")

        working_background_data = self.background_data[background_fold_to_use]
        background_preds = self.model(working_background_data)
        background_pred_mean = background_preds.mean()
        return_type = background_preds.dtype

        # Do cursory glances at the background and new data
        if isinstance(data, pd_DataFrame):
            assert set(data.columns) == set(self.col_names), "Columns don't match"
        else:
            assert data.shape[1] == self.num_columns, "Different number of columns"

        num_new_samples = data.shape[0]
        col_array = np.arange(self.num_columns)
        n_background_rows = working_background_data.shape[0]
        index = np.arange(num_new_samples)
        outer_batches = [
            index[i:np.min([i + outer_batch_size, index[-1] + 1])]
            for i in range(0, num_new_samples, outer_batch_size)
        ]

        data_preds = self.model(data)
        shap_values = np.empty(shape=(data.shape[0], data.shape[1] + 1)).astype(return_type) # +1 for expected value

        # Determine how many coalition sizes in the symmetric kernel are paired.
        # There may be one unpaired weight if the number of columns is even.
        # This is because we calculate k and it's complement for each coalition size
        # i.e. If we have 10 columns, when we evaluate the size 1 subsets, we also evaluate
        # the complement, which consists of all the size 9 subsets. However, we shouldn't
        # evaluate the complement of the size 5 subsets, because we would double count.
        n_choose_k_midpoint = (self.num_columns - 1) / 2.0
        coalition_sizes_to_combinate = np.min([np.ceil(n_choose_k_midpoint), n_coalition_sizes]).astype("int32")
        symmetric_sizes_to_combinate = np.min([np.floor(n_choose_k_midpoint), n_coalition_sizes]).astype("int32")
        coalition_pared_ind = [
            (cs in range(symmetric_sizes_to_combinate))
            for cs in range(coalition_sizes_to_combinate)
        ]

        # Number of coalition combinations (excluding complement) per coalition size.
        coalitions_per_coalition_size = [
            binom(self.num_columns, cs).astype("int32")
            for cs in range(1, coalition_sizes_to_combinate + 1)
        ]
        # Number of coalition combinations (including complement) per coalition size.
        total_coalitions_per_coalition_size = [
            coalitions_per_coalition_size[i] * (2 if coalition_pared_ind[i] else 1)
            for i in range(coalition_sizes_to_combinate)
        ]
        cc_cs = np.cumsum(coalitions_per_coalition_size)
        num_total_coalitions_to_run = np.sum(total_coalitions_per_coalition_size)
        logger.log(f"Number of coalitions to run per sample: {str(num_total_coalitions_to_run)}")

        # Theoretical weights if we use all possible coalition sizes (before scaling)
        coalition_size_weights = np.array([
            (self.num_columns - 1.0) / (i * (self.num_columns - i))
            for i in range(1, self.num_columns)
        ])
        # Weights are symmetric, so we can
        selected_coalition_size_weights = np.concatenate(
            [coalition_size_weights[:coalition_sizes_to_combinate],
            coalition_size_weights[-symmetric_sizes_to_combinate:]]
        )
        selected_coalition_size_weights /= selected_coalition_size_weights.sum()

        self.insert_times = []
        self.func_eval_times = []
        if isinstance(data, pd_DataFrame):
            self.convert_to_pd_times = []
        for outer_batch in outer_batches:
            # outer_batch = outer_batches[0]
            logger.log(f"Starting Samples {outer_batch[0]} - {outer_batch[-1]}")
            outer_batch_length = len(outer_batch)
            masked_coalition_avg = np.empty(shape=(num_total_coalitions_to_run,outer_batch_length)).astype(return_type)
            mask_matrix = np.zeros(shape=(num_total_coalitions_to_run, self.num_columns)).astype("int8")
            coalition_weights = np.empty(num_total_coalitions_to_run)

            inner_batches_relative = [
                slice(i,i+inner_batch_size)
                for i in range(0, outer_batch_length, inner_batch_size)
            ]
            inner_batches_absolute = [
                slice(outer_batch[0] + i,outer_batch[0] + i + inner_batch_size)
                for i in range(0, outer_batch_length, inner_batch_size)
            ]

            for coalition_size in range(1, coalition_sizes_to_combinate + 1):

                # coalition_size = 1

                logger.log(f"Coalition Size {str(coalition_size)}")
                has_complement = coalition_size <= symmetric_sizes_to_combinate
                choose_count = binom(self.num_columns, coalition_size).astype("int32")
                inds = combinations(np.arange(self.num_columns), coalition_size)
                listinds =[list(i) for i in inds]
                coalition_weight = selected_coalition_size_weights[coalition_size - 1] / choose_count

                # Get information about where these coalitions are stored in the arrays
                start = (cc_cs - coalitions_per_coalition_size)[coalition_size - 1]
                end = cc_cs[coalition_size - 1]
                coalition_loc = np.arange(start, end)
                mask_matrix[coalition_loc.reshape(-1,1), listinds] = 1
                coalition_weights[coalition_loc] = coalition_weight

                if has_complement:
                    end_c = num_total_coalitions_to_run - start
                    start_c = num_total_coalitions_to_run - end
                    coalition_c_loc = np.arange(start_c, end_c)
                    mask_matrix[coalition_c_loc] = 1-mask_matrix[coalition_loc]
                    coalition_weights[coalition_c_loc] = coalition_weight

                for inner_batch_i in range(len(inner_batches_absolute)):
                    # inner_batch_i = 0
                    slice_absolute = inner_batches_absolute[inner_batch_i]
                    slice_relative = inner_batches_relative[inner_batch_i]
                    inner_batch_size = len(range(*slice_relative.indices(masked_coalition_avg.shape[1])))
                    batch_data = self._view(data, (slice_absolute, slice(None)))
                    repeated_batch_data = np.repeat(
                        _to_numpy(batch_data).astype(return_type),
                        repeats=n_background_rows,
                        axis=0
                    )

                    # For each mask (and complement, if it is paired)
                    for coalition_i in range(choose_count):
                        # coalition_i = 0
                        masked_data = np.tile(
                            _to_numpy(working_background_data).astype(return_type),
                            (inner_batch_size, 1)
                        )

                        if has_complement:
                            masked_data_complement = masked_data.copy()
                        else:
                            masked_data_complement = None

                        mask = listinds[coalition_i]
                        mask_c = np.setdiff1d(col_array, mask)

                        mask_slices = consecutive_slices(mask)
                        mask_c_slices = consecutive_slices(mask_c)

                        # Overwrite masked data with real batch data.
                        # Order of masked_data is mask, background, batch
                        # Broken up into possible slices for faster insertion
                        s = dt.now()
                        for ms in mask_slices:

                            masked_data[:,ms] = repeated_batch_data[:,ms]

                        if has_complement:
                            for msc in mask_c_slices:
                                masked_data_complement[:,msc] = repeated_batch_data[:,msc]

                        self.insert_times.append(dt.now() - s)

                        # Coalitions are stored at the beginning of the array
                        # Complements are stored at the opposite end.
                        if isinstance(data, pd_DataFrame):
                            s = dt.now()
                            masked_data = self._concat([
                                pd_Series(masked_data[:,self.col_names.index(col)], dtype=self.dtypes[col], name=col)
                                for col in self.col_names
                            ], axis=1)
                            if has_complement:
                                masked_data_complement = self._concat([
                                    pd_Series(masked_data_complement[:, self.col_names.index(col)], dtype=self.dtypes[col], name=col)
                                    for col in self.col_names
                                ], axis=1)
                            self.convert_to_pd_times.append(dt.now() - s)

                        s = dt.now()
                        masked_coalition_avg[coalition_loc[coalition_i],slice_relative] = self.model(masked_data).reshape(
                            inner_batch_size, n_background_rows
                        ).mean(axis=1)
                        if has_complement:
                            masked_coalition_avg[coalition_c_loc[coalition_i], slice_relative] = self.model(masked_data_complement).reshape(
                                inner_batch_size, n_background_rows
                            ).mean(axis=1)
                        self.func_eval_times.append((dt.now() - s))


            # Back to outer batch
            mean_model_output = data_preds.mean()
            linear_features = mask_matrix[:, :-1] - mask_matrix[:, -1].reshape(-1, 1)

            for outer_batch_sample in range(outer_batch_length):
                linear_target = masked_coalition_avg[:,outer_batch_sample] - mean_model_output - (
                    mask_matrix[:, -1] * (
                        data_preds[outer_batch[outer_batch_sample]] - background_pred_mean
                    )
                )
                # lr = LinearRegression(fit_intercept=False)
                linear_model.fit(X=linear_features, sample_weight=coalition_weights, y=linear_target)
                shap_values[outer_batch[outer_batch_sample], :-2] = linear_model.coef_

        shap_values[:,-2] = data_preds - (shap_values[:,:-2].sum(1) + background_pred_mean)
        shap_values[:,-1] = background_pred_mean

        return shap_values
