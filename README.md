
[![Build
Status](https://app.travis-ci.com/AnotherSamWilson/fastshap.svg?branch=main)](https://app.travis-ci.com/github/AnotherSamWilson/fastshap)
[![CodeCov](https://codecov.io/gh/AnotherSamWilson/fastshap/branch/master/graphs/badge.svg?branch=master&service=github)](https://codecov.io/gh/AnotherSamWilson/fastshap)

## fastshap: A fast, approximate shap kernel

<a href='https://github.com/AnotherSamWilson/miceforest'><img src='https://i.imgur.com/nbrAQso.png' align="right" height="300" /></a>

Calculating shap values can take an extremely long time. `fastshap` was
designed to be as fast as possible, using numpy. This is done by
utilizing inner and outer batch assignments to keep the calculations
inside vectorized operations as often as it can.

A kernel explainer is ideal in situations where:

1)  The model you are using does not have model-specific methods
    available (for example, support vector machine)
2)  You need to explain a modeling pipeline which includes variable
    transformations.
3)  You wish to explain the raw probabilities in a classification model,
    instead of the log-odds.

**WARNING** This package specifically offers a kernel explainer, which
can calculate approximate shap values of f(X) towards y for any function
f. Much faster shap solutions are available specifically for gradient
boosted trees.

### Installation

This package can be installed using pip:

``` bash
# Using pip
$ pip install fastshap --no-cache-dir
```

You can also download the latest development version from this
repository. If you want to install from github with conda, you must
first run `conda install pip git`.

``` bash
$ pip install git+https://github.com/AnotherSamWilson/fastshap.git
```

### Basic Usage

We will use the iris dataset for this example. Here, we load the data
and train a simple lightgbm model on the dataset:

``` python
from sklearn.datasets import load_iris
import pandas as pd
import lightgbm as lgb
import numpy as np

# Define our dataset and target variable
data = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
data.rename({"target": "species"}, inplace=True, axis=1)
data["species"] = data["species"].astype("category")
target = data.pop("sepal length (cm)")

# Train our model
dtrain = lgb.Dataset(data=data, label=target)
lgbmodel = lgb.train(
    params={"seed": 1, "verbose": -1},
    train_set=dtrain,
    num_boost_round=10
)

# Define the function we wish to build shap values for.
model = lgbmodel.predict

preds = model(data)
```

We now have a `model` which takes a Pandas dataframe, and returns
predictions. We can create an explainer that will use `data` as a
background dataset to calculate the shap values of any dataset we wish:

``` python
import fastshap

ke = fastshap.KernelExplainer(model, data)
sv = ke.calculate_shap_values(data, verbose=False)

print(all(preds == sv.sum(1)))
```

    ## True

### Stratifying the Background Set

We can select a subset of our data to act as a background set. By
stratifying the background set on the results of the model output, we
will usually get very similar results, while decreasing the caculation
time drastically.

``` python
ke.stratify_background_set(5)
sv2 = ke.calculate_shap_values(
  data, 
  background_fold_to_use=0,
  verbose=False
)

print(np.abs(sv2 - sv).mean(0))
```

    ## [1.74764532e-03 1.61829094e-02 1.99534408e-03 4.02640884e-16
    ##  1.71084747e-02]

What we did is break up our background set into 10 different sets,
stratified by the model output. We then used the first of these sets as
our background set. We then compared the average difference between
these shap values, and the shap values we obtained from using the entire
dataset.

### Choosing Batch Sizes

If the entire process was vectorized, it would require an array of size
(`# Samples * # Coalitions * # Background samples`, `# Columns`). Where
`# Coalitions` is the sum of the total number of coalitions that are
going to be run. Even for small datasets, this becomes enormous.
`fastshap` breaks this array up into chunks by splitting the process
into a series of batches.

This is a list of the large arrays and their maximum size:

  - Global
      - Mask Matrix (`# Coalitions`, `# Columns`) dtype = `int8`
  - Outer Batch
      - Linear Targets (`Total Coalition Combinations`, `Outer Batch
        Size`, `Output Dimension`) dtype = `adaptive`
  - Inner Batch
      - Model Evaluation Features (`Inner Batch Size`, `# Background
        Samples`) dtype = `adaptive`

The `adaptive` datatypes of the arrays above will be matched to the data
types of the `model` output. Therefore, if your model returns `float32`,
these arrays will be stored as `float32`. The final, returned shap
values will also be returned as the datatype returned by the model.

These theoretical sizes can be calculated directly so that the user can
determine appropriate batch sizes for their machine:

``` python
# Combines our background data back into 1 DataFrame
ke.stratify_background_set(1)
(
    mask_matrix_size, 
    linear_target_size, 
    inner_model_eval_set_size
) = ke.get_theoretical_array_expansion_sizes(
    outer_batch_size=150,
    inner_batch_size=150,
    n_coalition_sizes=3,
    background_fold_to_use=None,
)

print(
  np.product(linear_target_size) + np.product(inner_model_eval_set_size)
)
```

    ## 92100

For the iris dataset, even if we sent the entire set (150 rows) through
as one batch, we only need 92100 elements stored in arrays. This is
manageable on most machines. However, this number ***grows extremely
quickly*** with the samples and number of columns. It is highly advised
to determine a good batch scheme before running this process.

Another way to determine optimal batch sizes is to use the function
`.get_theoretical_minimum_memory_requirements()`. This returns a list of
Gigabytes needed to build the arrays above:

``` python
# Combines our background data back into 1 DataFrame
(
    mask_matrix_GB, 
    linear_target_GB, 
    inner_model_eval_set_GB
) = ke.get_theoretical_minimum_memory_requirements(
    outer_batch_size=150,
    inner_batch_size=150,
    n_coalition_sizes=3,
    background_fold_to_use=None,
)

total_GB_needed = mask_matrix_GB + linear_target_GB + inner_model_eval_set_GB
print(f"We need {total_GB_needed} GB to calculate shap values with these batch sizes.")
```

    ## We need 0.000736856 GB to calculate shap values with these batch sizes.

### Specifying a Custom Linear Model

Any linear model available from sklearn.linear\_model can be used to
calculate the shap values. If you wish for some sparsity in the shap
values, you can use Lasso regression:

``` python
from sklearn.linear_model import Lasso

# Use our entire background set
ke.stratify_background_set(1)
sv_lasso = ke.calculate_shap_values(
  data, 
  background_fold_to_use=0,
  linear_model=Lasso(alpha=0.1),
  verbose=False
)

print(sv_lasso[0,:])
```

    ## [-0.         -0.33797832 -0.         -0.14634971  5.84333333]

The default model used is `sklearn.linear_model.LinearRegression`.

### Multiclass Outputs

If the model returns multiple outputs, the resulting shap values are
returned as an array of size (`rows`, `columns + 1`, `outputs`).
Therefore, to get the shap values for the effects on the second class,
you need to slice the resulting shap values using `shap_values[:,:,1]`.
Here is an example:

``` python
multi_features = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
multi_features.rename({"target": "species"}, inplace=True, axis=1)
target = multi_features.pop("species")

dtrain = lgb.Dataset(data=multi_features, label=target)
lgbmodel = lgb.train(
    params={"seed": 1, "objective": "multiclass", "num_class": 3, "verbose": -1},
    train_set=dtrain,
    num_boost_round=10
)
model = lgbmodel.predict

explainer_multi = fastshap.KernelExplainer(model, multi_features)
shap_values_multi = explainer_multi.calculate_shap_values(multi_features, verbose=False)

# To get the shap values for the second class:
print(shap_values_multi[:,:,1].shape)
```

    ## (150, 5)
