
from sklearn.datasets import load_boston, load_iris
import pandas as pd
import numpy as np
import lightgbm as lgb
from fastshap.KernelExplainer import KernelExplainer
from fastshap.plotting import plot_variable_effect_on_output
from datetime import datetime as dt


data = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
data.rename({"target": "species"}, inplace=True, axis=1)
data["species"] = data["species"].map({0: "a", 1: "b", 2: "c"}).astype("category")
target = data.pop("species")

dtrain = lgb.Dataset(data=data, label=target.cat.codes)
lgbmodel = lgb.train(
    params={"objective": "multiclass", "num_classes": 3, "seed": 1, "verbose": -1},
    train_set=dtrain,
    num_boost_round=10
)
model = lgbmodel.predict
preds = model(data)


def test_pd_stratifying():
    ke = KernelExplainer(model, data)
    sv = ke.calculate_shap_values(
        data,
        outer_batch_size=100,
        inner_batch_size=100,
        n_coalition_sizes=2
    )
    assert np.all(abs(preds - sv.sum(1)) < 0.00001)
    assert sv.shape == (150, 5, 3)

    aes = ke.get_theoretical_array_expansion_sizes(
        data=data,
        outer_batch_size=100,
        inner_batch_size=100,
        n_coalition_sizes=2
    )
    assert aes == ((14, 4), (14, 100, 3), (75000, 4))

    aegb = ke.get_theoretical_minimum_memory_requirements(
        data=data,
        outer_batch_size=100,
        inner_batch_size=100,
        n_coalition_sizes=2
    )
    assert all(a < 0.01 for a in aegb)

    plot_variable_effect_on_output(
        sv, data,
        variable="sepal width (cm)",
        interaction_variable="auto",
        class_labels=["a", "b", "c"]
    )

    # Stratify and re-compute
    ke.stratify_background_set(5)
    sv = ke.calculate_shap_values(
        data,
        outer_batch_size=100000,
        inner_batch_size=1,
        n_coalition_sizes=2
    )
    assert np.all(abs(preds - sv.sum(1)) < 0.00001)
    assert sv.shape == (150, 5, 3)

    ke.stratify_background_set(1)
    sv = ke.calculate_shap_values(
        data,
        outer_batch_size=10,
        inner_batch_size=10,
        n_coalition_sizes=2
    )
    assert np.all(abs(preds - sv.sum(1)) < 0.00001)
    assert sv.shape == (150, 5, 3)

def test_pd_single_sample():
    newdat = data.iloc[[0],:]
    ke = KernelExplainer(model, data)
    sv = ke.calculate_shap_values(
        newdat,
        outer_batch_size=1,
        inner_batch_size=1
    )
    assert sv.shape == (1,5,3)