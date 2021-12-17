
from sklearn.datasets import load_boston, load_iris
import pandas as pd
import numpy as np
import lightgbm as lgb
import fastshap
from datetime import datetime as dt


data = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
data.rename({"target": "species"}, inplace=True, axis=1)
data["species"] = data["species"].astype("category")
target = data.pop("sepal length (cm)")

data = data.values
target = target.values

dtrain = lgb.Dataset(data=data, label=target)
lgbmodel = lgb.train(
    params={"seed": 1, "verbose": -1},
    train_set=dtrain,
    num_boost_round=10
)
model = lgbmodel.predict
preds = model(data)


def test_np_stratifying():
    ke = fastshap.KernelExplainer(model, data)
    sv = ke.calculate_shap_values(
        data,
        outer_batch_size=100,
        inner_batch_size=100,
        n_coalition_sizes=2
    )
    assert all(abs(preds - sv.sum(1)) < 0.00001)
    assert sv.shape == (150, 5)

    # Stratify and re-compute
    ke.stratify_background_set(5)
    sv = ke.calculate_shap_values(
        data,
        outer_batch_size=100000,
        inner_batch_size=1,
        n_coalition_sizes=2
    )
    assert all(abs(preds - sv.sum(1)) < 0.00001)
    assert sv.shape == (150, 5)

    ke.stratify_background_set(1)
    sv = ke.calculate_shap_values(
        data,
        outer_batch_size=10,
        inner_batch_size=10,
        n_coalition_sizes=2
    )
    assert all(abs(preds - sv.sum(1)) < 0.00001)
    assert sv.shape == (150, 5)

def test_np_single_sample():
    newdat = data[0,:].reshape(1,4)
    ke = fastshap.KernelExplainer(model, data)
    sv = ke.calculate_shap_values(newdat, 1, 1)
    assert abs(sv.sum(1)[0] - preds[0]) < 0.00001
    assert sv.shape == (1, 5)