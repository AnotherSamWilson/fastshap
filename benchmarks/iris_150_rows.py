
from sklearn.datasets import load_iris
import pandas as pd
import lightgbm as lgb
import fastshap
import shap
from datetime import datetime as dt
import numpy as np

dataset_union_list = [1,2,3,4,5]
runs_per_union = 10


data = pd.concat(load_iris(as_frame=True,return_X_y=True),axis=1)
data.rename({"target": "species"}, inplace=True, axis=1)
target = data.pop("sepal length (cm)")
data = data.values
target = target.values

dtrain = lgb.Dataset(data=data, label=target)
lgbmodel = lgb.train(
    params={"seed": 1, "verbose": -1},
    train_set=dtrain,
    num_boost_round=100
)
model = lgbmodel.predict
preds = model(data)


fastshap_ke = fastshap.KernelExplainer(model, data)
shap_ke = shap.KernelExplainer(model, data)

fastshap_times = np.zeros((runs_per_union,len(dataset_union_list)))
shap_times = np.zeros((runs_per_union,len(dataset_union_list)))

for unions in dataset_union_list:
    union_ind = dataset_union_list.index(unions)
    newdat = np.tile(data, (unions, 1))
    for run in range(runs_per_union):
        s = dt.now()
        fastshap_sv = fastshap_ke.calculate_shap_values(newdat)
        fastshap_times[run, union_ind] = (dt.now() - s).total_seconds()

        s = dt.now()
        shap_sv = shap_ke.shap_values(newdat)
        shap_times[run, union_ind] = (dt.now() - s).total_seconds()


import matplotlib.pyplot as plt
plot_df = pd.DataFrame({
    "rows": [data.shape[0] * dsu for dsu in dataset_union_list],
    "fastshap": fastshap_times.mean(0),
    "shap": shap_times.mean(0),
    "fastshap_times_std": fastshap_times.std(0),
    "shap_times_std": shap_times.std(0),
    "Relative Difference Per Row": shap_times.mean(0) / fastshap_times.mean(0)
})
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
with plt.style.context("seaborn-deep"):
    plt.errorbar(
        data=plot_df, x="rows", y="fastshap",
        yerr="fastshap_times_std",
        # color="red"
    )
    plt.errorbar(
        data=plot_df, x="rows", y="shap",
        yerr="shap_times_std",
        # color="blue"
    )
    ax.set_xticks(plot_df["rows"])
    plt.legend(title="Package")
    plt.ylabel("Seconds")
    plt.xlabel("Rows in dataset (4 columns)")
    plt.title("Calculation Time on Iris dataset \n fastshap vs shap KernelExplainers")
    plt.savefig("benchmarks/iris_benchmark_time.png")

plot_df.to_csv("benchmarks/iris_benchmark_time_df.csv")