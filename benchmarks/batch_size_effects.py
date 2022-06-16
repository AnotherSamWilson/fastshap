
from sklearn.datasets import fetch_california_housing
import lightgbm as lgb
import fastshap
import shap
import pandas as pd
import numpy as np
from datetime import datetime as dt
background_data, target = fetch_california_housing(as_frame=False, return_X_y=True)
shap_data = background_data[:2000, :]

outer_batch_sizes = [50, 100, 200, 500, 1000, 2000]

# Build our model
dtrain = lgb.Dataset(data=background_data, label=target)
lgbmodel = lgb.train(
    params={"seed": 1, "verbose": -1},
    train_set=dtrain,
    num_boost_round=100
)
model = lgbmodel.predict

# Create fastshap kernel
fastshap_ke = fastshap.KernelExplainer(model, background_data)

fastshap_times = []
batch_size_list = []
for obs in outer_batch_sizes:
    # Stratify the dataset into X (almost) equal sized sets.
    fastshap_ke.stratify_background_set(20)
    batch_size_list.append(obs)
    s = dt.now()
    fastshap_sv = fastshap_ke.calculate_shap_values(
        shap_data,
        outer_batch_size=obs,
        inner_batch_size=None,
        background_fold_to_use=0
    )
    fastshap_times.append((dt.now() - s).total_seconds())


import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
fig, ax = plt.subplots()
with plt.style.context("seaborn-deep"):
    plt.plot(batch_size_list, fastshap_times, label="fastshap")
    # plt.legend(title="Package")
    ax.set_xscale("log")
    plt.minorticks_off()
    ax.set_xticks(batch_size_list)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.ylabel("Seconds")
    plt.xlabel("Outer Batch Size Used")
    plt.title("Calculation Time of California Housing Dataset \n Using Different Outer Batch Sizes")
    plt.savefig("benchmarks/batch_size_times.png")

plot_df = pd.DataFrame({
    "Outer Batch Size": batch_size_list,
    "fastshap": fastshap_times
})
plot_df["Relative Difference"] = np.array(shap_times) / np.array(fastshap_times)
plot_df.to_csv("benchmarks/cali_benchmark_time_df.csv")