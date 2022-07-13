
from sklearn.datasets import fetch_california_housing
import lightgbm as lgb
import fastshap
import shap
import pandas as pd
import numpy as np
from shap.maskers import Independent
from datetime import datetime as dt
background_data, target = fetch_california_housing(as_frame=False, return_X_y=True)
shap_data = background_data[:2000, :]

stratification_sizes = [500, 400, 300, 200, 100, 50, 25]

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
shap_times = []
row_list = []
for ss in stratification_sizes:
    ss = stratification_sizes[2]
    # Stratify the dataset into 200 (almost) equal sized sets.
    fastshap_ke.stratify_background_set(ss)
    background_rows = fastshap_ke.background_data[0].shape[0]
    row_list.append(background_rows)

    s = dt.now()
    fastshap_sv = fastshap_ke.calculate_shap_values(
        shap_data,
        outer_batch_size=2000,
        inner_batch_size=2000,
        background_fold_to_use=0
    )
    fastshap_times.append((dt.now() - s).total_seconds())

    # Use the same background dataset as fastshap
    # Shap sets the background set to 100 subsample unless a masker is
    # specifically created
    shap_ke = shap.KernelExplainer(
        model,
        data=fastshap_ke.background_data[0]
    )

    s = dt.now()
    shap_sv = shap_ke.shap_values(shap_data, nsamples=184)
    shap_times.append((dt.now() - s).total_seconds())

fastshap_sv[:,:-1] / shap_sv



import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
fig, ax = plt.subplots()
with plt.style.context("seaborn-deep"):
    plt.plot(row_list, fastshap_times, label="fastshap")
    plt.plot(row_list, shap_times, label="shap")
    plt.legend(title="Package")
    ax.set_xscale("log")
    plt.minorticks_off()
    ax.set_xticks(row_list)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.ylabel("Seconds")
    plt.xlabel("Background Rows Used")
    plt.title("Calculation Time on 2000 Rows of California Housing Dataset \n fastshap vs shap KernelExplainers")
    plt.savefig("benchmarks/cali_benchmark_time.png")

plot_df = pd.DataFrame({
    "rows": row_list,
    "fastshap": fastshap_times,
    "shap": shap_times
})
plot_df["Relative Difference"] = np.array(shap_times) / np.array(fastshap_times)
plot_df.to_csv("benchmarks/cali_benchmark_time_df.csv")