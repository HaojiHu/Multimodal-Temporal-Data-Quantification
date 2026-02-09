import timesfm
import mixed
from ross_mi import *
from normalize_clustered_mi import NormalizedClusteredMI


import pandas as pd
import numpy as np
from collections import defaultdict

Store = 88

historical_len = 512

variables = ['Open',  'DayOfWeek', 'Promo', 'StateHoliday']

# df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv')
# df['ds'] = pd.to_datetime(df['ds'])
# # print(df)

df_new = pd.read_csv('./data/grocery_sales/train.csv')
# print(df_new)

df_sub = df_new[df_new['Store']==Store].copy().reset_index(drop=True)
print("Num. instances: ", len(df_sub))

def cast_string(value):
    char_list = []
    for v in value:
        char_list.append(str(v))
    return char_list

def category2id(category):
    cat = set(category)
    cat2id = {}
    id = 1
    for c in cat:

        cat2id[c] = id
        id += 1

    cat_id = []
    for c in category:
        cat_id.append(cat2id[c])
    return np.array(cat_id)



sales = df_sub['Sales'].tolist()


for v in variables:
    category = cast_string(df_sub[v].tolist())
    category = category2id(category)

    mi = NormalizedClusteredMI(category[:historical_len], sales[:historical_len], 4)
    print(v, ' proposed mi: ', mi)

for v in variables:
    category = cast_string(df_sub[v].tolist())

    category = category2id(category)

    x = np.asarray(category)
    y = np.asarray(sales)

    mi = mixed.Mixed_KSG(x[:historical_len], y[:historical_len])
    print(v, 'mixture mi: ', mi)


    # mi = mixture_mi_mao.NMI(category, sales)
    # print('MI of ' + v + ': ', mi)


for v in variables:
    category = cast_string(df_sub[v].tolist())

    category = category2id(category)

    mi, _ = discrete_continuous_info(category[:historical_len], [sales[:historical_len]])
    print(v, 'ross mi: ', mi)


#
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'


import time
import timesfm

timesfm_backend = "cpu"  # @param

from jax._src import config
config.update(
    "jax_platforms", {"cpu": "cpu", "gpu": "cuda", "tpu": ""}[timesfm_backend]
)

model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          context_len=512,
          horizon_len=128,
          input_patch_len=32,
          output_patch_len=128,
          num_layers=20,
          model_dims=1280,
          backend=timesfm_backend,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
  )

def get_batched_data_new(
        batch_size: int = 16,
        context_len: int = historical_len,
        horizon_len: int = 64,
):
    examples = defaultdict(list)

    num_examples = 0
    for store in [Store]:
        sub_df = df_new[df_new["Store"] == store].copy().reset_index(drop=True)
        for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
            num_examples += 1
            examples["Store"].append(store)
            examples["inputs"].append(sub_df["Sales"][start:(context_end := start + context_len)].tolist())
            examples["Customers"].append(sub_df["Customers"][start:context_end + horizon_len].tolist())
            examples["DayOfWeek"].append(sub_df["DayOfWeek"][start:context_end + horizon_len].tolist())
            examples["Promo"].append(sub_df["Promo"][start:context_end + horizon_len].tolist())
            examples["StateHoliday"].append(sub_df["StateHoliday"][start:context_end + horizon_len].tolist())
            examples["Open"].append(sub_df["Open"][start:context_end + horizon_len].tolist())
            examples["outputs"].append(sub_df["Sales"][context_end:(context_end + horizon_len)].tolist())

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size): ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn

# Define metrics
def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

for variable in variables:
    print("-----------------------------------------")
    print(f"covariate variable: {variable}")

# Data pipelining


    # Benchmark
    batch_size = 128
    context_len = 512
    horizon_len = 64
    # input_data = get_batched_data_fn(batch_size = 128)
    input_data = get_batched_data_new(batch_size = 128)
    metrics = defaultdict(list)

    print(input_data)

    for i, example in enumerate(input_data()):
      raw_forecast, _ = model.forecast(
          inputs=example["inputs"], freq=[0] * len(example["inputs"])
      )
      start_time = time.time()
      # Forecast with covariates
      # Output: new forecast, forecast by the xreg
      cov_forecast, ols_forecast = model.forecast_with_covariates(
          inputs=example["inputs"],
          dynamic_numerical_covariates={
              # "gen_forecast": example["gen_forecast"],
              # "Customers": example["Customers"],
          },
          dynamic_categorical_covariates={
              # "week_day": example["week_day"],
              # "DayOfWeek": example["DayOfWeek"],
              # "Promo": example["Promo"],
              # "StateHoliday": example["StateHoliday"],
              variable: example[variable],
          },
          static_numerical_covariates={},
          static_categorical_covariates={
              # "country": example["country"]
              # "Store": example["Store"]
          },
          freq=[0] * len(example["inputs"]),
          xreg_mode="xreg + timesfm",              # default
          ridge=0.0,
          force_on_cpu=False,
          normalize_xreg_target_per_input=True,    # default
      )
      print(
          f"\rFinished batch {i} linear in {time.time() - start_time} seconds",
          end="",
      )
      metrics["eval_mae_timesfm"].extend(
          mae(raw_forecast[:, :horizon_len], example["outputs"])
      )
      metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, example["outputs"]))
      # metrics["eval_mae_xreg"].extend(mae(ols_forecast, example["outputs"]))
      metrics["eval_mse_timesfm"].extend(
          mse(raw_forecast[:, :horizon_len], example["outputs"])
      )
      metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, example["outputs"]))
      # metrics["eval_mse_xreg"].extend(mse(ols_forecast, example["outputs"]))

    print()

    for k, v in metrics.items():
      print(f"{k}: {np.mean(v)}")


