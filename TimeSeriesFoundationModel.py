import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

import pandas as pd
import numpy as np
from collections import defaultdict

import time
import timesfm

timesfm_backend = "cpu"  # @param


from jax._src import config
config.update(
    "jax_platforms", {"cpu": "cpu", "gpu": "cuda", "tpu": ""}[timesfm_backend]
)

historical_len = 128
horizon_len = 24

def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

def get_batched_data_new(
        sub_df: pd.DataFrame,
        batch_size: int = 128,
        context_len: int = historical_len,
        horizon_len: int = horizon_len,
):
    examples = defaultdict(list)

    num_examples = 0
    # for store in [Store]:
    #     sub_df = df_new[df_new["Store"] == store].copy().reset_index(drop=True)
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
        num_examples += 1
        examples["Store"].append(1)
        examples["inputs"].append(sub_df["Sales"][start:(context_end := start + context_len)].tolist())
        examples["Customers"].append(sub_df["Customers"][start:context_end + horizon_len].tolist())
        examples["DayOfWeek"].append(sub_df["DayOfWeek"][start:context_end + horizon_len].tolist())
        examples["Promo"].append(sub_df["Promo"][start:context_end + horizon_len].tolist())
        examples["StateHoliday"].append(sub_df["StateHoliday"][start:context_end + horizon_len].tolist())
        examples["SchoolHoliday"].append(sub_df["SchoolHoliday"][start:context_end + horizon_len].tolist())
        examples["Open"].append(sub_df["Open"][start:context_end + horizon_len].tolist())
        examples["outputs"].append(sub_df["Sales"][context_end:(context_end + horizon_len)].tolist())

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size): ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn

def get_batched_data_M5(y_df: pd.DataFrame,
        sub_df: pd.DataFrame,
        batch_size: int = 128,
        context_len: int = historical_len,
        horizon_len: int = horizon_len,
):
    examples = defaultdict(list)

    num_examples = 0
    # for store in [Store]:
    #     sub_df = df_new[df_new["Store"] == store].copy().reset_index(drop=True)
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
        num_examples += 1
        examples["Store"].append(1)
        examples["inputs"].append(y_df[start:(context_end := start + context_len)].tolist())
        examples["wm_yr_wk"].append(sub_df["wm_yr_wk"][start:context_end + horizon_len].tolist())
        examples["weekday"].append(sub_df["weekday"][start:context_end + horizon_len].tolist())
        examples["event_name_1"].append(sub_df["event_name_1"][start:context_end + horizon_len].tolist())
        examples["event_type_1"].append(sub_df["event_type_1"][start:context_end + horizon_len].tolist())
        examples["snap_CA"].append(sub_df["snap_CA"][start:context_end + horizon_len].tolist())
        examples["event_name_2"].append(sub_df["event_name_2"][start:context_end + horizon_len].tolist())
        examples["event_type_2"].append(sub_df["event_type_2"][start:context_end + horizon_len].tolist())
        examples["snap_TX"].append(sub_df["snap_TX"][start:context_end + horizon_len].tolist())
        examples["snap_WI"].append(sub_df["snap_WI"][start:context_end + horizon_len].tolist())
        examples["outputs"].append(y_df[context_end:(context_end + horizon_len)].tolist())

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size): ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn

def FM_forecasting(model, sub_df, variable,  dataset, train_len, y_df=None):

    batch_size = 512

    if dataset == "Sales":
        input_data = get_batched_data_new(sub_df, batch_size = batch_size)
    elif dataset == "M5":

        input_data = get_batched_data_M5(y_df, sub_df, batch_size = batch_size)

    metrics = defaultdict(list)

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
              "Store": example["Store"]
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

    return np.mean(metrics["eval_mae_xreg_timesfm"][int(train_len/len(sub_df) * (len(sub_df) / batch_size)):])