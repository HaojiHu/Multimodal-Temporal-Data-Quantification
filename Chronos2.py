
import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline


def Chronos2_forecasting(pipeline, sales, event, dataset, total_len, historical_len):

    df_data = pd.DataFrame({
        'sales': sales,
        'event': event
    })

    df_data['timestamp'] = pd.date_range(
        start='2024-01-01',
        periods=len(df_data),
        freq='D'   # hourly
    )

    df_data['id'] = 1

    context_df = df_data.iloc[:historical_len]
    test_df = df_data.iloc[historical_len:]
    future_df = test_df.drop(columns="sales")


    # Generate predictions with covariates
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=len(sales)-historical_len,  # Number of steps to forecast
        quantile_levels=[0.1, 0.5, 0.9],  # Quantiles for probabilistic forecast
        id_column="id",  # Column identifying different time series
        timestamp_column="timestamp",  # Column with datetime information
        target="sales",  # Column(s) with time series values to predict
    )

    return pred_df['predictions'].tolist()