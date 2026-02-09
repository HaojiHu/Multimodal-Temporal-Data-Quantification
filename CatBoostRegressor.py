import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mixed
from ross_mi import *
from normalize_clustered_mi import NormalizedClusteredMI
import mixture_mi
from sklearn.metrics import ndcg_score
from variants_for_ablation_study import NoMixtureNoClusteredMI, NoClusteredMI

import pandas as pd
from DeepAD import *
from TimeSeriesFoundationModel import *

# device = "cuda" if torch.cuda.is_available() else "cpu"


df_new = pd.read_csv('./data/grocery_sales/train.csv')
# print(df_new)



CatBoostRegressor_proposed_ndcg_list = []
CatBoostRegressor_mixtured_ndcg_list = []
CatBoostRegressor_ross_ndcg_list = []
CatBoostRegressor_noclustering_ndcg_list = []

CatBoostRegressor_proposed_win = 0
CatBoostRegressor_mixtured_win = 0
CatBoostRegressor_ross_win = 0
CatBoostRegressor_noclustering_win = 0

DeepAR_proposed_ndcg_list = []
DeepAR_mixtured_ndcg_list = []
DeepAR_ross_ndcg_list = []
DeepAR_noclustering_ndcg_list = []

DeepAR_proposed_win = 0
DeepAR_mixtured_win = 0
DeepAR_ross_win = 0
DeepAR_noclustering_win = 0

FM_proposed_ndcg_list = []
FM_mixtured_ndcg_list = []
FM_ross_ndcg_list = []
FM_noclustering_ndcg_list = []

FM_proposed_win = 0
FM_mixtured_win = 0
FM_ross_win = 0
FM_noclustering_win = 0

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

def get_categoryset(category):
    cat = set(category)
    return cat

def normalized_negativity(mi_list):
    min_mi = min(mi_list)
    normalized_mi = []
    for mi in mi_list:
        normalized_mi.append((mi - min_mi))
    return normalized_mi


for Store in range(11, 16):

    print("*****Store: ", Store)

    variables = ['Open', 'DayOfWeek', 'Promo', 'StateHoliday', "SchoolHoliday"]

    df_sub = df_new[df_new['Store'] == Store].copy().reset_index(drop=True)

    print("Num. instances: ", len(df_sub))
    # df_sub.to_csv("data.csv", index=False, encoding="utf-8")
    #
    historical_len = int(len(df_sub)/5*4)



    sales = df_sub['Sales'].tolist()

    proposed_mi = []

    for v in variables:
        category = cast_string(df_sub[v].tolist())
        category = category2id(category)
        num = len(get_categoryset(category))

        mi = NormalizedClusteredMI(category[:historical_len], sales[:historical_len], 4)
        # mi = mixture_mi.NMI(category[:historical_len], sales[:historical_len])
        print(v, ' proposed mi: ', mi)
        proposed_mi.append(mi)

    mixed_ksg = []

    for v in variables:
        category = cast_string(df_sub[v].tolist())

        category = category2id(category)

        x = np.asarray(category)
        y = np.asarray(sales)

        mi = mixed.Mixed_KSG(x[:historical_len], y[:historical_len])
        print(v, 'mixture mi: ', mi)
        mixed_ksg.append(mi)

        # mi = mixture_mi_mao.NMI(category, sales)
        # print('MI of ' + v + ': ', mi)

    ross_res =[]

    for v in variables:
        category = cast_string(df_sub[v].tolist())

        category = category2id(category)

        mi, _ = discrete_continuous_info(category[:historical_len], [sales[:historical_len]])
        print(v, 'ross mi: ', mi)
        ross_res.append(mi)

    noclustering_res =[]

    for v in variables:
        category = cast_string(df_sub[v].tolist())

        category = category2id(category)

        mi = NoClusteredMI(category[:historical_len], sales[:historical_len])
        print(v, 'No clustering mi: ', mi)
        noclustering_res.append(mi)

    # sales = df_sub['Sales'].tolist()
    # print(len(sales))

    for v in variables:
        category = cast_string(df_sub[v].tolist())
        # category = category2id(category)
        df_sub[v] = category

    df = df_sub
    df = df.drop(columns=["Date","Store",])

    df['Sales'] = df['Sales'].astype(float)

    lag_days = [1, 2, 3, 7]

    def create_lag_features(df, lag_days):
        for lag in lag_days:
            df[f"lag_{lag}"] = df["Sales"].shift(lag)
        return df.dropna()

    print("head: ", df.head())
    df = create_lag_features(df, lag_days)



    # print(df)

    def get_lef(v, variblelist):
        left = []
        for var in variblelist:
            if var!=v:
                left.append(var)
        return left

    def converted2id(char_list):
        id_list = []
        char2id = {}
        for v in char_list:
            if v not in char2id:
                char2id[v] = float(len(char2id))
            id_list.append(char2id[v])

        return np.array(id_list), len(char2id)

    CatBoostRegressor_forecast_res = []
    DeepAR_forecast_res = []
    FM_forecast_res = []
    Noclustering_forecast_res = []

    for v in variables:
        print('##########################################')
        print('Covariate: ', v)

        left = get_lef(v, variables)

        train_end = int(historical_len)
        train = df.iloc[:train_end]
        test = df.iloc[train_end:]

        ### CatBoostRegressor experiments

        X_train = train.drop(columns=left+["Sales"])
        # X_train['lag_1'] = df[:train_end]['lag_1']
        for lag in lag_days:
            X_train[f"lag_{lag}"] = df[:train_end][f"lag_{lag}"]
        y_train = train['Sales']
        X_test = test.drop(columns=left+["Sales"])
        # X_test['lag_1'] = df[train_end:]['lag_1']
        for lag in lag_days:
            X_test[f"lag_{lag}"] = df[train_end:][f"lag_{lag}"]
        y_test = test['Sales']
        cat_features = [v]
        # print("X_train: ", X_train)
        # print("y_train: ", y_train)
        # print("X_test: ", X_test)
        # print("y_test: ", y_test)
        #
        # print("X_train.shape: ", X_train.shape)
        # print("y_train.shape: ", y_train.shape)
        # print("y_train.head(): ", y_train.head())
        # print("y_train.isna().sum(): ", y_train.isna().sum())

        model = CatBoostRegressor(
            depth=6,
            learning_rate=0.02,
            iterations=2000,
            loss_function="MAE",
            early_stopping_rounds=200,
            verbose=100
        )

        model.fit(X_train, y_train,
                  cat_features=cat_features,
                  eval_set=(X_test, y_test))


        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("CatBoostRegressor MAE =", mae)
        print("CatBoostRegressor MSE =", mse)

        CatBoostRegressor_forecast_res.append(1 / mae)

        ### DeepAR experiment

        context_len = 256
        future_steps = len(y_test)

        epochs = 80
        lr = 0.5

        y = np.asarray(y_train.to_list())

        dynamic_feat_full, cat_cardinality = converted2id(X_train[v].tolist()+X_test[v].tolist())
        dynamic_features, _ = converted2id(X_train[v].tolist())


        model = DeepAR(
            context_len=context_len,
            future_steps=future_steps,
            num_dyn_feat=1,
            cat_cardinality=cat_cardinality+5,
            cat_emb_dim=4,
            hidden_size=64,
        ).to(device)

        model_train(model, y, 1, dynamic_features, context_len, epochs, lr)

        y_pred = forecast(model, y, 1, dynamic_feat_full, context_len, future_steps)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print("DeepAR MAE =", mae)
        print("DeepAR MSE =", mse)

        DeepAR_forecast_res.append(1 / mae)


        ### Time series foundation model

        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=128,
                horizon_len=24,
                # input_patch_len=16,
                # output_patch_len=32,
                num_layers=20,
                model_dims=1280,
                backend=timesfm_backend,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )

        mae = FM_forecasting(model, df, v,  "Sales", train_end)

        FM_forecast_res.append(1 / mae)

    mixed_ksg = normalized_negativity(mixed_ksg)
    ross_res = normalized_negativity(ross_res)

    CatBoostRegressor_forecast_res = np.array([CatBoostRegressor_forecast_res])
    DeepAR_forecast_res = np.array([DeepAR_forecast_res])
    FM_forecast_res = np.array([FM_forecast_res])
    proposed_mi = np.array([proposed_mi])
    mixed_ksg = np.array([mixed_ksg])
    ross_res = np.array([ross_res])
    noclustering_res = np.array([noclustering_res])

    CatBoostRegressor_proposed_ndcg = ndcg_score(proposed_mi, CatBoostRegressor_forecast_res)
    CatBoostRegressor_mixtured_ndcg = ndcg_score(mixed_ksg, CatBoostRegressor_forecast_res)
    CatBoostRegressor_ross_ndcg = ndcg_score(ross_res, CatBoostRegressor_forecast_res)
    CatBoostRegressor_noclustering_ndcg = ndcg_score(noclustering_res, CatBoostRegressor_forecast_res)

    print("CatBoostRegressor Proposed NDCG =", CatBoostRegressor_proposed_ndcg)
    print("CatBoostRegressor Mixtured NDCG =", CatBoostRegressor_mixtured_ndcg)
    print("CatBoostRegressor Ross method NDCG =", CatBoostRegressor_ross_ndcg)
    print("CatBoostRegressor noclustering NDCG =", CatBoostRegressor_noclustering_ndcg)

    CatBoostRegressor_proposed_ndcg_list.append(CatBoostRegressor_proposed_ndcg)
    CatBoostRegressor_mixtured_ndcg_list.append(CatBoostRegressor_mixtured_ndcg)
    CatBoostRegressor_ross_ndcg_list.append(CatBoostRegressor_ross_ndcg)
    CatBoostRegressor_noclustering_ndcg_list.append(CatBoostRegressor_noclustering_ndcg)

    CatBoostRegressor_max_ndcg = max(CatBoostRegressor_proposed_ndcg, CatBoostRegressor_mixtured_ndcg, CatBoostRegressor_ross_ndcg, CatBoostRegressor_noclustering_ndcg)
    if CatBoostRegressor_max_ndcg == CatBoostRegressor_proposed_ndcg:
        CatBoostRegressor_proposed_win += 1
    elif CatBoostRegressor_max_ndcg == CatBoostRegressor_mixtured_ndcg:
        CatBoostRegressor_mixtured_win += 1
    elif CatBoostRegressor_max_ndcg == CatBoostRegressor_ross_ndcg:
        CatBoostRegressor_ross_win += 1
    elif CatBoostRegressor_max_ndcg == CatBoostRegressor_noclustering_ndcg:
        CatBoostRegressor_noclustering_win += 1

    DeepAR_proposed_ndcg = ndcg_score(proposed_mi, DeepAR_forecast_res)
    DeepAR_mixtured_ndcg = ndcg_score(mixed_ksg, DeepAR_forecast_res)
    DeepAR_ross_ndcg = ndcg_score(ross_res, DeepAR_forecast_res)
    DeepAR_noclustering_ndcg = ndcg_score(noclustering_res, DeepAR_forecast_res)

    print("DeepAR Proposed NDCG =", DeepAR_proposed_ndcg)
    print("DeepAR Mixtured NDCG =", DeepAR_mixtured_ndcg)
    print("DeepAR Ross method NDCG =", DeepAR_ross_ndcg)
    print("DeepAR noclustering NDCG =", DeepAR_noclustering_ndcg)

    DeepAR_proposed_ndcg_list.append(DeepAR_proposed_ndcg)
    DeepAR_mixtured_ndcg_list.append(DeepAR_mixtured_ndcg)
    DeepAR_ross_ndcg_list.append(DeepAR_ross_ndcg)
    DeepAR_noclustering_ndcg_list.append(DeepAR_noclustering_ndcg)

    DeepAR_max_ndcg = max(DeepAR_proposed_ndcg, DeepAR_mixtured_ndcg, DeepAR_ross_ndcg, DeepAR_noclustering_ndcg)
    if DeepAR_max_ndcg == DeepAR_proposed_ndcg:
        DeepAR_proposed_win += 1
    elif DeepAR_max_ndcg == DeepAR_mixtured_ndcg:
        DeepAR_mixtured_win += 1
    elif DeepAR_max_ndcg == DeepAR_ross_ndcg:
        DeepAR_ross_win += 1
    elif DeepAR_max_ndcg == DeepAR_noclustering_ndcg:
        DeepAR_noclustering_win += 1

    FM_proposed_ndcg = ndcg_score(proposed_mi, FM_forecast_res)
    FM_mixtured_ndcg = ndcg_score(mixed_ksg, FM_forecast_res)
    FM_ross_ndcg = ndcg_score(ross_res, FM_forecast_res)
    FM_noclustering_ndcg = ndcg_score(noclustering_res, FM_forecast_res)

    print("FM Proposed NDCG =", FM_proposed_ndcg)
    print("FM Mixtured NDCG =", FM_mixtured_ndcg)
    print("FM Ross method NDCG =", FM_ross_ndcg)
    print("FM noclustering NDCG =", FM_noclustering_ndcg)

    FM_proposed_ndcg_list.append(FM_proposed_ndcg)
    FM_mixtured_ndcg_list.append(FM_mixtured_ndcg)
    FM_ross_ndcg_list.append(FM_ross_ndcg)
    FM_noclustering_ndcg_list.append(FM_noclustering_ndcg)

    FM_max_ndcg = max(FM_proposed_ndcg, FM_mixtured_ndcg, FM_ross_ndcg, FM_noclustering_ndcg)
    if FM_max_ndcg == FM_proposed_ndcg:
        FM_proposed_win += 1
    elif FM_max_ndcg == FM_mixtured_ndcg:
        FM_mixtured_win += 1
    elif FM_max_ndcg == FM_ross_ndcg:
        FM_ross_win += 1
    elif FM_max_ndcg == FM_noclustering_ndcg:
        FM_noclustering_win += 1

print("-------------CatBoostRegressor Results-------------")

print("Mean NDCG of proposed method: ", np.mean(CatBoostRegressor_proposed_ndcg_list))
print("Mean NDCG of mixture method: ", np.mean(CatBoostRegressor_mixtured_ndcg_list))
print("Mean NDCG of ross method: ", np.mean(CatBoostRegressor_ross_ndcg_list))
print("Mean NDCG of noclustering method: ", np.mean(CatBoostRegressor_noclustering_ndcg_list))

print("Proposed method wins: ", CatBoostRegressor_proposed_win)
print("Mixture method wins: ", CatBoostRegressor_mixtured_win)
print("Ross method wins: ", CatBoostRegressor_ross_win)
print("No clustering wins: ", CatBoostRegressor_noclustering_win)

print("-------------DeepAR Results-------------")

print("Mean NDCG of proposed method: ", np.mean(DeepAR_proposed_ndcg_list))
print("Mean NDCG of mixture method: ", np.mean(DeepAR_mixtured_ndcg_list))
print("Mean NDCG of ross method: ", np.mean(DeepAR_ross_ndcg_list))
print("Mean NDCG of noclustering method: ", np.mean(DeepAR_noclustering_ndcg_list))

print("Proposed method wins: ", DeepAR_proposed_win)
print("Mixture method wins: ", DeepAR_mixtured_win)
print("Ross method wins: ", DeepAR_ross_win)
print("No clustering wins: ", DeepAR_noclustering_win)

print("-------------Time Series Foundation Model Results-------------")

print("Mean NDCG of proposed method: ", np.mean(FM_proposed_ndcg_list))
print("Mean NDCG of mixture method: ", np.mean(FM_mixtured_ndcg_list))
print("Mean NDCG of ross method: ", np.mean(FM_ross_ndcg_list))
print("Mean NDCG of noclustering method: ", np.mean(FM_noclustering_ndcg_list))

print("Proposed method wins: ", FM_proposed_win)
print("Mixture method wins: ", FM_mixtured_win)
print("Ross method wins: ", FM_ross_win)
print("No clustering wins: ", FM_noclustering_win)