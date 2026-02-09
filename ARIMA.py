import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings
warnings.filterwarnings("ignore")
from normalize_clustered_mi import NormalizedClusteredMI

# 1) 构造示例数据
np.random.seed(0)
n = 300
dates = pd.date_range('2024-01-01', periods=n, freq='D')
y = 50 + 0.5*np.arange(n) + np.random.normal(0, 3, n)  # 带缓慢趋势
dow = [d.weekday() for d in dates]                      # 离散：星期几(0-6)
is_promo = np.random.choice([0,1], size=n, p=[0.85,0.15])  # 离散：是否促销

df = pd.DataFrame({'y': y, 'dow': dow, 'is_promo': is_promo}, index=dates)

# 2) 离散特征 one-hot（训练期 + 未来期保持同一列空间很重要）
X = pd.get_dummies(df[['dow','is_promo']].astype('category'), drop_first=True)

# print(X)


historical_len = 512

variables = ['Open',  'DayOfWeek', 'Promo', 'StateHoliday']

# df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv')
# df['ds'] = pd.to_datetime(df['ds'])
# # print(df)

df_new = pd.read_csv('./data/grocery_sales/train.csv')
# print(df_new)



df_sub = df_new[df_new['Store']==88].copy().reset_index(drop=True)

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
    df_sub[v] = category

df = df_sub

df['Sales'] = df['Sales'].astype(float)

for v in variables:
    print('Covariate: ', v)

    X = pd.get_dummies(df[[v]].astype('category'), drop_first=True)
    # print(X.head())
    # print(df.head())

    # 3) 拆分训练/测试
    train_end = int(len(df)/3*2)  # 留后30天做测试/演示
    y_tr, X_tr = df.loc[:train_end, 'Sales'], X.loc[:train_end]
    y_te, X_te = df.loc[train_end:, 'Sales'], X.loc[train_end:]

    # print("X_tr: ", X_tr)
    # print("X_te: ", X_te)

    p = d = q = range(1, 3)
    pdq = list(itertools.product(p, d, q))

    best_aic, best_order = 1e9, None
    for order in pdq:
        try:
            model = ARIMA(y_tr, order=order)
            res = model.fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, order
        except:
            continue
    # print("Best order:", best_order, "AIC:", best_aic)

    # 4) 拟合 ARIMAX（示例用 (p,d,q)=(2,1,1)，真实项目请做网格/AIC或pmdarima自动定阶）
    model = ARIMA(endog=y_tr, exog=X_tr, order=best_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit()
    # print(res.summary())

    # 5) 预测（必须提供测试/未来期的 exog）
    pred = res.get_forecast(steps=len(X_te), exog=X_te)
    mean = pred.predicted_mean
    ci = pred.conf_int()

    # 6) 简单评估
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    print("MAE =", mean_absolute_error(y_te, mean))
    print("MSE =", mean_squared_error(y_te, mean))