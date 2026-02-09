import pandas as pd
from DeepAD import *


T = 300
np.random.seed(42)

# 序列
y = np.sin(np.arange(T)/10) + np.random.normal(scale=0.1, size=T)

# 离散特征：店铺编号 0~1
cat_id = 1

context_len = 30
future_steps = 30

# 动态特征：是否周末
dates = pd.date_range("2023-01-01", periods=T)
is_weekend = (dates.weekday >= 5).astype("float32")


device = "cuda" if torch.cuda.is_available() else "cpu"


model = DeepAR(
    context_len=context_len,
    future_steps=future_steps,
    num_dyn_feat=1,
    cat_cardinality=2,
    cat_emb_dim=4,
    hidden_size=40,
).to(device)


model_train(model, y, cat_id, is_weekend, context_len, 10)



dates_full = pd.date_range("2023-01-01", periods=T+future_steps)
is_weekend_full = (dates_full.weekday >= 5).astype("float32")



future_preds =  forecast(model, y, cat_id, is_weekend_full, context_len, future_steps)

print(future_preds)