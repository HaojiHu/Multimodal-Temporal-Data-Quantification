import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load data
data1 = pd.read_csv('./data/traffic/51_6_7.csv', header=None)

# data1.plot()
# plt.show()
#
# plot_acf(data1)
# plt.show()

data2 = pd.read_csv('./data/traffic/51_12_1.csv', header=None)

# data2.plot()
# plt.show()
#
# plot_acf(data2)
# plt.show()

data1 = np.squeeze(data1.to_numpy())
data2 = np.squeeze(data2.to_numpy())

def normalize(conts):
    scale = np.max(conts) - np.min(conts)
    res = (conts - np.min(conts)) / scale
    return res

# s = min(len(data1), len(data2))
s = 7

print(pearsonr(data1[:-s], data1[s:]))

normalized_data2 = normalize(data2[:s])
print(pearsonr(data2[:-s], data2[s:]))


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

# Data

data = pd.read_csv('./data/traffic/51_6_7.csv', header=None)

N = len(data)

# --- STL decomposition ---
# Try STL with a range of possible periods; choose based on minimal residuals
p = 7

stl = STL(data, period=p, robust=True)
res = stl.fit()

print(pearsonr(res.seasonal[:-s], res.seasonal[s:]))

data = pd.read_csv('./data/traffic/51_12_1.csv', header=None)

p = 7

stl = STL(data, period=p, robust=True)
res = stl.fit()

print(pearsonr(res.seasonal[:-s], res.seasonal[s:])) 