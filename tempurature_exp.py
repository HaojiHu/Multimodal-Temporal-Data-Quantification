import mixed
import pandas as pd
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
from ross_mi import *
import normalize_clustered_mi
import mixture_mi
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# random.seed(0)
path = './data/tdmi/'

file = 'minneapolis_2023_day_high_night_low.csv'

new_df = pd.read_csv(path + file)

merged_list = (
    new_df[["day_high_f", "night_low_f"]]
    .to_numpy()
    .reshape(-1)
    .tolist()
)


df = new_df

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


day_df = df[["date", "day_high_f"]].rename(columns={"day_high_f": "temp_f"})
day_df["time"] = day_df["date"] + pd.Timedelta(hours=12)
day_df["phase"] = "day"

night_df = df[["date", "night_low_f"]].rename(columns={"night_low_f": "temp_f"})
night_df["time"] = night_df["date"] + pd.Timedelta(hours=0)
night_df["phase"] = "night"

# 合并成两行记录
out = pd.concat([night_df, day_df], ignore_index=True)

# 只保留你想要的列，并排序
out = out[["time", "phase", "temp_f"]].sort_values("time").reset_index(drop=True)


out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

df = out


discs = []
conts = []





y = merged_list



fig, ax = plt.subplots()
ax.plot(y)

import numpy as np

# 每个月的天数
days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]

# 累积天数
cum_days = np.cumsum([0] + days_in_month[:-1])

# 每天 2 个点（12h）
month_positions = [d * 2 for d in cum_days]

month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

plt.xlabel('12-hourly time step')
plt.ylabel('temperature (F)')
plt.title('Air Temperature in Minneapolis in 2023')

ax.set_xticks(month_positions)
ax.set_xticklabels(month_labels)
ax.set_xlabel('Month')



t = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")





dn = ["Night" if 6 <= ts.hour < 18 else "Day" for ts in t]
# inset
axins = inset_axes(
    ax,
    width="30%",      # inset 宽度（相对主图）
    height="30%",     # inset 高度
    loc="lower center",
    bbox_to_anchor = (0.05, 0.1, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
)



x1, x2 = 361, 484  # zoom 区间（主图 index）

# 区间数据
y_win = np.asarray(y[x1:x2+1], dtype=float)
dn_win = np.asarray(dn[x1:x2+1])

night_vals = y_win[dn_win == "Night"]
day_vals   = y_win[dn_win == "Day"]



night_mean = np.mean(night_vals) if night_vals.size > 0 else np.nan
day_mean   = np.mean(day_vals)   if day_vals.size > 0 else np.nan

# 样本标准差；如果样本数<=1，就用 0（你也可以改成 np.nan）
night_std = np.std(night_vals, ddof=1) if night_vals.size > 1 else 0.0
day_std   = np.std(day_vals, ddof=1)   if day_vals.size > 1 else 0.0

labels = ["Night", "Day"]
means  = np.array([night_mean, day_mean], dtype=float)
stds   = np.array([night_std, day_std], dtype=float)
xpos   = np.arange(len(labels))

# 重画 inset
axins.cla()

# bar 图：柱高=均值；误差棒=标准差
axins.bar(xpos, means, yerr=stds, capsize=4)

axins.set_xticks(xpos)
axins.set_xticklabels(labels)
# axins.set_ylabel("Temp (F)", fontsize=8)

axins.text(
    0.5, 0.92,                 # 子图内部坐标（0~1）
    "Mean ± STD (Jul–Aug)",
    transform=axins.transAxes,
    ha="center",
    va="top",
    fontsize=8,
    fontweight="bold"
)



axins.tick_params(axis='x', labelsize=8)
axins.tick_params(axis='y', labelsize=8)
axins.grid(True, axis='y', alpha=0.3)



# （可选）标注样本量

# y 轴留白（可选）
finite = np.isfinite(means)
if finite.any():
    pad = np.nanmax(stds[finite]) * 4.5
    pad = pad if np.isfinite(pad) and pad > 0 else 2.0
    axins.set_ylim(
        np.nanmin(means[finite]) - pad,
        np.nanmax(means[finite]) + 0.8*pad
    )

axins2 = inset_axes(
    ax,
    width="30%",
    height="30%",
    loc="upper left",          # 左上角
    borderpad=1.0
)

x1b, x2b = 0, 118  # 0-60 步区间

y_win2  = np.asarray(y[x1b:x2b+1], dtype=float)
dn_win2 = np.asarray(dn[x1b:x2b+1])

night_vals2 = y_win2[dn_win2 == "Night"]
day_vals2   = y_win2[dn_win2 == "Day"]

night_mean2 = np.mean(night_vals2) if night_vals2.size > 0 else np.nan
day_mean2   = np.mean(day_vals2)   if day_vals2.size > 0 else np.nan

night_std2 = np.std(night_vals2, ddof=1) if night_vals2.size > 1 else 0.0
day_std2   = np.std(day_vals2, ddof=1)   if day_vals2.size > 1 else 0.0

labels2 = ["Night", "Day"]
means2  = np.array([night_mean2, day_mean2], dtype=float)
stds2   = np.array([night_std2, day_std2], dtype=float)
xpos2   = np.arange(len(labels2))

axins2.cla()
axins2.bar(xpos2, means2, yerr=stds2, capsize=4)

axins2.set_xticks(xpos2)
axins2.set_xticklabels(labels2)
# axins2.set_ylabel("Temp (F)", fontsize=8)

axins2.text(
    0.5, 0.92,                 # 子图内部坐标（0~1）
    "Mean ± STD (Jan-Feb)",
    transform=axins2.transAxes,
    ha="center",
    va="top",
    fontsize=8,
    fontweight="bold"
)

axins2.tick_params(axis='x', labelsize=8)
axins2.tick_params(axis='y', labelsize=8)
axins2.grid(True, axis='y', alpha=0.3)



finite2 = np.isfinite(means2)
if finite2.any():
    pad2 = np.nanmax(stds2[finite2]) * 2.5
    pad2 = pad2 if np.isfinite(pad2) and pad2 > 0 else 2.0
    axins2.set_ylim(np.nanmin(means2[finite2]) - pad2, np.nanmax(means2[finite2]) + pad2)



x1a = x1
x2a = x2

# 高亮阴影（alpha 越大越显眼）
ax.axvspan(x1a, x2a, alpha=0.15)
ax.axvspan(x1b, x2b, alpha=0.15)

# （可选）画边界竖线更清晰
ax.axvline(x1a, linewidth=1, alpha=0.6)
ax.axvline(x2a, linewidth=1, alpha=0.6)
ax.axvline(x1b, linewidth=1, alpha=0.6)
ax.axvline(x2b, linewidth=1, alpha=0.6)


ymin, ymax = ax.get_ylim()

ax.plot(y, color="#4C5C68", linewidth=1.5)


# inset bar
axins.bar(
    xpos, means,
    yerr=stds,
    capsize=4,
    color=["#6B728E", "#C08457"],
    edgecolor="black",
    linewidth=0.6
)



axins2.bar(
    xpos2, means2,
    yerr=stds2,
    capsize=4,
    color=["#6B728E", "#C08457"],
    edgecolor="black",
    linewidth=0.6
)

for a in (axins, axins2):
    a.tick_params(axis='both', which='both',
                  direction='in',   # ticks 朝里
                  pad=-16,           # 关键：label 往里推（负数=往里）
                  top=True, right=True)  # 上/右也显示 ticks（可选）

for a, xpos_, labels_ in [
    (axins,  xpos,  labels),
    (axins2, xpos2, labels2)
]:
    a.set_xlim(-0.8, len(xpos_) - 0.2)

plt.show()
# plt.savefig('temperature.png', dpi=600, bbox_inches='tight')

conts = np.array(y)






def get_time_two_month_day_night_event_sequence(df):
    discs = []
    for index, row in df.iterrows():
        time = row['time']
        date_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        try:
            time_span = date_obj.month // 2
            if date_obj.hour == 0:
                discs.append('N'+str(time_span))
            else:
                discs.append('D'+str(time_span))
        except:
            print('Issue')
            discs.append('X'+str(time_span))
    return np.array(discs)

def get_time_two_month_event_sequence(df):
    discs = []
    for index, row in df.iterrows():
        time = row['time']
        date_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        time_span = date_obj.month // 2
        discs.append(str(time_span))

    return np.array(discs)

def get_time_day_night_event_sequence(df):
    discs = []
    for index, row in df.iterrows():
        time = row['time']
        date_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        try:
            time_span = date_obj.month // 2
            if date_obj.hour == 0:
                discs.append('N')
            else:
                discs.append('D')
        except:
            print('Issue')
            discs.append('X'+str(time_span))
    return np.array(discs)



def symbol2id(disc):
    res = []
    symbol_dict = {}
    idx = 1
    for i, s in enumerate(disc):
        if s not in symbol_dict:
            symbol_dict[s] = idx
            idx += 1
        symbol_id = symbol_dict[s]
        res.append(symbol_id)

    return res




print("Two_month_day_night_event_sequence experiment:")

discs = get_time_two_month_day_night_event_sequence(df)
reid = symbol2id(discs)

tdmi = mixture_mi.NMI(discs, conts)
print('Proposed: ', tdmi)

n = len(discs)
x = np.asarray(reid)
y = np.asarray(conts)
cmi = mixed.Mixed_KSG(x, y)
print('Mixture:, ', cmi)
ross, V = discrete_continuous_info(discs, [conts])
print('Ross: ', ross)


print("Two_month_event_sequence experiment:")

discs = get_time_two_month_event_sequence(df)
reid = symbol2id(discs)

tdmi = mixture_mi.NMI(discs, conts)
print('Proposed: ', tdmi)

n = len(discs)
x = np.asarray(reid)
y = np.asarray(conts)
cmi = mixed.Mixed_KSG(x, y)
print('Mixture:, ', cmi)
ross, V = discrete_continuous_info(discs, [conts])
print('Ross: ', ross)


print("Day_night_event_sequence experiment:")

discs = get_time_day_night_event_sequence(df)
reid = symbol2id(discs)

tdmi = mixture_mi.NMI(discs, conts)
print('Proposed: ', tdmi)

n = len(discs)
x = np.asarray(reid)
y = np.asarray(conts)
cmi = mixed.Mixed_KSG(x, y)
print('Mixture:, ', cmi)
ross, V = discrete_continuous_info(discs, [conts])
print('Ross: ', ross)
