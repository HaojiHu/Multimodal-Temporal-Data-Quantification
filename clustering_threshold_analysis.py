import matplotlib.pyplot as plt

x = [40, 50, 60, 70, 80, 90]

y1 = [0.8197, 0.8227, 0.8284, 0.8345, 0.8458, 0.8587]
y2 = [0.7679, 0.7669, 0.7880, 0.7854, 0.8110, 0.8230]
y3 = [0.8953, 0.8965, 0.9038, 0.9088, 0.9159, 0.9234]
y4 = [0.7899, 0.7964, 0.8032, 0.8102, 0.8208, 0.8314]

plt.plot(x, y1, marker="o", label="CatBoostRegressor")
plt.plot(x, y2, marker="s", label="DeepAR")
plt.plot(x, y3, marker="^", label="FMTimeSeries")
plt.plot(x, y4, marker="D", label="Chronos-2")

plt.xlabel("Percentile")
plt.ylabel("NDCG")
plt.title("NDCG by Percentile in M5 dataset")

plt.xticks(x, [f"{p}%" for p in x])
plt.xlim(40, 90)

plt.legend()
plt.grid(True)
# plt.show()

plt.savefig('clustering_analysis_NDCG.pdf', dpi=600, bbox_inches='tight', format='pdf', transparent=True)

plt.close()

import matplotlib.pyplot as plt

x = [40, 50, 60, 70, 80, 90]

y1 = [31, 36, 42, 46, 52, 62]
y2 = [50, 56, 63, 67, 71, 75]
y3 = [24, 26, 35, 37, 43, 46]
y4 = [46, 62, 62, 66, 68, 72]

plt.plot(x, y1, marker="o", label="CatBoostRegressor")
plt.plot(x, y2, marker="s", label="DeepAR")
plt.plot(x, y3, marker="^", label="FMTimeSeries")
plt.plot(x, y4, marker="D", label="Chronos-2")

plt.xlabel("Percentile")
plt.ylabel("Win")
plt.title("Win by Percentile in M5 dataset")

plt.xticks(x, [f"{p}%" for p in x])
plt.xlim(40, 90)

plt.legend()
plt.grid(True)
# plt.show()

plt.savefig('clustering_analysis_win.pdf', dpi=600, bbox_inches='tight', format='pdf', transparent=True)