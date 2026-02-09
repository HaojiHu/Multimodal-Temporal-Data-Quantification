import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import mixed
from ross_mi import *
import normalize_clustered_mi

matplotlib.use("Agg")
df = pd.read_csv('./data/traffic/51_6_7.csv', header=None)
# df = pd.read_csv('./data/traffic/51_12_1.csv', header=None)



period = 7
val = set()

# print(df[0].values*1000000000)
# for i in range(len(df[0].values)):
#     df[0].iloc[i] = int(df[0].iloc[i]%100000)


df.plot()
plt.show()

# data = pd.DataFrame(val)
# data.plot()
# data.show()

df.columns = ['traffic volume']
filtered_df = df

# df.plot(title="Jun.-Jul.")
df.plot(title="Dec.-Jan.")

# plt.show()
plt.savefig('12_1.pdf', dpi=600, bbox_inches='tight', format='pdf', transparent=True)

# Extracting the indices as an array
indices_array = filtered_df.index.to_numpy()
first_column_array = filtered_df.iloc[:, 0].to_numpy()

first_column_array = first_column_array



# build groups
groups = dict()
size = len(indices_array)
days = []
for i in range(size):
    label = (indices_array[i]) % period
    days.append(label)
    if label not in groups:
        groups[label] = [first_column_array[i]]
    else:
        groups[label].append(first_column_array[i])


mi = normalize_clustered_mi.NormalizedClusteredMI(days, first_column_array, 3)
print(f"Proposed mutual information is {mi}")


mi, V = discrete_continuous_info(days, [first_column_array])
print(f"ross mutual information is {mi}")

n = len(days)
x = np.asarray(days).reshape((int(n),1))
y = np.asarray(first_column_array).reshape((int(n),1))

cmi = mixed.Mixed_KSG(x, y)
print(f"Mixture mutual information is {cmi}")


plt.figure()
plt.scatter(days, first_column_array)
plt.xlabel('days')
plt.ylabel('traffic')
plt.title('Scatter Plot')
plt.savefig("scatter_6_7.png")