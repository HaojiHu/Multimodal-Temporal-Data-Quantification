import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Original data
data = np.array([
120388.0,120444.0,119251.0,99527.0,92491.0,110251.0,119888.0,118943.0,121652.0,
112119.0,95954.0,90304.0,107959.0,116014.0,116205.0,117096.0,110104.0,92830.0,
87210.0,108770.0,118460.0,115634.0,117932.0,109072.0,96026.0,85902.0,109844.0,
117709.0,118480.0,117694.0,106354.0,76287.0,74133.0,67398.0,89575.0,110499.0,
112927.0,108129.0,97561.0,81177.0,75223.0,103879.0,119150.0,118791.0,108201.0,
4602.0,0.0,78595.0,77176.0,114488.0,108169.0,107802.0,80567.0,78692.0,100672.0,
92709.0,97468.0,102160.0,106240.0,90761.0
])

N = len(data)
t = np.arange(N)

# Remove mean for fitting pure periodic part
mean_val = data.mean()
y = data - mean_val

# Construct sin & cos with period = 7
period = 7.0
omega = 2 * np.pi / period
X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])  # [N x 2]

# Least squares fit: y ≈ a*cos + b*sin
coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
a, b = coeffs

# Reconstructed pure period-7 waveform
pure_periodic = X @ coeffs + mean_val  # add back mean to match scale


import pandas as pd

# dates
start_date = "2023-06-01"
dates = pd.date_range(start=start_date, periods=N, freq="D")

# Plot original vs pure period-7 component
plt.figure(figsize=(12,8))
plt.plot(dates, data, label="Traffic volume")
plt.plot(dates, pure_periodic, label="Pure Period-7 Component", ls='--')
plt.title("Jun-Jul", fontsize=26)
plt.xlabel("Time Index", fontsize=26)
plt.ylabel("Volume", fontsize=26)
plt.legend(fontsize=26)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()

ax = plt.gca()


ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))


ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
# plt.show()
plt.savefig('6_7.pdf', dpi=600, bbox_inches='tight')


data = np.array([
107261.0,106406.0,90552.0,77476.0,103849.0,108052.0,104246.0,110814.0,109221.0,
88132.0,76377.0,108634.0,111123.0,105875.0,95635.0,106419.0,88123.0,80022.0,
97304.0,112109.0,66254.0,74831.0,73495.0,72932.0,52820.0,71906.0,91447.0,
96355.0,97237.0,97859.0,64780.0,78231.0,58941.0,59782.0,95298.0,104969.0,
84913.0,73604.0,104246.0,105631.0,100736.0,108661.0,102523.0,84869.0,74118.0,
88352.0,34728.0
])

N = len(data)
t = np.arange(N)

# Remove mean
mean_val = data.mean()
y = data - mean_val

# Force period = 7
period = 7.0
omega = 2 * np.pi / period

# Construct cosine & sine basis
X = np.column_stack([np.cos(omega * t), np.sin(omega * t)])

# Least squares fit
coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

# Reconstruct pure periodic component
pure_periodic = X @ coeffs + mean_val

start_date = "2023-12-01"
dates = pd.date_range(start=start_date, periods=N, freq="D")


# Plot result
plt.figure(figsize=(12,8))
plt.plot(dates, data, label="Traffic volume")
plt.plot(dates, pure_periodic, label="Pure Period-7 Component", ls='--')



plt.title("Dec-Jan", fontsize=26)
plt.xlabel("Time Index", fontsize=26)
plt.ylabel("Volume", fontsize=26)
plt.legend(fontsize=26)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.tight_layout()

ax = plt.gca()


ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))


ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))

plt.savefig('12_1.pdf', dpi=600, bbox_inches='tight')