import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# file_path = './data/traffic/51_6_7.csv'  # Replace with your file path
file_path = './data/traffic/51_12_1.csv'
data = pd.read_csv(file_path, header=None)

# data[0].values[:] = 2
# Ensure the data is in the correct format (single column)
time_series_data = data[0].values

time_series_data = time_series_data - np.mean(time_series_data)

# Perform Fourier Transform
fft_values = np.fft.fft(time_series_data)
fft_frequencies = np.fft.fftfreq(len(fft_values))

# Only take the positive half of the frequencies and values for plotting
positive_frequencies = fft_frequencies[:len(fft_frequencies)//2]
positive_amplitudes = np.abs(fft_values)[:len(fft_values)//2]

# Plot the periodogram
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_amplitudes)
plt.title('Fourier Transform - Periodogram')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


idx = np.argsort(positive_amplitudes)[::-1]   # 按幅值从大到小的索引

dom_freqs = positive_frequencies[idx[:5]]
dom_amps  = positive_amplitudes[idx[:5]]
dom_T     = 1.0 / dom_freqs

print("Dominant freqs:", dom_freqs)
print("Periods:", dom_T)
print("Amplitudes:", dom_amps)