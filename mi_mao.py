import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from scipy.special import psi
from entropy_mao import entropy_estimate_1d_continuous, entropy_estimate_1d_discrete

size = 50
week = 2
t = np.linspace(1, size, size)
weekday = np.mod(t, week) + 1
noise_mag = weekday * 5.
mu = -40.0 + 60.0 * weekday
signal = mu + noise_mag * np.random.normal(size=t.shape)

def mutual_information(discs, conts):
    # two 1D numpy arrays of the same size
    # elements of the same position pair up
    discrete_range = set(discs)
    # print(f"discrete_range: {discrete_range}")
    groups = defaultdict(np.array)
    for d in discrete_range:
        groups[d] = conts[discs == d]
        # print(f"group with d equal to {d}, shape {groups[d].shape}")
    size = len(discs)
    mi = 0
    mi += entropy_estimate_1d_continuous(conts, islog2=True)
    # print(f"fisrt term is {mi}")
    for g in groups:
        mi -= len(groups[g]) / size * entropy_estimate_1d_continuous(groups[g], islog2=True)
    # print(f"mutual information is {mi}")
    return mi
# mi = mutual_information(weekday, signal)
# print("mi from mutual_information is ", mi)
def NMI(discs, conts):

    scale = np.max(conts) - np.min(conts)
    conts = (conts - np.min(conts)) / scale
    # disc = entropy_estimate_1d_discrete(discs, islog2=True)


    unit = 1
    for i in range(len(conts)-1):
        if unit > conts[i+1] - conts[i] > 0:
            unit = conts[i+1] - conts[i]
    unit = unit / len(conts)


    mi = mutual_information(discs, conts)



    return math.sqrt(1 - math.exp((-2*mi)/(unit+1)))

if __name__ == "__main__":
    plt.figure()
    plt.plot(t, weekday)
    plt.savefig("weekday.png")
    plt.close()

    plt.figure()
    plt.plot(t, signal)
    plt.savefig("signal.png")
    plt.close()

    plt.figure(dpi=300)
    plt.scatter(weekday, signal, s=1)
    plt.xlim(0, 4)
    plt.grid()
    plt.savefig("distribution.png")
    plt.close()

    pairs = np.zeros(shape=(size, 2))
    pairs[:, 0] = signal
    pairs[:, 1] = weekday

    pairs = pairs.tolist()
    pairs.sort()
    pairs = np.array(pairs)

    discrete_range = set(pairs[:, 1].flatten())
    print(f"discrete_range: {discrete_range}")
    groups = defaultdict(np.array)
    for d in discrete_range:
        groups[d] = pairs[pairs[:, 1] == d][:, 0]
        print(f"group with d equal to {d}, shape {groups[d].shape}")

    # find mutual information
    mi = 0
    mi += entropy_estimate_1d_continuous(pairs[:, 0], islog2=True)
    print(f"fisrt term is {mi}")
    for g in groups:
        mi -= len(groups[g]) / size * entropy_estimate_1d_continuous(groups[g], islog2=True)
    print(f"mutual information is {mi}")

    # Define two sets of points
    blue_points = groups[1.0]  # Set 1 (blue points)
    red_points = groups[2.0]  # Set 2 (red points)

    # Create a figure
    plt.figure(dpi=300)
    fig, ax = plt.subplots(dpi=300)

    # Plot blue points on the axis
    ax.scatter(blue_points, [0] * len(blue_points), color='b', s=5,
               label=r'group $\mathcal{A}$')  # 'b' is for blue, s is size

    # Plot red points on the axis
    ax.scatter(red_points, [0] * len(red_points), color='r', s=5, label=r'group $\mathcal{B}$')  # 'r' is for red

    # Adjust the y-axis to effectively hide it and focus on the horizontal dimension
    ax.set_yticks([])  # No y ticks
    ax.spines['left'].set_visible(False)  # Hide the left spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['top'].set_visible(False)  # Hide the top spine

    # Optionally, you can adjust the x-axis limits if needed
    ax.set_xlim(0, max(red_points + blue_points) + 1)

    # Add a legend to identify the sets
    ax.legend()

    # Show the plot
    plt.savefig("distribution.png")
    plt.close()
