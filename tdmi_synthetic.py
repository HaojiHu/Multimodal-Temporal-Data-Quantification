import random
import bisect
import mixed
import matplotlib.pyplot as plt
from ross_mi import *
from scipy.stats import pearsonr
from single_modality_processing import *
import mixture_mi


# import matplotlib
# matplotlib.use('Agg')

digit_precision = 3

sample_num = 1000

def generate_label(cdf, labels):
    # Generate a random floating-point number between 0 and 1
    r = random.random()

    # Use binary search to find the right interval for the random number
    index = bisect.bisect_left(cdf, r)

    # Map the index to the corresponding label
    return labels[index]

def generate_cont(label, mapping):
    b,e = mapping[label]
    v = random.uniform(b, e)
    return round(v, 5)

def generate_mixture(label, mapping, digit=1):
    b,e = mapping[label]
    v = random.uniform(b, e)
    return round(v, digit)

tau = 5
cdf = [1./3., 2./3., 1.]
# labels = {0: 'A', 1: 'B', 2: 'C'}
labels = {0: 1, 1: 2, 2: 3}
mapping = {1: (-10, 0), 2: (-5, 5), 3: {0, 10}}

discs = []
conts = []
for i in range(tau):
    disc = generate_label(cdf, labels)
    # cont = generate_cont(disc, mapping)
    cont = generate_mixture(disc, mapping, digit_precision)
    conts.append(cont)
for i in range(sample_num):
    disc = generate_label(cdf, labels)
    # cont = generate_cont(disc, mapping)
    cont = generate_mixture(disc, mapping, digit_precision)
    discs.append(disc)
    conts.append(cont)
for i in range(tau):
    disc = generate_label(cdf, labels)
    discs.append(disc)
discs = np.array(discs)
conts = np.array(conts)

# discs = discs + np.random.normal(0, 0.01, size=len(discs))
# conts = conts + np.random.normal(0, 0.01, size=len(conts))
# plt.scatter(x_jittered, y_jittered, alpha=0.5)

# plt.figure(dpi=200)
# plt.scatter(discs[:-tau], conts[tau:], s=10, alpha=0.5)
# plt.xlabel('Discrete symbols for temporal event sequence')
# plt.ylabel('Continuous values for time series')
# plt.show()
# plt.title('2D Data Scatter Plot')

scale = np.max(conts) - np.min(conts)
conts = (conts - np.min(conts)) / scale
# disc = entropy_estimate_1d_discrete(discs, islog2=True)


#print(discs, conts)
size = len(discs)
proposed = ''
compared = ''
ross = ''
correlation = ''
pnmi = ''
for tau in range(10):
    # tdmi = mixture_mi.NMI(discs[:size-tau], conts[tau:])
    tdmi = mixture_mi.mixture_mi(discs[:size - tau], conts[tau:])
    # tdmi = mi.NMI(discs[:size-tau], conts[tau:])
    # print(f"tdmi with tau {tau} is {tdmi}")
    proposed += str(round(tdmi,4)) + '  '

    n = len(discs[:size-tau])
    x = np.asarray(discs[:size-tau])
    y = np.asarray(conts[tau:])

    cmi = mixed.Mixed_KSG(x, y)
    # print(f"2 tdmi with tau {tau} is {cmi}")
    compared += str(round(cmi,4))+'  '
    # cmi = mixed.KSG(x, y)
    # print(f"3 tdmi with tau {tau} is {cmi}")
    #
    tdmi, V = discrete_continuous_info(discs[:size - tau], [conts[tau:]])
    ross += str(round(tdmi, 4)) + '  '

    series = sequence2series(discs[:size - tau])
    correlation += str(round(pearsonr(series, conts[tau:])[0],4)) + '  '

    seq = series2sequence(conts[tau:], 5)
    pnmi += str(round(poitwise_mutual_information(discs[:size - tau], seq), 4)) + '  '

print('Series2Seq: ', pnmi)
print('Seq2Series:', correlation)
print('Ross method: ', ross)
print('Mixtures: ', compared)
print('proposed: ', proposed)

#
# print('Mean square error to ground truth:')
#
# gt = 0.6365141683
# iteration = 100
#
#
# #print(discs, conts)
# size = len(discs)
# proposed = 0
# compared = 0
# ross = 0
#
# for i in range(iteration):
#     tau = 5
#
#     tdmi = mixture_mi.mixture_mi(discs[:size-tau], conts[tau:])
#
#     proposed += (tdmi - gt)**2
#
#     n = len(discs[:size-tau])
#     x = np.asarray(discs[:size-tau])
#     y = np.asarray(conts[tau:])
#
#     cmi = mixed.Mixed_KSG(x, y)
#     # print(f"2 tdmi with tau {tau} is {cmi}")
#
#     compared += (cmi - gt)**2
#     # cmi = mixed.KSG(x, y)
#     # print(f"3 tdmi with tau {tau} is {cmi}")
#     #
#     tdmi, V = discrete_continuous_info(discs[:size - tau], [conts[tau:]])
#
#     ross += (tdmi - gt)**2
#
#
#
# print('Ross method: ', round(ross / iteration, 6))
# print('Mixtures: ', round(compared / iteration, 6))
# print('proposed: ', round(proposed / iteration, 6))
