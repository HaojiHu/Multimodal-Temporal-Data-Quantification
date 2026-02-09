import random
import bisect
import numpy as np
from entropy_mao import entropy_estimate_1d_continuous, entropy_estimate_1d_discrete

from ross_mi import *
from scipy.stats import pearsonr
from single_modality_processing import *
import mixture_mi
import mixed

digit_precision = 3

def generate_label(cdf, labels):
    # Generate a random floating-point number between 0 and 1
    r = random.random()

    # Use binary search to find the right interval for the random number
    index = bisect.bisect_left(cdf, r)

    # Map the index to the corresponding label
    return labels[index]

def generate_cont(label, mapping):
    b,e = mapping[label]
    return random.uniform(b, e)

def generate_mixture(label, mapping, digit=1):
    b,e = mapping[label]
    v = random.uniform(b, e)
    return round(v, digit)

cdf = [1./3., 2./3., 1.]
labels = {0: 'A', 1: 'B', 2: 'C'}
mapping = {'A': (-10, 0), 'B': (-5, 5), 'C': {0, 10}}
reverse_mapping = {v: k for k, v in labels.items()}

discs = []
conts = []
# produce (disc, cont)
for i in range(10000):
    disc = generate_label(cdf, labels)
    # cont = generate_cont(disc, mapping)
    cont = generate_mixture(disc, mapping, digit_precision)
    discs.append(disc)
    conts.append(cont)
# reverse mapping to get discrete values
discs = [reverse_mapping[d] for d in discs]



deltas = [0.001]

for dy in deltas:
    discs = np.array(discs)
    conts = np.array(conts)


    scale = np.max(conts) - np.min(conts)
    conts = (conts - np.min(conts)) / scale

    print(f"discretizing conts using dy = {dy}")
    discretized_conts = conts
    H_disc = entropy_estimate_1d_discrete(np.array(discs),islog2=False)
    H_cont = entropy_estimate_1d_discrete(discretized_conts, islog2=False)

    H_XY = 0
    groups = dict()
    for x, y in zip(discs, discretized_conts):
        if (x,y) in groups:
            groups[(x,y)] += 1
        else:
            groups[(x,y)] = 1

    for k in groups.keys():
        p = groups[k]/len(discs)
        H_XY += -p * np.log(p)
    # print(f"H_X: {H_disc}\n")
    # print(f"H_Y: {H_cont}\n")
    # print(f"H_XY_A: {H_XY}\n")
    MI = H_disc + H_cont - H_XY
    print(f"MI: {MI}\n")

    gt = MI


    # disc = entropy_estimate_1d_discrete(discs, islog2=True)

    # print(discs, conts)
    size = len(discs)
    proposed = ''
    compared = ''
    ross = ''
    correlation = ''
    pnmi = ''
    for tau in [0]:
        # tdmi = mixture_mi.NMI(discs[:size-tau], conts[tau:])
        tdmi = mixture_mi.mixture_mi(discs[:size - tau], conts[tau:])
        # tdmi = mi.NMI(discs[:size-tau], conts[tau:])
        # print(f"tdmi with tau {tau} is {tdmi}")
        proposed += str(round(tdmi, 4)) + '  '

        n = len(discs[:size - tau])
        x = np.asarray(discs[:size - tau])
        y = np.asarray(conts[tau:])

        cmi = mixed.Mixed_KSG(x, y)
        # print(f"2 tdmi with tau {tau} is {cmi}")
        compared += str(round(cmi, 4)) + '  '
        # cmi = mixed.KSG(x, y)
        # print(f"3 tdmi with tau {tau} is {cmi}")
        #
        tdmi, V = discrete_continuous_info(discs[:size - tau], [conts[tau:]])
        ross += str(round(tdmi, 4)) + '  '

        series = sequence2series(discs[:size - tau])
        correlation += str(round(pearsonr(series, conts[tau:])[0], 4)) + '  '

        seq = series2sequence(conts[tau:], 5)
        pnmi += str(round(poitwise_mutual_information(discs[:size - tau], seq), 4)) + '  '

    print('Series2Seq: ', pnmi)
    print('Seq2Series:', correlation)
    print('Ross method: ', ross)
    print('Mixtures: ', compared)
    print('proposed: ', proposed)

    print('Mean square error to ground truth:')

    # gt = 0.6365141683
    iteration = 1

    # print(discs, conts)
    size = len(discs)
    proposed = 0
    compared = 0
    ross = 0

    for i in range(iteration):
        tau = 0

        tdmi = mixture_mi.mixture_mi(discs[:size - tau], conts[tau:])

        proposed += (tdmi - gt) ** 2

        n = len(discs[:size - tau])
        x = np.asarray(discs[:size - tau])
        y = np.asarray(conts[tau:])

        cmi = mixed.Mixed_KSG(x, y)
        # print(f"2 tdmi with tau {tau} is {cmi}")

        compared += (cmi - gt) ** 2
        # cmi = mixed.KSG(x, y)
        # print(f"3 tdmi with tau {tau} is {cmi}")
        #
        tdmi, V = discrete_continuous_info(discs[:size - tau], [conts[tau:]])

        ross += (tdmi - gt) ** 2

    print('Ross method: ', round(ross / iteration, 6))
    print('Mixtures: ', round(compared / iteration, 6))
    print('proposed: ', round(proposed / iteration, 6))
