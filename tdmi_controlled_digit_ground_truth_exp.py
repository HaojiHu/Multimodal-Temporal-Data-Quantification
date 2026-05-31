import random
import bisect
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from entropy_mao import entropy_estimate_1d_continuous, entropy_estimate_1d_discrete
from ross_mi import *
from scipy.stats import pearsonr
from single_modality_processing import *
import mixture_mi
import mixed

from collections import Counter

# Table 6
observation_samples = [10000, 10000, 10000]
digit_precisions = [2, 3, 4]
ground_truth_sample_nums = [1000, 10000, 100000]
dys = [0.001, 0.0001, 0.00001]

# # Table 7
# observation_samples = [100, 1000, 10000]
# digit_precisions = [2, 2, 2]
# ground_truth_sample_nums = [1000, 10000, 100000]
# dys = [0.001, 0.001, 0.001]

for i in range(3):

    observation_sample = observation_samples[i]

    digit_precision = digit_precisions[i]
    sample_num = ground_truth_sample_nums[i]
    dy = dys[i]

    print('----------------------------------------------------------------------------')
    print("observation_sample: ", observation_sample)
    print('digit_precision: ', digit_precision)
    print('ground_truth_sample_num: ', sample_num)
    print('dy: ', dy)
    print('')

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

    # def generate_mixture(label, mapping, digit=1):
    #     b,e = mapping[label]
    #     v = random.uniform(b, e)
    #     return round(v, digit)
    #
    # mapping = {'A': (-10, 0), 'B': (-5, 5), 'C': {0, 10}}

    def generate_mixture(label, mapping, digit=1):
        b,e, form = mapping[label]
        if form =="normal":
            v = random.gauss(b, e)
        elif form == "uniform":
            v = random.uniform(b, e)
        return round(v, digit)

    mu = 0.7
    var = 0.02

    mapping = {'A': (-mu, var, "normal"), 'B': (0, var, "normal"), 'C': (mu, var, "normal")}

    cdf = [1./3., 2./3., 1.]
    labels = {0: 'A', 1: 'B', 2: 'C'}


    gt_tau = 5
    reverse_mapping = {v: k for k, v in labels.items()}

    proposed = ''
    compared = ''
    ross = ''
    correlation = ''
    pnmi = ''

    proposed_mse = ''
    compared_mse = ''
    ross_mse = ''

    gt_mi = ''

    tau_list = ''

    discs = []
    conts = []

    gt_discs = []
    gt_conts = []


    for i in range(gt_tau):
        disc = generate_label(cdf, labels)
        # cont = generate_cont(disc, mapping)
        cont = generate_mixture(disc, mapping, digit_precision)
        gt_conts.append(cont)
    for i in range(sample_num):
        disc = generate_label(cdf, labels)
        # cont = generate_cont(disc, mapping)
        cont = generate_mixture(disc, mapping, digit_precision)
        gt_discs.append(disc)
        gt_conts.append(cont)
    for i in range(gt_tau):
        disc = generate_label(cdf, labels)
        gt_discs.append(disc)

    gt_discs = [reverse_mapping[d] for d in gt_discs]

    for i in range(gt_tau):
        disc = generate_label(cdf, labels)
        # cont = generate_cont(disc, mapping)
        cont = generate_mixture(disc, mapping, digit_precision)
        conts.append(cont)
    for i in range(observation_sample):
        disc = generate_label(cdf, labels)
        # cont = generate_cont(disc, mapping)
        cont = generate_mixture(disc, mapping, digit_precision)
        discs.append(disc)
        conts.append(cont)
    for i in range(gt_tau):
        disc = generate_label(cdf, labels)
        discs.append(disc)

    discs = [reverse_mapping[d] for d in discs]

    gt_discs = np.array(gt_discs)
    gt_conts = np.array(gt_conts)

    gt_size = len(gt_discs)

    # freq = Counter(conts)
    #
    # print(freq)


    discs = np.array(discs)
    conts = np.array(conts)


    size = len(discs)



    sync_disc = gt_discs
    sync_cont = gt_conts

    ob_disc = discs
    ob_cont = conts


    for tau in range(10):

        tau_list += 'tau=' + str(tau) + '  '

        gtdiscs = sync_disc[:gt_size-tau]
        gtconts = sync_cont[tau:]


        # print(f"discretizing conts using dy = {dy}")
        discretized_conts = gtconts
        H_disc = entropy_estimate_1d_discrete(np.array(gtdiscs),islog2=False)
        H_cont = entropy_estimate_1d_discrete(discretized_conts, islog2=False)

        H_XY = 0
        groups = dict()
        for x, y in zip(gtdiscs, discretized_conts):
            if (x,y) in groups:
                groups[(x,y)] += 1
            else:
                groups[(x,y)] = 1

        for k in groups.keys():
            p = groups[k]/len(gtdiscs)
            H_XY += -p * np.log(p)
        # print(f"H_X: {H_disc}\n")
        # print(f"H_Y: {H_cont}\n")
        # print(f"H_XY_A: {H_XY}\n")
        MI = H_disc + H_cont - H_XY
        # print(f"MI: {MI}\n")

        gt = MI

        gt_mi += str(round(gt, 6)) + '  '

        # disc = entropy_estimate_1d_discrete(discs, islog2=True)

        # print(discs, conts)

        discs = ob_disc[:size-tau]
        conts = ob_cont[tau:]

        # tdmi = mixture_mi.NMI(discs[:size-tau], conts[tau:])
        tdmi = mixture_mi.mixture_mi(discs, conts)
        # tdmi = mi.NMI(discs[:size-tau], conts[tau:])
        # print(f"tdmi with tau {tau} is {tdmi}")
        proposed += str(round(tdmi, 4)) + '  '

        n = len(discs)
        x = np.asarray(discs)
        y = np.asarray(conts)


        cmi = mixed.Mixed_KSG(x, y)
        # print(f"2 tdmi with tau {tau} is {cmi}")
        compared += str(round(cmi, 4)) + '  '
        # cmi = mixed.KSG(x, y)
        # print(f"3 tdmi with tau {tau} is {cmi}")
        #
        tdmi, V = discrete_continuous_info(discs, [conts])
        ross += str(round(tdmi, 4)) + '  '

        series = sequence2series(discs)
        correlation += str(round(pearsonr(series, conts)[0], 4)) + '  '

        seq = series2sequence(conts, 5)
        pnmi += str(round(poitwise_mutual_information(discs, seq), 4)) + '  '




        tdmi = mixture_mi.mixture_mi(discs, conts)

        proposed_mse += str(round((tdmi - gt) ** 2, 5)) + '  '

        n = len(discs)
        x = np.asarray(discs)
        y = np.asarray(conts)

        cmi = mixed.Mixed_KSG(x, y)
        # print(f"2 tdmi with tau {tau} is {cmi}")

        compared_mse += str(round((cmi - gt) ** 2, 5)) + '  '
        # cmi = mixed.KSG(x, y)
        # print(f"3 tdmi with tau {tau} is {cmi}")
        #
        tdmi, V = discrete_continuous_info(discs, [conts])

        ross_mse += str(round((tdmi - gt) ** 2 , 5)) + '  '

    print('Tau: ', tau_list)

    print('ground truth mi: ', gt_mi)

    print('')

    print('Series2Seq: ', pnmi)
    print('Seq2Series:', correlation)
    print('Ross method: ', ross)
    print('Mixtures: ', compared)
    print('proposed: ', proposed)

    print('')

    print('Ross method MSE: ', ross_mse)
    print('Mixtures MSE: ', compared_mse)
    print('proposed MSE: ', proposed_mse)


import numpy as np
import matplotlib.pyplot as plt

# parameters
mu = 0.7
var = 0.02

gaussians = [
    (-mu, var),
    (0.0, var),
    (mu, var),
]

x = np.linspace(-1.5, 1.5, 1000)

for idx, (mean, var) in enumerate(gaussians):
    std = np.sqrt(var)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    plt.plot(x, y, label=f"Caused by event {idx}")

plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("The Gaussian Distributions for Values in Time Series")
plt.legend(    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})
plt.savefig("data_distribution.png", dpi=500, bbox_inches="tight")